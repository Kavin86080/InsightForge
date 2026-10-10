import os
import google.generativeai as genai
import json
import re

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key and os.path.exists(".env"):
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.strip().startswith("GEMINI_API_KEY="):
                    api_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    break
    except Exception:
        pass

if api_key:
    genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-flash-latest")


# ─────────────────────────────────────────────
#  Schema builder
# ─────────────────────────────────────────────

def build_schema_context(df):
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_vals = df[col].dropna().unique()[:6].tolist()
        n_unique = df[col].nunique()
        col_info.append(f"  - {col} ({dtype}, {n_unique} unique): sample = {unique_vals}")
    schema_str = "\n".join(col_info)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    date_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            sample = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else ""
            if any(c in sample for c in ["-", "/", "20", "19"]) and len(sample) >= 6:
                date_cols.append(col)
    return schema_str, numeric_cols, categorical_cols, date_cols


def build_conversation_context(conversation_history):
    """Build a compact conversation summary for the LLM."""
    if not conversation_history:
        return "No previous conversation."
    lines = []
    # Last 6 turns for context (3 back-and-forth)
    recent = conversation_history[-6:]
    for turn in recent:
        role = "User" if turn["role"] == "user" else "Analyst"
        content = turn["content"][:300]  # truncate long replies
        result = turn.get("result_summary", "")
        if result:
            lines.append(f"{role}: {content}\n  [Result preview: {result[:200]}]")
        else:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  Query parser — with conversation memory
# ─────────────────────────────────────────────

def parse_prompt(prompt, df, conversation_history=None):
    schema_str, numeric_cols, categorical_cols, date_cols = build_schema_context(df)
    row_count = len(df)
    conv_context = build_conversation_context(conversation_history or [])

    full_prompt = f"""You are a data analysis engine with memory of the conversation.

Dataset: {row_count} rows
COLUMNS:
{schema_str}

Numeric columns: {numeric_cols}
Categorical columns: {categorical_cols}
Possible date columns: {date_cols}

CONVERSATION HISTORY (use this to resolve "it", "that", "same", "also", "what about", etc.):
{conv_context}

Convert the LATEST user question into a JSON query. Detect analysis type:

1. DESCRIPTIVE: {{"analysis_type":"descriptive","group_by":"col","aggregate":{{"col":"sum|mean|count|max|min"}},"filter":[...],"sort_by":"col","ascending":false,"limit":N}}
2. TREND: {{"analysis_type":"trend","time_col":"date_col","value_col":"numeric_col","group_col":"optional","filter":[...]}}
3. COMPARATIVE: {{"analysis_type":"comparative","compare_col":"cat_col","value_col":"numeric_col","aggregate":"sum|mean","filter":[...]}}
4. RANKING: {{"analysis_type":"ranking","rank_by":"numeric_col","group_by":"cat_col","aggregate":"sum|mean","limit":N,"ascending":false,"filter":[...]}}
5. DISTRIBUTION: {{"analysis_type":"distribution","column":"numeric_col","bins":10,"group_col":"optional"}}
6. CORRELATION: {{"analysis_type":"correlation","columns":["col1","col2","col3"]}}
7. ANOMALY: {{"analysis_type":"anomaly","column":"numeric_col","group_by":"optional","method":"zscore"}}
8. PATTERN: {{"analysis_type":"pattern","column":"col","time_col":"optional","group_by":"optional"}}
9. FORECAST: {{"analysis_type":"forecast","time_col":"date_col","value_col":"numeric_col","periods":5}}

FILTER format: [{{"column":"col","op":"==|!=|>|<|>=|<=|contains","value":val}}]

Important rules:
- Use conversation history to resolve follow-up questions and references.
- If user says "now show by region" after asking about products, switch group_by to region.
- If user says "same but top 5", keep same query but set limit to 5.
- If user says "what about Electronics?", add a filter for that category.
- Only use columns that exist in the schema.
- Return ONLY valid JSON. No markdown, no explanation.

Latest user question: "{prompt}"
"""
    response = model.generate_content(full_prompt)
    text = re.sub(r"```json|```", "", response.text.strip()).strip()
    print("QUERY JSON:", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return {"analysis_type": "descriptive"}


# ─────────────────────────────────────────────
#  Chart recommender
# ─────────────────────────────────────────────

CHART_MAP = {
    "descriptive": "bar", "trend": "line", "comparative": "bar",
    "ranking": "bar", "distribution": "bar", "correlation": "scatter",
    "anomaly": "line", "pattern": "bar", "forecast": "line",
}


def recommend_chart(query, result_df):
    analysis_type = query.get("analysis_type", "descriptive")
    chart_type = CHART_MAP.get(analysis_type, "bar")
    numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
    cat_cols = result_df.select_dtypes(exclude="number").columns.tolist()

    if result_df.empty or not numeric_cols:
        return None

    x_col = cat_cols[0] if cat_cols else result_df.columns[0]
    y_cols = numeric_cols[:3]
    labels = result_df[x_col].astype(str).tolist()

    palette = [
        "rgba(37,211,102,0.85)", "rgba(91,255,216,0.85)",
        "rgba(167,139,250,0.85)", "rgba(251,191,36,0.85)",
        "rgba(248,113,113,0.85)", "rgba(52,211,153,0.85)",
    ]

    datasets = []
    if analysis_type == "correlation" and len(numeric_cols) >= 2:
        chart_type = "scatter"
        datasets = [{
            "label": f"{numeric_cols[0]} vs {numeric_cols[1]}",
            "data": [{"x": round(float(row[numeric_cols[0]]), 2), "y": round(float(row[numeric_cols[1]]), 2)}
                     for _, row in result_df.iterrows()],
            "backgroundColor": palette[0],
            "pointRadius": 5,
        }]
        labels = []
    else:
        for i, col in enumerate(y_cols):
            datasets.append({
                "label": col,
                "data": result_df[col].round(2).tolist(),
                "backgroundColor": palette[i % len(palette)],
                "borderColor": palette[i % len(palette)].replace("0.85", "1"),
                "borderWidth": 2,
                "tension": 0.4,
                "fill": analysis_type in ["trend", "forecast"],
                "pointRadius": 3 if analysis_type in ["trend", "forecast", "anomaly"] else 0,
            })

    return {
        "type": chart_type,
        "labels": labels,
        "datasets": datasets,
        "title": f"{analysis_type.replace('_',' ').title()} Analysis",
        "x_label": x_col,
        "y_label": y_cols[0] if y_cols else "",
        "analysis_type": analysis_type,
        "indexAxis": "y" if analysis_type == "ranking" else "x",
    }


# ─────────────────────────────────────────────
#  Conversational reply generator (WhatsApp analyst)
# ─────────────────────────────────────────────

def generate_chat_reply(user_message, parsed_query, result_df, df, conversation_history):
    """Generate a short, friendly, WhatsApp-style analyst reply."""
    if result_df.empty:
        return "Hmm, that query returned no results. Could you rephrase or try a different filter? 🤔"

    analysis_type = parsed_query.get("analysis_type", "descriptive")
    result_preview = result_df.head(10).to_string(index=False)
    shape = f"{len(result_df)} rows"
    conv_context = build_conversation_context(conversation_history[:-1])  # exclude current user msg

    stats = {}
    for col in result_df.select_dtypes(include="number").columns[:3]:
        stats[col] = {
            "min": round(float(result_df[col].min()), 2),
            "max": round(float(result_df[col].max()), 2),
            "mean": round(float(result_df[col].mean()), 2),
            "sum": round(float(result_df[col].sum()), 2),
        }

    reply_prompt = f"""You are a friendly, sharp data analyst chatting on WhatsApp with a colleague.
The user asked: "{user_message}"
Analysis performed: {analysis_type}
Result: {shape}

Result data (top 10 rows):
{result_preview}

Key stats: {json.dumps(stats)}

Previous conversation:
{conv_context}

Write a WhatsApp-style reply (2-4 sentences MAX). Rules:
- Be conversational and warm, like texting a smart analyst friend
- Lead with the most interesting finding or direct answer
- Use 1-2 emojis naturally (not forced)
- Mention specific numbers from the data
- If it's a follow-up question, acknowledge the context ("Building on what we saw before...")
- End with ONE optional follow-up question or suggestion to go deeper
- DO NOT use bullet points, headers, or markdown
- DO NOT say "Based on the data" or "According to the results" — just say it directly
- Keep it SHORT. Max 4 sentences.
"""

    try:
        response = model.generate_content(reply_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Reply generation error: {e}")
        # Fallback
        if stats:
            col = list(stats.keys())[0]
            s = stats[col]
            return f"Got it! {col} ranges from {s['min']} to {s['max']}, with an average of {s['mean']}. Found {len(result_df)} records. Want me to dig deeper into any specific area? 📊"
        return f"Done! Found {len(result_df)} results for your query. Check the table and chart below 👇"