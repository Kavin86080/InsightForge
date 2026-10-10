from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import json

from llm_parser import parse_prompt, recommend_chart, generate_chat_reply
from query_engine import run_query

app = Flask(__name__)
app.secret_key = "datalens-chat-2024"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── In-memory store ──
_store = {
    "df": None,
    "filename": None,
    "conversation": [],   # list of {role, content, result_summary}
}


def get_df():
    return _store["df"], _store["filename"]


def set_df(df, filename):
    _store["df"] = df
    _store["filename"] = filename
    _store["conversation"] = []   # reset chat when new file loaded


def get_dataset_summary(df, filename):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    return {
        "filename": filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_list": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "preview": df.head(5).to_html(classes="preview-table", index=False, border=0),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("csv", "xlsx", "xls"):
        return jsonify({"error": "Unsupported file type. Use CSV or Excel."}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath) if ext == "csv" else pd.read_excel(filepath)
        set_df(df, file.filename)
        summary = get_dataset_summary(df, file.filename)
        return jsonify({"success": True, "summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    df, filename = get_df()
    if df is None:
        return jsonify({"error": "No dataset loaded. Please upload a file first."}), 400

    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Add user message to conversation history
    _store["conversation"].append({
        "role": "user",
        "content": user_message,
    })

    try:
        # Parse with conversation context
        parsed_query = parse_prompt(user_message, df, _store["conversation"])

        # Run query
        result_df = run_query(df, parsed_query)

        # Chart
        chart_config = recommend_chart(parsed_query, result_df)

        # Build result summary for memory
        result_summary = result_df.head(10).to_string(index=False) if not result_df.empty else "No results"

        # Generate conversational reply (like a WhatsApp analyst)
        reply_text = generate_chat_reply(
            user_message=user_message,
            parsed_query=parsed_query,
            result_df=result_df,
            df=df,
            conversation_history=_store["conversation"],
        )

        # Table HTML
        result_html = result_df.to_html(classes="result-table", index=False, border=0) if not result_df.empty else ""

        # Add assistant reply to history (with summary for context)
        _store["conversation"].append({
            "role": "assistant",
            "content": reply_text,
            "result_summary": result_summary,
            "analysis_type": parsed_query.get("analysis_type", "descriptive"),
        })

        # Keep conversation to last 20 turns to avoid token explosion
        if len(_store["conversation"]) > 20:
            _store["conversation"] = _store["conversation"][-20:]

        return jsonify({
            "success": True,
            "reply": reply_text,
            "result_html": result_html,
            "chart": chart_config,
            "parsed_query": parsed_query,
            "rows": len(result_df),
            "cols": len(result_df.columns),
            "analysis_type": parsed_query.get("analysis_type", "descriptive"),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        err_msg = f"Sorry, I ran into an error: {str(e)}"
        _store["conversation"].append({"role": "assistant", "content": err_msg})
        return jsonify({"error": err_msg}), 500


@app.route("/suggest", methods=["GET"])
def suggest():
    df, _ = get_df()
    if df is None:
        return jsonify({"suggestions": []})

    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(exclude="number").columns.tolist()

    suggestions = []
    if categorical and numeric:
        suggestions += [
            f"What are the top 10 {categorical[0]}s by {numeric[0]}?",
            f"Show me the total {numeric[0]} by {categorical[0]}",
            f"Which {categorical[0]} has the highest {numeric[0]}?",
        ]
    if len(numeric) >= 2:
        suggestions.append(f"Is there a correlation between {numeric[0]} and {numeric[1]}?")
    if categorical:
        suggestions.append(f"What patterns do you see in {categorical[0]}?")
    if numeric:
        suggestions += [
            f"Are there any anomalies in {numeric[0]}?",
            f"Give me a summary of the entire dataset",
        ]
    if len(categorical) >= 2 and numeric:
        suggestions.append(f"Compare {numeric[0]} across different {categorical[0]}s")

    return jsonify({"suggestions": suggestions[:7]})


@app.route("/clear", methods=["POST"])
def clear_chat():
    _store["conversation"] = []
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True)