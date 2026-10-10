import pandas as pd
import numpy as np

AGGREGATE_FUNCS = {
    "sum": "sum", "mean": "mean", "avg": "mean", "average": "mean",
    "count": "count", "max": "max", "min": "min", "std": "std", "median": "median",
}


def apply_filters(df, filters):
    if not filters:
        return df
    if isinstance(filters, dict):
        filters = [filters]
    for f in filters:
        col, op, val = f.get("column"), f.get("op"), f.get("value")
        if col not in df.columns:
            continue
        try:
            if op == "==":     df = df[df[col] == val]
            elif op == "!=":   df = df[df[col] != val]
            elif op == ">":    df = df[df[col] > val]
            elif op == "<":    df = df[df[col] < val]
            elif op == ">=":   df = df[df[col] >= val]
            elif op == "<=":   df = df[df[col] <= val]
            elif op == "contains":
                df = df[df[col].astype(str).str.contains(str(val), case=False, na=False)]
        except Exception as e:
            print(f"Filter error on {col}: {e}")
    return df


def run_query(df, query):
    if not query:
        return df.head(100)

    analysis_type = query.get("analysis_type", "descriptive")

    # Apply filters (shared by most types)
    filters = query.get("filter") or query.get("filters", [])
    filtered_df = apply_filters(df.copy(), filters)

    # Dispatch to handler
    handlers = {
        "descriptive": _descriptive,
        "trend": _trend,
        "comparative": _comparative,
        "ranking": _ranking,
        "distribution": _distribution,
        "correlation": _correlation,
        "anomaly": _anomaly,
        "pattern": _pattern,
        "forecast": _forecast,
    }
    handler = handlers.get(analysis_type, _descriptive)
    result = handler(filtered_df, query, df)

    # Round numerics
    num_cols = result.select_dtypes(include="number").columns
    result[num_cols] = result[num_cols].round(3)
    return result


# ─────────── HANDLERS ───────────

def _descriptive(filtered_df, query, full_df):
    group_by = query.get("group_by")
    aggregate = query.get("aggregate")

    if group_by:
        if isinstance(group_by, str):
            group_by = [c.strip() for c in group_by.split(",")]
        group_by = [c for c in group_by if c in filtered_df.columns]
        if not group_by:
            return pd.DataFrame({"Error": ["Group-by columns not found."]})

        if aggregate:
            agg_dict = {c: AGGREGATE_FUNCS.get(str(f).lower(), f)
                        for c, f in aggregate.items() if c in filtered_df.columns}
            result = filtered_df.groupby(group_by).agg(agg_dict).reset_index() if agg_dict \
                else filtered_df.groupby(group_by).size().reset_index(name="Count")
        else:
            # Default: describe numeric cols per group
            num = filtered_df.select_dtypes(include="number").columns.tolist()
            if num:
                result = filtered_df.groupby(group_by)[num].mean().reset_index()
            else:
                result = filtered_df.groupby(group_by).size().reset_index(name="Count")
    elif aggregate:
        agg_result = {}
        for col, func in aggregate.items():
            if col in filtered_df.columns:
                fn = AGGREGATE_FUNCS.get(str(func).lower(), func)
                agg_result[col] = getattr(filtered_df[col], fn)()
        result = pd.DataFrame([agg_result])
    else:
        # No group_by, no aggregate → full descriptive stats
        num_cols = filtered_df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            result = filtered_df[num_cols].describe().reset_index()
            result = result.rename(columns={"index": "Statistic"})
        else:
            result = filtered_df.head(100)

    # Sort and limit
    sort_by = query.get("sort_by")
    ascending = query.get("ascending", False)
    if sort_by and sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=ascending)
    elif group_by and aggregate:
        first_agg = [c for c in (aggregate or {}) if c in result.columns]
        if first_agg:
            result = result.sort_values(first_agg[0], ascending=False)

    limit = query.get("limit")
    if limit:
        result = result.head(int(limit))

    return result


def _trend(filtered_df, query, full_df):
    time_col = query.get("time_col")
    value_col = query.get("value_col")
    group_col = query.get("group_col")

    if not time_col or time_col not in filtered_df.columns:
        # fallback: try to detect a time column
        for col in filtered_df.columns:
            if filtered_df[col].dtype == "object":
                try:
                    filtered_df[col] = pd.to_datetime(filtered_df[col])
                    time_col = col
                    break
                except:
                    pass
        if not time_col:
            return pd.DataFrame({"Error": ["No time column found for trend analysis."]})

    try:
        filtered_df[time_col] = pd.to_datetime(filtered_df[time_col], errors="coerce")
    except:
        pass

    if not value_col or value_col not in filtered_df.columns:
        num_cols = filtered_df.select_dtypes(include="number").columns.tolist()
        value_col = num_cols[0] if num_cols else None
        if not value_col:
            return pd.DataFrame({"Error": ["No numeric column for trend."]})

    if group_col and group_col in filtered_df.columns:
        result = filtered_df.groupby([time_col, group_col])[value_col].sum().reset_index()
    else:
        result = filtered_df.groupby(time_col)[value_col].sum().reset_index()

    result = result.sort_values(time_col)
    result[time_col] = result[time_col].astype(str)
    return result


def _comparative(filtered_df, query, full_df):
    compare_col = query.get("compare_col")
    value_col = query.get("value_col")
    agg_func = AGGREGATE_FUNCS.get(str(query.get("aggregate", "sum")).lower(), "sum")

    if not compare_col or compare_col not in filtered_df.columns:
        cat_cols = filtered_df.select_dtypes(exclude="number").columns.tolist()
        compare_col = cat_cols[0] if cat_cols else None
        if not compare_col:
            return pd.DataFrame({"Error": ["No categorical column for comparison."]})

    if not value_col or value_col not in filtered_df.columns:
        num_cols = filtered_df.select_dtypes(include="number").columns.tolist()
        value_col = num_cols[0] if num_cols else None
        if not value_col:
            return pd.DataFrame({"Error": ["No numeric column for comparison."]})

    result = filtered_df.groupby(compare_col)[value_col].agg(agg_func).reset_index()
    result.columns = [compare_col, f"{agg_func}_{value_col}"]
    result = result.sort_values(f"{agg_func}_{value_col}", ascending=False)
    return result


def _ranking(filtered_df, query, full_df):
    rank_by = query.get("rank_by")
    group_by = query.get("group_by")
    agg_func = AGGREGATE_FUNCS.get(str(query.get("aggregate", "sum")).lower(), "sum")
    limit = int(query.get("limit", 10))
    ascending = query.get("ascending", False)

    if not rank_by or rank_by not in filtered_df.columns:
        num_cols = filtered_df.select_dtypes(include="number").columns.tolist()
        rank_by = num_cols[0] if num_cols else None

    if not group_by or group_by not in filtered_df.columns:
        cat_cols = filtered_df.select_dtypes(exclude="number").columns.tolist()
        group_by = cat_cols[0] if cat_cols else None

    if group_by and rank_by:
        result = filtered_df.groupby(group_by)[rank_by].agg(agg_func).reset_index()
        result.columns = [group_by, f"{agg_func}_{rank_by}"]
        result = result.sort_values(f"{agg_func}_{rank_by}", ascending=ascending)
    else:
        result = filtered_df.sort_values(rank_by, ascending=ascending) if rank_by else filtered_df

    return result.head(limit)


def _distribution(filtered_df, query, full_df):
    column = query.get("column")
    bins = int(query.get("bins", 10))
    group_col = query.get("group_col")

    if not column or column not in filtered_df.columns:
        num_cols = filtered_df.select_dtypes(include="number").columns.tolist()
        column = num_cols[0] if num_cols else None
        if not column:
            return pd.DataFrame({"Error": ["No numeric column for distribution."]})

    if group_col and group_col in filtered_df.columns:
        result = filtered_df.groupby(group_col)[column].describe().reset_index()
        return result

    counts, bin_edges = np.histogram(filtered_df[column].dropna(), bins=bins)
    labels = [f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}" for i in range(len(counts))]
    result = pd.DataFrame({"Bin": labels, "Count": counts})
    result["Percentage"] = (result["Count"] / result["Count"].sum() * 100).round(2)
    return result


def _correlation(filtered_df, query, full_df):
    columns = query.get("columns", [])
    valid = [c for c in columns if c in filtered_df.columns]

    num_cols = filtered_df.select_dtypes(include="number").columns.tolist()

    if len(valid) >= 2:
        corr_df = filtered_df[valid].corr().reset_index()
        corr_df = corr_df.rename(columns={"index": "Column"})
        return corr_df
    elif len(num_cols) >= 2:
        # Return pairwise scatter data for top 2 numeric cols
        cols = num_cols[:5]
        corr_df = filtered_df[cols].corr().reset_index()
        corr_df = corr_df.rename(columns={"index": "Column"})
        return corr_df
    return pd.DataFrame({"Error": ["Need at least 2 numeric columns for correlation."]})


def _anomaly(filtered_df, query, full_df):
    column = query.get("column")
    method = query.get("method", "zscore")
    group_by = query.get("group_by")

    num_cols = filtered_df.select_dtypes(include="number").columns.tolist()
    if not column or column not in filtered_df.columns:
        column = num_cols[0] if num_cols else None
        if not column:
            return pd.DataFrame({"Error": ["No numeric column for anomaly detection."]})

    work = filtered_df.copy()

    if method == "iqr":
        Q1 = work[column].quantile(0.25)
        Q3 = work[column].quantile(0.75)
        IQR = Q3 - Q1
        mask = (work[column] < Q1 - 1.5 * IQR) | (work[column] > Q3 + 1.5 * IQR)
    else:  # zscore
        mean = work[column].mean()
        std = work[column].std()
        work["_zscore"] = ((work[column] - mean) / std).abs() if std else 0
        mask = work["_zscore"] > 2.5

    work["is_anomaly"] = mask
    work["anomaly_label"] = work["is_anomaly"].map({True: "⚠️ Anomaly", False: "✅ Normal"})
    if "_zscore" in work.columns:
        work = work.drop(columns=["_zscore"])

    # Return anomalies + summary
    anomalies = work[work["is_anomaly"]].drop(columns=["is_anomaly"])
    if anomalies.empty:
        # Return all with label
        return work.drop(columns=["is_anomaly"]).head(50)
    return anomalies.head(50)


def _pattern(filtered_df, query, full_df):
    column = query.get("column")
    time_col = query.get("time_col")
    group_by = query.get("group_by")

    if column and column in filtered_df.columns:
        # Value frequency
        result = filtered_df[column].value_counts().reset_index()
        result.columns = [column, "Frequency"]
        result["Percentage"] = (result["Frequency"] / result["Frequency"].sum() * 100).round(2)
        return result.head(20)
    elif group_by and group_by in filtered_df.columns:
        result = filtered_df[group_by].value_counts().reset_index()
        result.columns = [group_by, "Frequency"]
        return result.head(20)
    else:
        cat_cols = filtered_df.select_dtypes(exclude="number").columns.tolist()
        if cat_cols:
            result = filtered_df[cat_cols[0]].value_counts().reset_index()
            result.columns = [cat_cols[0], "Frequency"]
            return result.head(20)
        return filtered_df.head(50)


def _forecast(filtered_df, query, full_df):
    time_col = query.get("time_col")
    value_col = query.get("value_col")
    periods = int(query.get("periods", 5))

    if not time_col or time_col not in filtered_df.columns:
        return pd.DataFrame({"Error": ["No time column for forecast."]})
    if not value_col or value_col not in filtered_df.columns:
        num_cols = filtered_df.select_dtypes(include="number").columns.tolist()
        value_col = num_cols[0] if num_cols else None
        if not value_col:
            return pd.DataFrame({"Error": ["No numeric column for forecast."]})

    try:
        filtered_df[time_col] = pd.to_datetime(filtered_df[time_col], errors="coerce")
    except:
        pass

    ts = filtered_df.groupby(time_col)[value_col].sum().reset_index().sort_values(time_col)
    ts = ts.dropna()

    if len(ts) < 3:
        return pd.DataFrame({"Error": ["Not enough data points to forecast."]})

    # Simple linear regression forecast
    ts["_t"] = np.arange(len(ts))
    x = ts["_t"].values
    y = ts[value_col].values
    m, b = np.polyfit(x, y, 1)

    # Generate future periods
    last_time = ts[time_col].max()
    try:
        freq = pd.infer_freq(ts[time_col])
        if not freq:
            freq = "ME"
        future_dates = pd.date_range(start=last_time, periods=periods + 1, freq=freq)[1:]
    except:
        future_dates = pd.date_range(start=last_time, periods=periods + 1, freq="ME")[1:]

    future_t = np.arange(len(ts), len(ts) + periods)
    future_vals = m * future_t + b

    forecast_df = pd.DataFrame({
        time_col: future_dates.astype(str),
        value_col: np.round(future_vals, 2),
        "Type": "📈 Forecast"
    })

    ts["Type"] = "📊 Historical"
    ts[time_col] = ts[time_col].astype(str)
    ts = ts.drop(columns=["_t"])

    return pd.concat([ts, forecast_df], ignore_index=True)