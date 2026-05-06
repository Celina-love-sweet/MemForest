import argparse
import json
from pathlib import Path
from statistics import mean


LONGMEMEVAL_CATEGORY_ORDER = [
    "knowledge_update",
    "multi_session",
    "single_session_assistant",
    "single_session_preference",
    "single_session_user",
    "temporal_reasoning",
]

PERSONAMEM_CATEGORY_ORDER = [
    "recall_user_shared_facts",
    "recalling_facts_mentioned_by_the_user",
    "recalling_the_reasons_behind_previous_updates",
    "track_full_preference_evolution",
    "provide_preference_aligned_recommendations",
    "generalizing_to_new_scenarios",
    "suggest_new_ideas",
]

# Compatibility aliases across old/new exports.
CATEGORY_ALIASES = {
    "recall_user_shared_fact": "recall_user_shared_facts",
    "recall_facts_mentioned_by_the_user": "recalling_facts_mentioned_by_the_user",
    "recall_the_reasons_behind_previous_updates": "recalling_the_reasons_behind_previous_updates",
    "track_preference_evolution": "track_full_preference_evolution",
    "provide_preference_aligned_recommendation": "provide_preference_aligned_recommendations",
    "generalize_to_new_scenarios": "generalizing_to_new_scenarios",
    "suggest_new_idea": "suggest_new_ideas",
}


def load_items(input_file: Path):
    data = json.loads(input_file.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        all_items = []
        for _, value in data.items():
            if isinstance(value, list):
                all_items.extend(value)
        return all_items
    if isinstance(data, list):
        return data
    return []


def _to_float(value):
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except Exception:
        return None


def normalize_category(value: str) -> str:
    c = str(value).strip().lower()
    c = c.replace(" ", "_").replace("-", "_")
    c = CATEGORY_ALIASES.get(c, c)
    return c


def choose_category_order(items):
    observed = [normalize_category(item.get("category", "")) for item in items]
    observed_set = {c for c in observed if c}
    long_overlap = len(observed_set & set(LONGMEMEVAL_CATEGORY_ORDER))
    persona_overlap = len(observed_set & set(PERSONAMEM_CATEGORY_ORDER))

    if persona_overlap > long_overlap:
        return PERSONAMEM_CATEGORY_ORDER, "PersonaMem"
    if long_overlap > 0:
        return LONGMEMEVAL_CATEGORY_ORDER, "LongMemEval"
    return sorted(observed_set), "Unknown(custom)"


def pick_score(item, metric: str):
    if metric in item:
        return _to_float(item[metric])

    if metric == "llm_score":
        for alt in ("llm_label", "is_correct"):
            if alt in item:
                return _to_float(item[alt])
    elif metric == "bleu_score":
        for alt in ("bleu1", "bleu_1", "b1"):
            if alt in item:
                return _to_float(item[alt])
    elif metric == "f1_score":
        if "f1" in item:
            return _to_float(item["f1"])
    return None


def fmt_percent(value):
    return "N/A" if value is None else f"{value * 100:.1f}%"


def safe_mean(values):
    return mean(values) if values else None


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-category scores for LongMemEval / PersonaMem evaluation_metrics.json."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("evaluation_metrics.json"),
        help="Path to evaluation_metrics.json",
    )
    args = parser.parse_args()

    items = load_items(args.input_file)
    if not items:
        raise ValueError(f"No items found in {args.input_file}")

    category_order, dataset_name = choose_category_order(items)
    stats = {
        category: {"count": 0, "acc": [], "b1": [], "f1": []}
        for category in category_order
    }
    overall = {"count": 0, "acc": [], "b1": [], "f1": []}

    for item in items:
        category = normalize_category(item.get("category", ""))
        if category not in category_order:
            continue

        acc = pick_score(item, "llm_score")
        b1 = pick_score(item, "bleu_score")
        f1 = pick_score(item, "f1_score")

        stats[category]["count"] += 1
        overall["count"] += 1
        if acc is not None:
            stats[category]["acc"].append(acc)
            overall["acc"].append(acc)
        if b1 is not None:
            stats[category]["b1"].append(b1)
            overall["b1"].append(b1)
        if f1 is not None:
            stats[category]["f1"].append(f1)
            overall["f1"].append(f1)

    if overall["count"] == 0:
        raise ValueError("No recognized category items found after filtering.")

    rows = []
    for category in category_order:
        row = stats[category]
        rows.append(
            (
                category,
                row["count"],
                fmt_percent(safe_mean(row["acc"])),
                fmt_percent(safe_mean(row["b1"])),
                fmt_percent(safe_mean(row["f1"])),
            )
        )

    overall_row = (
        "Overall",
        overall["count"],
        fmt_percent(safe_mean(overall["acc"])),
        fmt_percent(safe_mean(overall["b1"])),
        fmt_percent(safe_mean(overall["f1"])),
    )

    category_w = max(len("Category"), len("Overall"), *(len(r[0]) for r in rows))
    count_w = max(len("Count"), *(len(str(r[1])) for r in rows + [overall_row]))
    score_w = max(
        len("Accuracy"),
        len("B1"),
        len("F1"),
        *(len(r[2]) for r in rows + [overall_row]),
        *(len(r[3]) for r in rows + [overall_row]),
        *(len(r[4]) for r in rows + [overall_row]),
    )

    print(f"Input: {args.input_file}")
    print(f"Detected dataset: {dataset_name}")
    print(
        f"{'Category':<{category_w}}  {'Count':>{count_w}}  "
        f"{'Accuracy':>{score_w}}  {'B1':>{score_w}}  {'F1':>{score_w}}"
    )
    for category, count, acc, b1, f1 in rows:
        print(
            f"{category:<{category_w}}  {count:>{count_w}}  "
            f"{acc:>{score_w}}  {b1:>{score_w}}  {f1:>{score_w}}"
        )
    print(
        f"{overall_row[0]:<{category_w}}  {overall_row[1]:>{count_w}}  "
        f"{overall_row[2]:>{score_w}}  {overall_row[3]:>{score_w}}  {overall_row[4]:>{score_w}}"
    )


if __name__ == "__main__":
    main()
