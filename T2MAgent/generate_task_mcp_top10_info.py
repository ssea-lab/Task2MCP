import os
import json
import ast
import argparse
from typing import Any, Dict, List

import pandas as pd


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def first_existing(row: pd.Series, candidates: List[str], default: Any = "") -> Any:
    for col in candidates:
        if col in row.index:
            value = row[col]
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            return value
    return default


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of the candidate columns exists: {candidates}")


def parse_top10(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(int(item))
            except Exception:
                continue
        return out

    text = clean_text(value)
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                out = []
                for item in parsed:
                    try:
                        out.append(int(item))
                    except Exception:
                        continue
                return out
        except Exception:
            pass

    numbers = []
    current = ""
    for ch in text:
        if ch.isdigit() or (ch == '-' and not current):
            current += ch
        else:
            if current:
                try:
                    numbers.append(int(current))
                except Exception:
                    pass
                current = ""
    if current:
        try:
            numbers.append(int(current))
        except Exception:
            pass
    return numbers


def build_task_record(task_row: pd.Series) -> Dict[str, Any]:
    return {
        "task_id": int(first_existing(task_row, ["task_id", "Task_id"], 0)),
        "task_name": clean_text(first_existing(task_row, ["Task_name", "name", "task_name"])),
        "task_description": clean_text(first_existing(task_row, ["description"])),
        "task_programming_language": clean_text(first_existing(task_row, ["programming_language", "language"])),
        "task_category": clean_text(first_existing(task_row, ["category"])),
        "task_subcategory": clean_text(first_existing(task_row, ["subcategory", "new_category"], clean_text(first_existing(task_row, ["category"])))),
        "task_theme": clean_text(first_existing(task_row, ["theme"])),
    }


def build_mcp_slot(rank: int, mcp_row: pd.Series | None) -> Dict[str, Any]:
    prefix = f"mcp{rank}_"
    if mcp_row is None:
        return {
            f"{prefix}rank": rank,
            f"{prefix}num": "",
            f"{prefix}name": "",
            f"{prefix}description": "",
            f"{prefix}star": "",
            f"{prefix}watching": "",
            f"{prefix}license": "",
            f"{prefix}language": "",
            f"{prefix}activity": "",
            f"{prefix}subcategory": "",
            f"{prefix}system": "",
            f"{prefix}official": "",
        }

    return {
        f"{prefix}rank": rank,
        f"{prefix}num": first_existing(mcp_row, ["num", "mcp_num", "id"], ""),
        f"{prefix}name": clean_text(first_existing(mcp_row, ["name"])),
        f"{prefix}description": clean_text(first_existing(mcp_row, ["description"])),
        f"{prefix}star": first_existing(mcp_row, ["stars_count", "stars", "star"], ""),
        f"{prefix}watching": first_existing(mcp_row, ["watching_count", "watching", "watchers", "watch_count"], ""),
        f"{prefix}license": clean_text(first_existing(mcp_row, ["license"])),
        f"{prefix}language": clean_text(first_existing(mcp_row, ["language"])),
        f"{prefix}activity": first_existing(mcp_row, ["activity"], ""),
        f"{prefix}subcategory": clean_text(first_existing(mcp_row, ["subcategory", "new_category", "category"])),
        f"{prefix}system": clean_text(first_existing(mcp_row, ["system"])),
        f"{prefix}official": clean_text(first_existing(mcp_row, ["official"])),
    }


def expected_columns() -> List[str]:
    cols = [
        "task_id",
        "task_name",
        "task_description",
        "task_programming_language",
        "task_category",
        "task_subcategory",
        "task_theme",
    ]
    for rank in range(1, 11):
        prefix = f"mcp{rank}_"
        cols.extend([
            f"{prefix}rank",
            f"{prefix}num",
            f"{prefix}name",
            f"{prefix}description",
            f"{prefix}star",
            f"{prefix}watching",
            f"{prefix}license",
            f"{prefix}language",
            f"{prefix}activity",
            f"{prefix}subcategory",
            f"{prefix}system",
            f"{prefix}official",
        ])
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate task_mcp_top10_info.csv from recs_test.csv, task.csv, and mcp_raw.csv.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--recs_file", type=str, default="./output/recs_test.csv")
    parser.add_argument("--output_file", type=str, default="./output/task_mcp_top10_info.csv")
    args = parser.parse_args()

    task_path = os.path.join(args.data_dir, "task.csv")
    mcp_path = os.path.join(args.data_dir, "mcp_raw.csv")

    recs_df = pd.read_csv(args.recs_file)
    task_df = pd.read_csv(task_path)
    mcp_df = pd.read_csv(mcp_path)

    rec_task_col = find_column(recs_df, ["task_id", "Task_id"])
    task_id_col = find_column(task_df, ["task_id", "Task_id"])
    mcp_id_col = find_column(mcp_df, ["num", "mcp_num", "id"])

    task_lookup: Dict[int, pd.Series] = {
        int(row[task_id_col]): row
        for _, row in task_df.iterrows()
    }
    mcp_lookup: Dict[int, pd.Series] = {
        int(row[mcp_id_col]): row
        for _, row in mcp_df.iterrows()
    }

    rows: List[Dict[str, Any]] = []

    for _, rec_row in recs_df.iterrows():
        task_id = int(rec_row[rec_task_col])
        task_row = task_lookup.get(task_id)
        if task_row is None:
            continue

        out_row = build_task_record(task_row)
        top10_ids = parse_top10(first_existing(rec_row, ["top10"]))[:10]

        for rank in range(1, 11):
            mcp_row = None
            if rank <= len(top10_ids):
                mcp_row = mcp_lookup.get(int(top10_ids[rank - 1]))
            out_row.update(build_mcp_slot(rank, mcp_row))

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    ordered_cols = expected_columns()
    for col in ordered_cols:
        if col not in out_df.columns:
            out_df[col] = ""
    out_df = out_df[ordered_cols]

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    out_df.to_csv(args.output_file, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {args.output_file}")
    print(f"[INFO] rows={len(out_df)}")


if __name__ == "__main__":
    main()
