"""Profile an uploaded register without inventing missing sales transactions."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


def normalize_name(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def review_company_register(frame):
    """Retain every source entry; group repeated names, never delete invoices."""
    headers = frame.iloc[:, 0].map(lambda v: normalize_name(v).casefold() if pd.notna(v) else "")
    matches = headers[headers.eq("company name")]
    if matches.empty:
        raise ValueError("A Company Name header was not found; this register layout is unsupported")
    header_index = matches.index[0]
    entries = []
    for index, value in frame.iloc[header_index + 1:, 0].items():
        if pd.isna(value) or not normalize_name(value):
            continue
        clean = normalize_name(value)
        entries.append({"source_row": int(index) + 1, "original_company_name": str(value),
                        "company_name": clean, "name_key": clean.casefold(),
                        "whitespace_changed": str(value) != clean})
    records = pd.DataFrame(entries)
    if records.empty:
        raise ValueError("No company entries found")
    counts = records.groupby("name_key", sort=False).size().to_dict()
    group_ids = {key: index + 1 for index, key in enumerate(counts)}
    groups = []
    for key, group in records.groupby("name_key", sort=False):
        groups.append({"company_name": group.company_name.iloc[0], "occurrences": len(group),
                       "first_source_row": int(group.source_row.iloc[0]),
                       "name_group": group_ids[key]})
    for entry in entries:
        entry["occurrences"] = int(counts[entry["name_key"]])
        entry["name_group"] = group_ids[entry["name_key"]]
        del entry["name_key"]
    groups.sort(key=lambda row: row["company_name"].casefold())
    period = None
    for value in frame.iloc[:header_index].to_numpy().ravel():
        if isinstance(value, str) and re.search(r"\d{2}-[A-Za-z]{3}-\d{4} to", value):
            period = value
    # Only a register with no populated transaction columns may use this adapter.
    if frame.iloc[header_index + 1:, 1:].notna().any().any():
        raise ValueError("Additional populated columns need a transaction-level schema review")
    report = {"source_entries": len(entries), "unique_companies": len(groups),
              "repeated_name_occurrences": len(entries)-len(groups),
              "whitespace_changes": sum(e["whitespace_changed"] for e in entries),
              "blank_worksheet_rows": int(frame.isna().all(axis=1).sum()),
              "source_header_row": int(header_index)+1,
              "report_period": period, "forecast_ready": False,
              "food_classification_available": False,
              "missing_fields": ["Transaction date", "Food item / SKU", "Quantity sold", "Unit price or sales amount"],
              "reason": "This register contains company names, not item-level sales.",
              "cleaning_rules": ["Separate title/address/period rows from the company list.",
                  "Normalize Unicode and repeated/leading/trailing whitespace.",
                  "Group exact names after case-insensitive whitespace normalization.",
                  "Preserve every source entry; repeated company names are not proof of duplicate transactions.",
                  "Do not infer food items, dates, sales, prices or quantities from company names."]}
    return {"report": report, "entries": entries, "groups": groups}


def inspect_workbook(path, output_dir="reports"):
    path = Path(path)
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    if len(sheets) != 1:
        raise ValueError("Select a worksheet explicitly for a workbook with multiple sheets")
    sheet, frame = next(iter(sheets.items()))
    result = review_company_register(frame)
    result["report"].update({"source_filename": path.name, "source_sheet": sheet,
                              "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "spreadsheet_review.json").write_text(json.dumps(result["report"], indent=2))
    pd.DataFrame(result["entries"]).to_csv(output / "company_entries.csv", index=False)
    pd.DataFrame(result["groups"]).to_csv(output / "company_names.csv", index=False)
    return result
