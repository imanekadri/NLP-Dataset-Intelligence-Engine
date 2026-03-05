import os
import csv
import json
import hashlib
from datetime import datetime
from langdetect import detect
from core.utils import extract_text_from_file, extract_html_metadata
from core.config import agent1_config as config

def _hash_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def run_ingestion(state):
    os.makedirs(config.EXTRACTED_DIR, exist_ok=True)
    os.makedirs(config.AGENT_OUTPUT_DIR, exist_ok=True)

    trace_records = []
    extracted_files = []
    languages_detected = {}
    seen_hashes = set()
    duplicate_count = 0
    file_counter = 0

    datasets_dir = state.get("input_dir", config.DATASETS_DIR)
    print(f"[Agent1] Scanning {datasets_dir}...")

    for root, _, files in os.walk(datasets_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in config.SUPPORTED_FORMATS:
                continue

            file_path = os.path.join(root, file)
            try:
                file_hash = _hash_file(file_path)
            except:
                continue

            if file_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(file_hash)

            file_id = f"doc_{file_counter}"

            try:
                text = extract_text_from_file(file_path, config.ENCODINGS_TO_TRY)
            except Exception as e:
                print(f"Error extracting text from {file_path}: {e}")
                continue
            html_meta = {}
            if ext in ['.html', '.htm']:
                try:
                    html_meta = extract_html_metadata(file_path, config.ENCODINGS_TO_TRY)
                except Exception:
                    html_meta = {}

            if not text or not text.strip():
                print("EMPTY TEXT:", file_path)
                continue

           
            if ext in ['.csv', '.json', '.xml']:
                extracted_path = os.path.join(config.EXTRACTED_DIR, f"{file_id}{ext}")
            else:
                extracted_path = os.path.join(config.EXTRACTED_DIR, f"{file_id}.txt")

            with open(extracted_path, "w", encoding="utf-8") as f:
                f.write(text)

            try:
                lang = detect(text[:5000])
            except:
                lang = "unknown"

            trace_info = {
                "doc_id": file_id,
                "origin_path": file_path,
                "format": ext,
                "extracted_path": extracted_path,
                "lang": lang,
                "words": len(text.split()),
                "char_count": len(text),
                "timestamp": datetime.now().isoformat(),
                "metadata": html_meta
            }

            # if HTML had JSON-LD payloads, save them as a separate json file
            if ext in ['.html', '.htm'] and html_meta.get('json_ld'):
                try:
                    json_ld_path = os.path.join(config.EXTRACTED_DIR, f"{file_id}.json")
                    with open(json_ld_path, "w", encoding="utf-8") as jf:
                        json.dump(html_meta.get('json_ld'), jf, ensure_ascii=False, indent=2)
                    trace_info['json_ld_path'] = json_ld_path
                    extracted_files.append(json_ld_path)
                except Exception:
                    pass

            trace_records.append(trace_info)
            extracted_files.append(extracted_path)
            languages_detected[file_id] = lang
            file_counter += 1

    # save trace_index.csv
    trace_csv_path = os.path.join(config.AGENT_OUTPUT_DIR, "trace_index.csv")
    if trace_records:
        # ensure CSV contains all possible keys (some traces may have extra fields like json_ld_path)
        fieldnames = []
        for r in trace_records:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(trace_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trace_records)

    #save report.json
    report_path = os.path.join(config.AGENT_OUTPUT_DIR, "report.json")
    report = {
        "agent": "TextIngestionAgent",
        "timestamp": datetime.now().isoformat(),
        "stats": {
            "total_docs": len(trace_records),
            "languages": list({r["lang"] for r in trace_records}),
            "formats": list({r["format"] for r in trace_records}),
            "duplicates": duplicate_count,
            "avg_length": int(sum(r["words"] for r in trace_records) / len(trace_records)) if trace_records else 0,
            "total_size_mb": round(sum(os.path.getsize(r["origin_path"]) for r in trace_records) / 1024 / 1024, 3)
        }
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # save documents_info.json
    documents_info_path = os.path.join(config.AGENT_OUTPUT_DIR, "documents_info.json")
    documents_info = []
    for r in trace_records:
        info = {
            "text_id": r["doc_id"],
            "source": r["origin_path"],
            "format": r["format"],
            "language": r["lang"],
            "length": r["char_count"],
            "word_count": r["words"],
            "text_sample": open(r["extracted_path"], "r", encoding="utf-8").read()[:200],
            "metadata": r.get("metadata", {"source": r["origin_path"]}),
            "extended_path": r["extracted_path"],
            "hash_md5": _hash_file(r["origin_path"])
        }
        if r.get('json_ld_path'):
            info['json_ld_path'] = r.get('json_ld_path')
        documents_info.append(info)

    with open(documents_info_path, "w", encoding="utf-8") as f:
        json.dump(documents_info, f, ensure_ascii=False, indent=2)

    print(f"[Agent1] {len(trace_records)} documents processed, {duplicate_count} duplicates removed.")
    print(f"[Agent1] Report saved to: {report_path}")
    print(f"[Agent1] Documents info saved to: {documents_info_path}")

    return {
        **state,
        "raw_docs": trace_records,
        "extracted_files": extracted_files,
        "languages_detected": languages_detected,
        "duplicates_removed": duplicate_count,
        "trace_csv_path": trace_csv_path,
        "report_json_path": report_path,
        "documents_info_json_path": documents_info_path
    }