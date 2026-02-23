import os
import csv
import json
import hashlib
from datetime import datetime
from langdetect import detect

from pipeline.state import NLPPipelineState
from pipeline.config import PipelineConfig
from utils.extraction import extract_text_from_file, init_tesseract


def _hash_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def run_ingestion(state: NLPPipelineState) -> NLPPipelineState:
    """
    Agent 1 — Data Ingestion.
    Scanne input_dir, extrait le texte, déduplique, détecte la langue.
    """
    config = PipelineConfig()
    config.ensure_dirs()
    init_tesseract(config.TESSERACT_CMD)

    datasets_dir = state.get("input_dir", config.DATASETS_DIR)

    trace_records = []
    extracted_files = []
    languages_detected = {}
    seen_hashes = set()
    duplicate_count = 0
    file_counter = 0

    print(f"[Agent 1] Scanning {datasets_dir}...")

    for root, dirs, files in os.walk(datasets_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in config.SUPPORTED_FORMATS:
                continue

            file_path = os.path.join(root, file)
            file_hash = _hash_file(file_path)

            if file_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(file_hash)

            file_id = f"doc_{file_counter}"

            try:
                text = extract_text_from_file(
                    file_path, config.ENCODINGS_TO_TRY,
                    poppler_path=config.POPPLER_PATH
                )
            except Exception as e:
                print(f"[Agent 1] Error reading {file_path}: {e}")
                continue

            # Sauvegarde du fichier extrait
            if ext in ['.csv', '.json', '.xml']:
                extracted_path = os.path.join(config.EXTRACTED_DIR, f"{file_id}{ext}")
            else:
                extracted_path = os.path.join(config.EXTRACTED_DIR, f"{file_id}.txt")

            with open(extracted_path, "w", encoding="utf-8") as f:
                f.write(text)

            # Détection de langue
            try:
                lang = detect(text[:5000]) if len(text) > 10 else "unknown"
            except Exception:
                lang = "unknown"

            trace_info = {
                "doc_id": file_id,
                "origin_path": file_path,
                "format": ext,
                "extracted_path": extracted_path,
                "lang": lang,
                "words": len(text.split()),
                "timestamp": datetime.now().isoformat()
            }
            trace_records.append(trace_info)
            extracted_files.append(extracted_path)
            languages_detected[file_id] = lang
            file_counter += 1

    # Sauvegarde trace_index.csv
    trace_csv_path = os.path.join(config.TRACES_DIR, "trace_index.csv")
    if trace_records:
        with open(trace_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=trace_records[0].keys())
            writer.writeheader()
            writer.writerows(trace_records)

    # Sauvegarde documents_info.json
    documents_info = []
    for r in trace_records:
        with open(r["extracted_path"], "r", encoding="utf-8") as f:
            content = f.read()
        documents_info.append({
            "text_id": r["doc_id"],
            "source": r["origin_path"],
            "format": r["format"],
            "language": r["lang"],
            "length": len(content),
            "word_count": r["words"],
            "text_sample": content[:200],
            "extracted_path": r["extracted_path"],
        })

    docs_info_path = os.path.join(config.TRACES_DIR, "documents_info.json")
    with open(docs_info_path, "w", encoding="utf-8") as f:
        json.dump(documents_info, f, ensure_ascii=False, indent=2)

    # Sauvegarde report.json
    report = {
        "agent": "Agent1_Ingestion",
        "timestamp": datetime.now().isoformat(),
        "total_docs": len(trace_records),
        "languages": list({r["lang"] for r in trace_records}),
        "formats": list({r["format"] for r in trace_records}),
        "duplicates": duplicate_count,
        "avg_words": int(sum(r["words"] for r in trace_records) / len(trace_records)) if trace_records else 0,
    }
    report_path = os.path.join(config.TRACES_DIR, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[Agent 1] {len(trace_records)} documents processed, {duplicate_count} duplicates removed.")

    return {
        **state,
        "raw_docs": trace_records,
        "trace_csv_path": trace_csv_path,
        "extracted_files": extracted_files,
        "languages_detected": languages_detected,
        "duplicates_removed": duplicate_count,
    }
