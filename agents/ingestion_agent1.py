import os
import csv
import json
import hashlib
from datetime import datetime
from langdetect import detect
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from core.utils import extract_text_from_file, extract_html_metadata , _hash_file
from core.config import agent1_config as config
import logging
import yake
from sentence_transformers import SentenceTransformer, util

print("Loading embedding model...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("Loading yake...")
# kw_model = KeyBERT(model=embedding_model)
kw_extractor = yake.KeywordExtractor(
    lan="auto",
    n=3,
    top=20
)
print("yake model loaded.")


# ==========================
# Chunking Function
# ==========================

def chunk_text(text, chunk_size=1000):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ==========================
# Fast Keyword Extraction
# ==========================
def extract_keywords(text, top_n=10):

    if not text or len(text) < 100:
        return []

    # Step 1 — Fast keyword candidates
    keywords = kw_extractor.extract_keywords(text)

    candidates = [kw[0] for kw in keywords]

    if not candidates:
        return []

    # Step 2 — semantic filtering
    text_embedding = embedding_model.encode(text[:2000], convert_to_tensor=True)
    kw_embeddings = embedding_model.encode(candidates, convert_to_tensor=True)

    scores = util.cos_sim(text_embedding, kw_embeddings)[0]

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [kw for kw, score in ranked[:top_n]]
# def _extract_keywords(text, top_n=20):
#
#     if not text or len(text) < 300:
#         return []
#
#     try:
#         # limit text size
#         text = text[:20000]
#
#         chunks = chunk_text(text)
#
#         keywords = []
#
#         # only first few chunks
#         for chunk in chunks[:5]:
#
#             kw = kw_model.extract_keywords(
#                 chunk,
#                 keyphrase_ngram_range=(1, 2),
#                 stop_words=None,
#                 use_mmr=True,
#                 diversity=0.5,
#                 nr_candidates=20,
#                 top_n=3
#             )
#
#             keywords.extend([k[0] for k in kw])
#
#         # remove duplicates
#         keywords = list(set(keywords))
#
#         return keywords[:top_n]
#
#     except Exception as e:
#         logging.warning(f"Keyword extraction failed: {e}")
#         return []




# ==========================
# Main Ingestion Function
# ==========================

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
            doc_name = os.path.splitext(os.path.basename(file_path))[0]

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

            keywords = extract_keywords(text)
            keywords_with_doc = [doc_name] + keywords
            trace_info = {
                "doc_id": file_id,
                "origin_path": file_path,
                "format": ext,
                "extracted_path": extracted_path,
                "lang": lang,
                "words": len(text.split()),
                "char_count": len(text),
                "timestamp": datetime.now().isoformat(),
                "metadata": html_meta,
                "key_words":  keywords_with_doc,
            }

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

    trace_csv_path = os.path.join(config.AGENT_OUTPUT_DIR, "trace_index.csv")
    if trace_records:
        fieldnames = []
        for r in trace_records:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(trace_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trace_records)

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

    print(f"[Agent1] {len(trace_records)} documents processed, {duplicate_count} duplicates removed.")
    print(f"[Agent1] Report saved to: {report_path}")

    return {
        **state,
        "raw_docs": trace_records,
        "extracted_files": extracted_files,
        "languages_detected": languages_detected,
        "duplicates_removed": duplicate_count,
        "trace_csv_path": trace_csv_path,
        "report_json_path": report_path
    }