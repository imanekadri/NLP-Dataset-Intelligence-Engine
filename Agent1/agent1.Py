# agent1.py
import os
import csv
import json
import hashlib
from datetime import datetime
from config import config
from utils import extract_text_from_file
from langdetect import detect

class TextIngestionAgent:
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.EXTRACTED_DIR, exist_ok=True)
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def hash_file(self, path):
        """ count hash MD5 for the file """
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    def process_file(self, file_path, file_id):
       
        try:
            text = extract_text_from_file(file_path, self.config.ENCODINGS_TO_TRY)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

        ext = os.path.splitext(file_path)[1].lower()

      
        if ext in ['.csv', '.json', '.xml']:
            extracted_file_path = os.path.join(self.config.EXTRACTED_DIR, f"{file_id}{ext}")
        else:
            extracted_file_path = os.path.join(self.config.EXTRACTED_DIR, f"{file_id}.txt")

       
        with open(extracted_file_path, "w", encoding="utf-8") as f:
            f.write(text)

        #  detect langue 
        try:
            lang = detect(text)
        except:
            lang = "unknown"

        # trace
        trace_info = {
            "doc_id": file_id,
            "origin_path": file_path,
            "format": ext,
            "extracted_path": extracted_file_path,
            "lang": lang,
            "words": len(text.split()),
            "timestamp": datetime.now().isoformat()
        }
        return trace_info

    def scan_datasets(self):
        
        trace_records = []
        file_counter = 0
        seen_hashes = set()
        duplicate_count = 0

        for root, dirs, files in os.walk(self.config.DATASETS_DIR):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.config.SUPPORTED_FORMATS:
                    file_path = os.path.join(root, file)
                    file_hash = self.hash_file(file_path)

                    #  detect duplicate
                    if file_hash in seen_hashes:
                        duplicate_count += 1
                        continue
                    seen_hashes.add(file_hash)

                    file_id = f"doc_{file_counter}"
                    trace_info = self.process_file(file_path, file_id)
                    if trace_info:
                        trace_records.append(trace_info)
                        file_counter += 1

        if not trace_records:
            print("no file to process")
            return {}

        # حفظ trace_index.csv
        with open(self.config.TRACE_CSV, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=trace_records[0].keys())
            writer.writeheader()
            writer.writerows(trace_records)

        # حفظ معلومات تفصيلية لكل ملف
        documents_info = []
        for r in trace_records:
            info = {
                "text_id": r["doc_id"],
                "source": r["origin_path"],
                "format": r["format"],
                "language": r["lang"],
                "length": len(open(r["extracted_path"], "r", encoding="utf-8").read()),
                "word_count": r["words"],
                "text_sample": open(r["extracted_path"], "r", encoding="utf-8").read()[:200],
                "metadata": {"source": r["origin_path"]},
                "extended_path": r["extracted_path"],
                "hash_md5": self.hash_file(r["origin_path"])
            }
            documents_info.append(info)

        with open(os.path.join(self.config.OUTPUT_DIR, "documents_info.json"), "w", encoding="utf-8") as f:
            json.dump(documents_info, f, ensure_ascii=False, indent=2)

        
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

        with open(self.config.REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"{len(trace_records)} nbr for file process")
       
        return report


if __name__ == "__main__":
    agent = TextIngestionAgent(config)
    report = agent.scan_datasets()
