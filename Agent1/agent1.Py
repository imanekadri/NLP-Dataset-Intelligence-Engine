import os
import csv
import json
import hashlib
from datetime import datetime
from config import config
from utils import extract_text_from_file
from langdetect import detect_langs # Import pour détection multiple

class TextIngestionAgent:
    def __init__(self, cfg):
        self.config = cfg
        os.makedirs(self.config.EXTRACTED_DIR, exist_ok=True)
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

    def hash_file(self, path):
        """Détection de doublons via MD5"""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    def process_file(self, file_path, file_id):
        """Traitement, nettoyage et détection de langue"""
        try:
            text = extract_text_from_file(file_path, self.config.ENCODINGS_TO_TRY)
            if not text or not text.strip():
                return None

            # --- [Phase 2 : CLEANING] Nettoyage des espaces et sauts de ligne ---
            clean_text = " ".join(text.split())

            extracted_file_path = os.path.join(self.config.EXTRACTED_DIR, f"{file_id}.txt")
            with open(extracted_file_path, "w", encoding="utf-8") as f:
                f.write(clean_text)

            # --- [Détection Langue Arabe/Français] ---
            try:
                # Analyse les probabilités de langues sur les 2000 premiers caractères
                predictions = detect_langs(clean_text[:2000])
                lang = predictions[0].lang
                # Priorité à l'arabe ou au français si détectés à plus de 20%
                for p in predictions:
                    if p.lang in ['ar', 'fr'] and p.prob > 0.2:
                        lang = p.lang
                        break
            except:
                lang = "unknown"

            return {
                "doc_id": file_id,
                "origin_path": file_path,
                "format": os.path.splitext(file_path)[1].lower(),
                "extracted_path": extracted_file_path,
                "lang": lang,
                "words": len(clean_text.split()),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return None

    def scan_datasets(self):
        """Scanne récursivement tous les dossiers dans 'data'"""
        trace_records = []
        seen_hashes = set()
        file_counter = 0

        print(f"🔍 Scan du dossier : {self.config.DATASETS_DIR}")

        for root, _, files in os.walk(self.config.DATASETS_DIR):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.config.SUPPORTED_FORMATS:
                    file_path = os.path.join(root, file)
                    file_hash = self.hash_file(file_path)

                    if file_hash in seen_hashes:
                        continue
                    
                    seen_hashes.add(file_hash)
                    file_id = f"doc_{file_counter}"
                    
                    print(f"📄 Analyse : {file}...")
                    info = self.process_file(file_path, file_id)
                    
                    if info:
                        trace_records.append(info)
                        file_counter += 1

        self._save_outputs(trace_records)
        print(f"✅ Ingestion terminée : {len(trace_records)} fichiers traités.")

    def _save_outputs(self, trace_records):
        """Sauvegarde les résultats en JSON et CSV"""
        if not trace_records: return
        with open(self.config.TRACE_CSV, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=trace_records[0].keys())
            writer.writeheader()
            writer.writerows(trace_records)

        with open(self.config.DOCUMENTS_INFO_JSON, "w", encoding="utf-8") as f:
            json.dump(trace_records, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("🚀 Initialisation Agent 1...")
    agent = TextIngestionAgent(config)
    agent.scan_datasets()