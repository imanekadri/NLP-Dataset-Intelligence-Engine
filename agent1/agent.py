"""Agent 1 - Data Ingestion (Scan Intelligent)"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import Counter
import pandas as pd
from pathlib import Path

from .config import config
from .utils import FileUtils, TextExtractor, TextCleaner

class TextIngestionAgent:
    """
    Agent 1: Data Ingestion - Scan Intelligent
    
    Rôle: Scanner toutes les sources texte et produire un rapport détaillé
    """
    
    def __init__(self, name: str = "TextIngestionAgent"):
        self.name = name
        self.file_utils = FileUtils()
        self.text_extractor = TextExtractor()
        self.stats = {
            'total_docs': 0,
            'languages': [],
            'formats': [],
            'duplicates': 0,
            'avg_length': 0,
            'total_size_mb': 0,
            'encoding_stats': {},
            'file_hashes': set()
        }
        self.processed_documents = []
        
    def scan_datasets(self, datasets_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Scan tous les datasets dans le dossier spécifié
        
        Args:
            datasets_path: Chemin vers le dossier des datasets
                          (par défaut: config.DATASETS_DIR)
        
        Returns:
            Rapport détaillé du scan
        """
        if datasets_path is None:
            datasets_path = config.DATASETS_DIR
        
        print(f"🔍 Scan du dossier: {datasets_path}")
        
        # Vérifier si le dossier existe
        if not os.path.exists(datasets_path):
            print(f"❌ Dossier non trouvé: {datasets_path}")
            return self._generate_report()
        
        # Scanner tous les fichiers
        all_files = []
        for root, dirs, files in os.walk(datasets_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in config.SUPPORTED_FORMATS:
                    all_files.append(file_path)
        
        print(f"📊 {len(all_files)} fichiers trouvés")
        
        # Traiter chaque fichier
        for file_path in all_files:
            self._process_file(file_path)
        
        # Calculer les statistiques finales
        self._compute_final_stats()
        
        return self._generate_report()
    
    def _process_file(self, file_path: str) -> None:
        """Traite un fichier individuel"""
        try:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            file_ext = os.path.splitext(file_path)[1].lower()
            file_hash = self.file_utils.compute_file_hash(file_path)
            
            # Vérifier les doublons
            if file_hash in self.stats['file_hashes']:
                self.stats['duplicates'] += 1
                print(f"⚠️ Doublon détecté: {file_path}")
                return
            
            self.stats['file_hashes'].add(file_hash)
            self.stats['total_size_mb'] += file_size
            
            # Détecter l'encodage pour les fichiers texte
            encoding = None
            if file_ext in ['.txt', '.csv', '.json', '.xml', '.log']:
                encoding = self.file_utils.detect_encoding(file_path)
                self.stats['encoding_stats'][encoding] = self.stats['encoding_stats'].get(encoding, 0) + 1
            
            # Extraire le texte selon le format
            texts = self._extract_text(file_path, file_ext, encoding)
            
            # Analyser les textes extraits
            for text_data in texts:
                self._analyze_text(text_data, file_path, file_ext)
            
            print(f"✅ Traité: {os.path.basename(file_path)} ({len(texts)} documents)")
            
        except Exception as e:
            print(f"❌ Erreur sur {file_path}: {str(e)}")
    
    def _extract_text(self, file_path: str, file_ext: str, encoding: Optional[str]) -> List[Dict]:
        """Extrait le texte selon le format du fichier"""
        extractors = {
            '.csv': lambda: self.text_extractor.extract_from_csv(file_path, encoding=encoding),
            '.json': lambda: self.text_extractor.extract_from_json(file_path),
            '.txt': lambda: self.text_extractor.extract_from_txt(file_path, encoding or 'utf-8'),
            '.log': lambda: self.text_extractor.extract_from_log(file_path, encoding or 'utf-8'),
            '.pdf': lambda: self.text_extractor.extract_from_pdf(file_path),
            '.docx': lambda: self.text_extractor.extract_from_docx(file_path),
        }
        
        extractor = extractors.get(file_ext)
        if extractor:
            return extractor()
        
        return []
    
    def _analyze_text(self, text_data: Dict, file_path: str, file_ext: str) -> None:
        """Analyse un texte extrait"""
        text = text_data['text']
        
        # Appliquer le nettoyage automatique
        text = TextCleaner.clean(text)
        
        if not text or len(text.strip()) < 10:  # Ignorer les textes trop courts
            return
        
        # Détecter la langue
        lang = self.file_utils.detect_language(text)
        if lang:
            self.stats['languages'].append(lang)
        
        # Ajouter aux documents traités
        doc_info = {
            'text_id': f"doc_{len(self.processed_documents)}",
            'source': file_path,
            'format': file_ext,
            'language': lang,
            'length': len(text),
            'word_count': len(text.split()),
            'text_sample': text[:200] + '...' if len(text) > 200 else text,
            'metadata': text_data.get('metadata', {})
        }
        
        self.processed_documents.append(doc_info)
        
        # Mettre à jour les statistiques
        self.stats['formats'].append(file_ext)
    
    def _compute_final_stats(self) -> None:
        """Calcule les statistiques finales"""
        if not self.processed_documents:
            return
        
        # Nombre total de documents
        self.stats['total_docs'] = len(self.processed_documents)
        
        # Langues les plus fréquentes
        lang_counter = Counter(self.stats['languages'])
        self.stats['languages'] = [
            {'lang': lang, 'count': count}
            for lang, count in lang_counter.most_common()
        ]
        
        # Formats les plus fréquents
        format_counter = Counter(self.stats['formats'])
        self.stats['formats'] = [
            {'format': fmt, 'count': count}
            for fmt, count in format_counter.most_common()
        ]
        
        # Longueur moyenne
        total_words = sum(doc['word_count'] for doc in self.processed_documents)
        self.stats['avg_length'] = f"{total_words // self.stats['total_docs']} words"
        
        # Statistiques d'encodage
        encoding_counter = Counter(self.stats['encoding_stats'])
        self.stats['encoding_stats'] = dict(encoding_counter.most_common())
    
    def _generate_report(self) -> Dict[str, Any]:
        """Génère le rapport final"""
        report = {
            'agent': self.name,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats,
            'documents_summary': {
                'total_processed': self.stats['total_docs'],
                'unique_files': len(self.stats['file_hashes']) - self.stats['duplicates'],
                'sample_documents': self.processed_documents[:5]  # 5 exemples
            }
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str = "ingestion_report.json") -> None:
        """Sauvegarde le rapport au format JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"📝 Rapport sauvegardé: {output_path}")
    
    def get_processed_dataframe(self) -> pd.DataFrame:
        """Retourne les documents traités sous forme de DataFrame"""
        return pd.DataFrame(self.processed_documents)

# Point d'entrée pour tester l'agent
if __name__ == "__main__":
    # Créer l'agent
    agent = TextIngestionAgent()
    
    # Scanner les datasets
    report = agent.scan_datasets()
    
    # Afficher le rapport
    print("\n" + "="*50)
    print("📊 RAPPORT D'INGESTION")
    print("="*50)
    print(json.dumps(report['stats'], indent=2, ensure_ascii=False))
    
    # Sauvegarder le rapport
    agent.save_report(report)
    
    # Afficher quelques exemples
    print("\n📄 Exemples de documents traités:")
    for doc in report['documents_summary']['sample_documents']:
        print(f"- {doc['text_id']} | Langue: {doc['language']} | Mots: {doc['word_count']}")