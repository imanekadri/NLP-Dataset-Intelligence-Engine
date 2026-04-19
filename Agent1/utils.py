import os
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import numpy as np
import logging

# --- [PHASE DE NETTOYAGE : CLEAN PHASE] ---
# Éviter les messages d'avertissement rouges de oneDNN et des bibliothèques
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['FLAGS_use_onednn'] = '0' 
logging.getLogger("ppocr").setLevel(logging.ERROR) # Masquer les logs du moteur OCR

# --- [CONFIGURATION DU MOTEUR : MULTI-LANGUAGE OCR] ---
# Configuration du moteur pour supporter l'Arabe et le Français simultanément
try:
    # L'utilisation de lang='ar' supporte à la fois les caractères arabes et latins
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='ar', show_log=False, use_gpu=False)
    print("✅ System: Le moteur OCR (Arabe + Français) est actif.")
except Exception as e:
    print(f"⚠️ Warning: Échec de l'initialisation de l'OCR : {e}")
    ocr_engine = None

def extract_text_from_file(file_path, encodings):
    """Fonction principale pour identifier le type de fichier et choisir l'extracteur approprié"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return extract_pdf_smart(file_path)
    
    # Pour les fichiers texte classiques (TXT, CSV, JSON)
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except:
            continue
    return ""

def extract_pdf_smart(pdf_path):
    """Extraction intelligente : supporte les textes numériques et les fichiers scannés (Images)"""
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        
        for page_num, page in enumerate(doc):
            # 1. Tentative d'extraction du texte numérique (Digital Text)
            text = page.get_text().strip()
            
            # 2. [Logique OCR] : Si peu de texte est détecté, le fichier est considéré comme un scan/image
            # Seuil fixé à 50 caractères pour forcer l'OCR sur les pages scannées
            if len(text) < 50:
                if ocr_engine:
                    # Conversion de la page PDF en image haute résolution pour le traitement
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    
                    # Exécution de l'OCR pour extraire l'Arabe et le Français
                    result = ocr_engine.ocr(img)
                    if result and result[0]:
                        # Fusion des mots extraits en un seul texte propre
                        text = " ".join([line[1][0] for line in result[0]])
                
            full_text.append(text)
        
        doc.close()
        # Fusion de toutes les pages et nettoyage des espaces superflus
        return "\n".join(full_text)
        
    except Exception as e:
        # Retourne un texte vide en cas d'erreur technique pour éviter l'arrêt du programme
        return ""