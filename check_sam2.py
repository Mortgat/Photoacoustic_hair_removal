import torch
import os

print("--- 🔍 VÉRIFICATION DE L'ENVIRONNEMENT SAM 2 ---")

# 1. Vérification du GPU (Crucial pour le Video Predictor)
if torch.cuda.is_available():
    print(f"✅ GPU détecté : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM disponible : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ ATTENTION : Aucun GPU CUDA détecté ! L'inférence vidéo va être affreusement lente ou planter.")

# 2. Vérification de l'installation du package
try:
    import sam2
    from sam2.build_sam import build_sam2_video_predictor
    print("✅ Package 'sam2' importé avec succès.")
except ImportError:
    print("❌ Package 'sam2' introuvable. As-tu fait 'pip install -e .' dans le dossier officiel de SAM 2 ?")

# 3. Vérification des Poids du Modèle (Checkpoints)
# Modifie ce chemin selon l'endroit où tu as téléchargé le fichier .pt
CHEMIN_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"

if os.path.exists(CHEMIN_CHECKPOINT):
    print(f"✅ Poids du modèle trouvés : {CHEMIN_CHECKPOINT}")
else:
    print(f"❌ Poids introuvables à : {CHEMIN_CHECKPOINT}")
    print("   -> Pense à télécharger 'sam2.1_hiera_large.pt' (ou base/small) via le script 'download_ckpts.sh' de Meta.")