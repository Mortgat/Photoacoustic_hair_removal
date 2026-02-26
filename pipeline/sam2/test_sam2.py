import torch
from sam2.build_sam import build_sam2_video_predictor
import os

# 1. On définit les chemins vers le cerveau de SAM 2
# SAM 2 vient avec ses fichiers de configuration par défaut quand on l'installe via pip
config_file = "configs/sam2.1/sam2.1_hiera_l.yaml" 
checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"

print("🚀 Démarrage de l'initialisation de SAM 2...")

# 2. On vérifie la configuration GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU détecté : {torch.cuda.get_device_name(0)}")
    # Optimisation vitale pour l'imagerie médicale avec Ampere/Ada Lovelace
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
else:
    device = torch.device("cpu")
    print("⚠️ Attention, GPU non trouvé. SAM 2 va être extrêmement lent sur CPU.")

# 3. Chargement du modèle
try:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Le fichier {checkpoint_path} est introuvable. L'as-tu bien téléchargé ?")
        
    predictor = build_sam2_video_predictor(config_file, checkpoint_path, device=device)
    print("🎉 SUCCÈS : SAM 2 Large est chargé dans la VRAM de ton GPU et prêt à analyser tes données !")
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")