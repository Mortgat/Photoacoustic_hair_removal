import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

DOSSIER_FRAMES = "frames_sam2"  
FICHIER_PROMPTS = "prompts_sam2_v4.json"
FICHIER_MASK = "mask_prompt_v4.npy"
FICHIER_SORTIE = "masque_sam2_v4.npz" 
DOSSIER_VERIF_FINALE = "verif_finale_v4"

CHEMIN_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
CONFIG_MODELE = "configs/sam2.1/sam2.1_hiera_l.yaml" 

def executer_inference_video():
    print("--- 🧠 DÉMARRAGE SAM 2 : MASK PROMPTING (BIDIRECTIONNEL) ---")
    os.makedirs(DOSSIER_VERIF_FINALE, exist_ok=True)

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    with open(FICHIER_PROMPTS, 'r') as f: d_prompts = json.load(f)
    initial_frame = d_prompts["initial_frame_index"]
    
    print("📂 Chargement du Masque Ruban...")
    mask_prompt = np.load(FICHIER_MASK) 
    
    nb_frames_total = len([f for f in os.listdir(DOSSIER_FRAMES) if f.endswith('.jpg')])

    with torch.inference_mode():
        predictor = build_sam2_video_predictor(CONFIG_MODELE, CHEMIN_CHECKPOINT, device="cuda")
        
        inference_state = predictor.init_state(
            video_path=DOSSIER_FRAMES, 
            async_loading_frames=False,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True
        )
        
        # ⚡ INJECTION DU MASQUE D'AMORÇAGE (Sur la Golden Frame)
        print(f"🎯 Injection du Mask Prompt sur la frame {initial_frame}...")
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=initial_frame,
            obj_id=1,
            mask=mask_prompt
        )

        volume_final_1024 = np.zeros((nb_frames_total, 1024, 1024), dtype=bool)

        # ⚡ 1. PROPAGATION VERS L'AVANT (Forward)
        print("\n🚀 Propagation Spatio-Temporelle (Vers la fin de la vidéo)...")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            volume_final_1024[out_frame_idx] = (out_mask_logits[0, 0].cpu().numpy() > 0.0)
            if out_frame_idx % 100 == 0:
                print(f"   -> Frame {out_frame_idx} propagée.")

        # ⚡ 2. PROPAGATION VERS L'ARRIÈRE (Backward)
        print("\n⏪ Propagation Spatio-Temporelle (Vers le début de la vidéo)...")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            volume_final_1024[out_frame_idx] = (out_mask_logits[0, 0].cpu().numpy() > 0.0)
            if out_frame_idx % 100 == 0:
                print(f"   -> Frame {out_frame_idx} propagée (Reverse).")

        print(f"💾 Sauvegarde du tenseur complet dans '{FICHIER_SORTIE}'...")
        np.savez_compressed(FICHIER_SORTIE, masque=volume_final_1024)
        
        # --- EXPORT VISUEL ---
        print("\n📸 Génération des images de vérification finale (10 frames)...")
        fichiers_jpg = sorted([f for f in os.listdir(DOSSIER_FRAMES) if f.endswith('.jpg')])
        frames_a_sauver = np.linspace(0, nb_frames_total - 1, 10, dtype=int)
        
        for f_idx in frames_a_sauver:
            img_path = os.path.join(DOSSIER_FRAMES, fichiers_jpg[f_idx])
            img = Image.open(img_path)
            masque = volume_final_1024[f_idx]
            
            plt.figure(figsize=(10, 10), facecolor='black')
            plt.imshow(img, cmap='gray')
            
            overlay = np.zeros((*masque.shape, 4))
            if np.any(masque):
                overlay[..., 0] = 1.0 # Calque Rouge
                overlay[..., 3] = masque * 0.4
            plt.imshow(overlay)
            
            plt.title(f"SAM 2 Mask Prompting - Frame {f_idx}", color='white')
            plt.axis('off')
            plt.savefig(os.path.join(DOSSIER_VERIF_FINALE, f"resultat_v4_{f_idx:04d}.jpg"), facecolor='black', bbox_inches='tight')
            plt.close()

        print(f"🎉 Inférence terminée ! Vérifie le dossier '{DOSSIER_VERIF_FINALE}'.")

if __name__ == "__main__":
    executer_inference_video()