import tifffile
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# CIBLE : Adaptation à ta nouvelle config
# ==========================================
# 1. Dossier racine (d'après ton image)
root_folder = r"Tiff_files\Tiff_stacks"

# 2. Sous-dossier pour la View 1 (Axe 0)
sub_folder = "3851_PA797_Stack_View1"

# 3. Nom du fichier
# ATTENTION : Si tu as utilisé mon code précédent, le nom est "slice_0_0114.tif" (axe_numero).
# Si tu as un doute, vérifie juste le nom exact dans ton dossier.
filename = "slice_0114.tif" 

chemin_image = os.path.join(root_folder, sub_folder, filename)
# ==========================================

def audit_image():
    print(f"--- 🕵️ AUDIT DE L'IMAGE : {os.path.basename(chemin_image)} ---")
    print(f"📍 Chemin complet : {chemin_image}\n")
    
    if not os.path.exists(chemin_image):
        print(f"❌ Fichier introuvable sur le disque.")
        print(f"   -> Le script cherche ici : {chemin_image}")
        print("   -> Vérifie que le dossier 'Tiff_files' est bien créé à côté de ce script.")
        return

    # 1. Lecture brute
    try:
        img = tifffile.imread(chemin_image)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier : {e}")
        return
    
    print(f"   📊 Dimensions : {img.shape}")
    print(f"   💾 Type donnée : {img.dtype}")
    
    # 2. Analyse statistique
    mini = img.min()
    maxi = img.max()
    moyenne = img.mean()
    
    print(f"\n   🔍 STATISTIQUES DES PIXELS :")
    print(f"      🌑 Valeur Min : {mini}")
    print(f"      ☀️ Valeur Max : {maxi}")
    print(f"      ⚖️ Moyenne    : {moyenne:.2f}")
    
    # 3. Verdict
    if maxi == 0:
        print("\n   ⚠️ ALERTE : L'image est 100% VIDE (Uniquement des zéros).")
        print("      -> Il y a eu un problème de conversion ou cette tranche est hors du tissu.")
    else:
        print(f"\n   ✅ RASSURE-TOI : Il y a du signal !")
        print(f"      Le pixel le plus fort est à {maxi}.")
        
        # Petit calcul pour voir le % de luminosité si c'est du 16 bit (65535)
        # Si c'est du 8 bit (255), le calcul s'adapte
        echelle = 65535 if img.dtype == np.uint16 else 255
        print(f"      Sur une échelle de {echelle}, c'est {(maxi/echelle)*100:.2f}% de luminosité.")
        
        # 4. Affichage corrigé (Boost de visibilité)
        print("\n   🖼️ Affichage avec contraste boosté (ferme la fenêtre pour finir)...")
        plt.figure(figsize=(10, 8))
        
        # On ignore le min absolu s'il est à 0 pour le contraste, pour mieux voir les détails faibles
        vmin_val = mini
        vmax_val = np.percentile(img, 99.5) # On sature les 0.5% les plus brillants pour voir le reste
        
        plt.imshow(img, cmap='inferno', vmin=vmin_val, vmax=vmax_val)
        plt.colorbar(label='Signal Photoacoustique')
        plt.title(f"Slice 114 (View 1) - Max: {maxi} - Contrast Boosted")
        plt.show()

if __name__ == "__main__":
    audit_image()