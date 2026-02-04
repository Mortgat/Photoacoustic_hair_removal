import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Copie-colle le chemin de ton fichier ici
FICHIER_A_TESTER = r"dicom_data\dicom\3851_PA835.mat" 
# ==========================================

def trouver_variable_volume(mat_dict):
    """Trouve la variable 3D dans le .mat"""
    for k, v in mat_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
            return k
    return None

def verifier_orientation_z():
    print(f"--- 🧭 AUDIT D'ORIENTATION Z : {os.path.basename(FICHIER_A_TESTER)} ---")
    
    if not os.path.exists(FICHIER_A_TESTER):
        print(f"❌ Erreur : Fichier introuvable -> {FICHIER_A_TESTER}")
        return

    # 1. Chargement
    try:
        mat = scipy.io.loadmat(FICHIER_A_TESTER)
        key = trouver_variable_volume(mat)
        if key is None:
            print("❌ Aucune donnée 3D trouvée.")
            return
        volume = mat[key]
        dims = volume.shape
        print(f"   📊 Dimensions brutes : {dims} (Variable: '{key}')")
    except Exception as e:
        print(f"❌ Erreur de lecture : {e}")
        return

    # 2. Détection Intelligente de l'Axe Z (La plus petite dimension)
    # np.argmin renvoie l'index de la valeur min (0, 1 ou 2)
    axe_z_detecte = np.argmin(dims)
    nb_tranches = dims[axe_z_detecte]
    
    print(f"   🧠 Intelligence : L'axe Z semble être l'axe n°{axe_z_detecte} (Taille : {nb_tranches})")
    print(f"      (Car c'est la plus petite dimension parmi {dims})")

    # 3. Calcul du Profil Moyen
    # On moyenne sur les deux AUTRES axes
    axes_lat = tuple([i for i in range(3) if i != axe_z_detecte])
    
    # Si Z est l'axe 2, on moyenne sur (0, 1). Si Z est 0, on moyenne sur (1, 2).
    profil_z = np.mean(volume, axis=axes_lat)
    
    # Normalisation (0-1)
    profil_z_norm = (profil_z - profil_z.min()) / (profil_z.max() - profil_z.min())

    # 4. Analyse du Pic
    z_pic = np.argmax(profil_z)
    ratio_pic = z_pic / nb_tranches

    print(f"   📍 Pic d'intensité max à la tranche : {z_pic} / {nb_tranches}")
    
    # 5. Verdict
    verdict = ""
    couleur_verdict = ""
    
    if ratio_pic < 0.33:
        verdict = "✅ ORIENTATION NORMALE (Haut = Début)"
        couleur_verdict = 'green'
        advice = f"L'axe Z est bien l'axe {axe_z_detecte}. Pas besoin de flip."
    elif ratio_pic > 0.66:
        verdict = "⚠️ ORIENTATION INVERSÉE (Haut = Fin)"
        couleur_verdict = 'red'
        advice = f"Il faudra faire np.flip(volume, axis={axe_z_detecte})."
    else:
        verdict = "❓ INCERTAIN (Pic au milieu)"
        couleur_verdict = 'orange'
        advice = "Vérification visuelle requise."

    print(f"   🏁 Verdict : {verdict}")
    print(f"      -> {advice}")

    # 6. Visualisation
    plt.figure(figsize=(10, 5))
    plt.plot(profil_z_norm, label='Intensité Moyenne', color='blue')
    plt.axvline(x=z_pic, color='red', linestyle='--', label=f'Pic Max (Tranche {z_pic})')
    
    plt.axvspan(0, nb_tranches*0.33, color='green', alpha=0.1, label='Zone Peau (Attendue)')
    plt.axvspan(nb_tranches*0.66, nb_tranches, color='red', alpha=0.1, label='Zone Fond (Suspecte)')
    
    plt.title(f"Diagnostic Z (Axe {axe_z_detecte}) - {os.path.basename(FICHIER_A_TESTER)}\n{verdict}", color=couleur_verdict, fontweight='bold')
    plt.xlabel(f"Profondeur (Indice le long de l'axe {axe_z_detecte})")
    plt.ylabel("Intensité Moyenne")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    verifier_orientation_z()