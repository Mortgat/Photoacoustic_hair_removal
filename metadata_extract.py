import os
import pydicom
import scipy.io

# ==========================================
# 1. LISTE DES FICHIERS À ANALYSER
# ==========================================
liste_fichiers = [
    r"dicom_data/dicom/3851_PA797.mat",
    r"dicom_data/dicom/3851_PA835.mat",
    r"dicom_data/dicom/3888_20180222_084017227_Volume2.dcm",
    r"dicom_data/dicom/4261_20180326_112555703_Volume2.dcm"
]

def analyser_dicom_complet(chemin):
    nom_fichier = os.path.basename(chemin)
    print(f"\n--- 🎞️ ANALYSE DICOM COMPLÈTE : {nom_fichier} ---")
    
    if not os.path.exists(chemin):
        print("❌ Fichier introuvable sur le disque.")
        return

    try:
        # Lecture optimisée (sans charger les pixels)
        ds = pydicom.dcmread(chemin, stop_before_pixels=True)

        # --- A. INFOS MACHINE (Le contexte) ---
        manufacturer = ds.get('Manufacturer', 'Inconnu')
        model = ds.get('ManufacturerModelName', 'Inconnu')
        print(f"🏭 Machine        : {manufacturer} - {model}")
        print(f"ℹ️  Patient ID     : {ds.get('PatientID', 'Inconnu')}")

        # --- B. DIMENSIONS DU CUBE ---
        frames = ds.get('NumberOfFrames', 1)
        rows = ds.Rows
        cols = ds.Columns
        
        print(f"📊 TAILLE DU CUBE (Voxels) :")
        print(f"   X (Largeur)    : {cols}")
        print(f"   Y (Hauteur)    : {rows}")
        if frames > 1:
            print(f"   Z (Profondeur) : {frames} tranches (Volume 3D détecté)")
        else:
            print(f"   Z (Profondeur) : 1 (⚠️ Image 2D unique)")

        # --- C. RÉSOLUTION SPATIALE (La chasse aux indices) ---
        print(f"📏 RÉSOLUTION SPATIALE (mm) :")

        # 1. Résolution X/Y (Classique)
        ps = ds.get('PixelSpacing', None)
        if ps:
            print(f"   ✅ Voxel X        : {float(ps[1]):.5f} mm")
            print(f"   ✅ Voxel Y        : {float(ps[0]):.5f} mm")
        else:
            print("   ❌ Voxel X/Y      : NON TROUVÉ")

        # 2. Résolution Z (Approfondie)
        z_trouve = False
        
        # Etape 1 : Tags standards de premier niveau
        if 'SpacingBetweenSlices' in ds:
            print(f"   ✅ Voxel Z        : {float(ds.SpacingBetweenSlices):.5f} mm (Tag standard)")
            z_trouve = True
        elif 'SliceThickness' in ds:
            print(f"   ✅ Voxel Z        : {float(ds.SliceThickness):.5f} mm (Tag SliceThickness)")
            z_trouve = True
            
        # Etape 2 : Séquences cachées (Multiframe / Deep Dive)
        if not z_trouve and 'SharedFunctionalGroupsSequence' in ds:
            print("   🔎 Recherche dans les séquences fonctionnelles...")
            shared_seq = ds.SharedFunctionalGroupsSequence[0]
            if 'PixelMeasuresSequence' in shared_seq:
                pixel_seq = shared_seq.PixelMeasuresSequence[0]
                
                if 'SliceThickness' in pixel_seq:
                    print(f"   🎉 Voxel Z TROUVÉ (Caché dans PixelMeasures) : {pixel_seq.SliceThickness} mm")
                    z_trouve = True
                elif 'SpacingBetweenSlices' in pixel_seq:
                    print(f"   🎉 Voxel Z TROUVÉ (Caché dans PixelMeasures) : {pixel_seq.SpacingBetweenSlices} mm")
                    z_trouve = True
        
        # Etape 3 : Scan des textes privés (Dernier recours)
        if not z_trouve:
            print("   ❌ Voxel Z toujours introuvable. Scan des champs textes...")
            for element in ds:
                if element.VR in ['LO', 'SH', 'LT', 'UT']: 
                    valeur = str(element.value)
                    if 'step' in valeur.lower() or 'slice' in valeur.lower() or 'spacing' in valeur.lower():
                        print(f"      ❓ Indice potentiel (Tag {element.tag}) : {valeur[:50]}...")
            print("      -> Si aucun indice, assumer Z = X (Isotrope) pour cette machine.")

    except Exception as e:
        print(f"❌ Erreur critique DICOM : {e}")

def analyser_mat(chemin):
    nom_fichier = os.path.basename(chemin)
    print(f"\n--- 💾 ANALYSE FICHIER MATLAB : {nom_fichier} ---")
    
    if not os.path.exists(chemin):
        print("❌ Fichier introuvable.")
        return

    try:
        infos = scipy.io.whosmat(chemin)
        variables = [x[0] for x in infos]
        
        print(f"📦 Variables : {variables}")
        
        mots_cles = ['Resolution', 'resolution', 'Meta', 'meta', 'Spacing']
        trouve = [v for v in variables if any(m in v for m in mots_cles)]
        
        if trouve:
            print(f"✅ MÉTADONNÉES PRÉSENTES : Variable(s) '{trouve}' détectée(s).")
            print("   (Ce fichier est prêt pour l'analyse scientifique).")
        else:
            print("❌ MÉTADONNÉES ABSENTES : Fichier 'Orphelin' (Images seules).")
            print("   (Nécessite une reconstruction avec les paramètres du DICOM).")
            
    except Exception as e:
        print(f"❌ Erreur lecture MAT : {e}")

def main():
    print("=== DÉMARRAGE DU SUPER DIAGNOSTIC ===")
    for fichier in liste_fichiers:
        if fichier.endswith('.dcm'):
            analyser_dicom_complet(fichier)
        elif fichier.endswith('.mat'):
            analyser_mat(fichier)
        else:
            print(f"\n⚠️ Format ignoré : {fichier}")
    print("\n=== FIN DE L'ANALYSE ===")

if __name__ == "__main__":
    main()