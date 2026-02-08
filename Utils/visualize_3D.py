import scipy.io
import numpy as np
import plotly.graph_objects as go
from skimage.measure import marching_cubes
import os

# ==========================================
# CONFIGURATION
# ==========================================
fichier_mat = r"dicom_data/dicom/4261_fromdcm_denoised_doux.mat"

# FACTEUR DE RÉDUCTION
# On réduit la résolution avant de calculer la surface
DOWNSAMPLE = 3  # On garde 1 pixel sur 3

# SEUIL D'ISO-SURFACE
SEUIL_RELATIF = 0.10 # 10% du max

# ==========================================

def load_volume(filepath):
    if not os.path.exists(filepath):
        print(f"❌ Fichier non trouvé : {filepath}")
        return None
    try:
        mat = scipy.io.loadmat(filepath)
        for k, v in mat.items():
            if isinstance(v, np.ndarray) and v.ndim == 3 and not k.startswith('__'):
                return v
    except Exception as e:
        print(f"Erreur : {e}")
    return None

def generate_surface_html():
    print("--- 🕸️ GÉNÉRATION DU MAILLAGE 3D (SURFACE) ---")
    vol = load_volume(fichier_mat)
    if vol is None: return

    # 1. Sous-échantillonnage
    print(f"Volume original : {vol.shape}")
    # On réduit uniformément. Z on réduit un peu moins (::2) pour garder la précision verticale
    vol_small = vol[::DOWNSAMPLE, ::DOWNSAMPLE, ::2] 
    print(f"Volume réduit : {vol_small.shape}")
    
    # 2. Calcul du seuil absolu
    val_max = np.max(vol_small)
    seuil_absolu = val_max * SEUIL_RELATIF
    print(f"Seuil de surface : {seuil_absolu:.2f} (Max={val_max})")

    # 3. Marching Cubes
    print("Calcul de la surface géométrique (Marching Cubes)...")
    try:
        verts, faces, normals, values = marching_cubes(vol_small, level=seuil_absolu)
        print(f"Surface générée : {len(verts)} sommets, {len(faces)} triangles.")
    except Exception as e:
        print(f"Erreur Marching Cubes (Seuil trop haut ?): {e}")
        return

    # 4. Création de la figure Plotly (Mesh3d)
    
    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 1], # Y du volume devient X visuel
        y=verts[:, 0], # X du volume devient Y visuel
        
        # --- MODIFICATION ICI ---
        # On met un signe moins (-) devant verts[:, 2] pour inverser l'axe vertical
        z=-verts[:, 2], 
        # ------------------------
        
        # Définition des triangles
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        
        opacity=0.8, 
        color='orange',
        flatshading=True,
        lighting=dict(ambient=0.4, diffuse=0.5, roughness=0.1, specular=0.4, fresnel=0.1),
    )])

    fig.update_layout(
        title=f"Skin surface (threshold {SEUIL_RELATIF*100}%)",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data' # Respecte les proportions physiques
        )
    )

    output_file = "surface_skin_3d.html"
    print(f"💾 Sauvegarde dans {output_file}...")
    fig.write_html(output_file)
    print("✅ TERMINE ! Ouvre le fichier HTML.")

if __name__ == "__main__":
    generate_surface_html()