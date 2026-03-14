import numpy as np
import plotly.graph_objects as go
import os

# ==========================================
# ⚙️ CONFIGURATION DES PARAMÈTRES
# ==========================================
fichier_volume_original = r"dicom_data/dicom/4261_fromdcm.npz" # Ajouté pour récupérer la profondeur totale nz
fichier_peau = r"pipeline/body_hair_extraction_methods/surface_peau.npy"
fichier_poils_predits = r"pipeline/body_hair_extraction_methods/masque_predit_pu.npy"
output_file = r"pipeline/body_hair_extraction_methods/visualisation_prediction_reseau_3d.html"

DOWNSAMPLE = 1

def generate_prediction_html():
    print("--- 🕸️ GÉNÉRATION DE LA VISUALISATION 3D PLOTLY ---")

    if not os.path.exists(fichier_poils_predits) or not os.path.exists(fichier_peau):
        print("❌ Fichiers introuvables. Vérifie les chemins.")
        return

    print("Chargement des données...")
    masque_predit = np.load(fichier_poils_predits)
    surface_z = np.load(fichier_peau).astype(float) 
    nx, ny = surface_z.shape
    
    # On charge le volume original juste pour récupérer nz et forcer la bonne échelle Z
    try:
        nz = np.load(fichier_volume_original)['volume'].shape[2]
    except:
        nz = np.nanmax(surface_z) # Fallback si le fichier n'est pas là

    # --- 🔨 DESTRUCTION DES MURS ---
    # On met à NaN tout ce qui est proche du fond absolu (vide)
    profondeur_max = np.nanmax(surface_z)
    surface_z[surface_z >= profondeur_max - 5] = np.nan

    print("Préparation de la géométrie de la peau...")
    x_grid = np.arange(0, nx, DOWNSAMPLE)
    y_grid = np.arange(0, ny, DOWNSAMPLE)
    Z_peau = surface_z[::DOWNSAMPLE, ::DOWNSAMPLE]

    print("Extraction des poils...")
    masque_reduit = masque_predit[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]
    X_poils, Y_poils, Z_poils = np.where(masque_reduit == 1)

    X_poils = X_poils * DOWNSAMPLE
    Y_poils = Y_poils * DOWNSAMPLE
    Z_poils = Z_poils * DOWNSAMPLE

    nb_poils = len(X_poils)
    print(f"Éléments trouvés : {nb_poils} voxels de poils.")
    if nb_poils == 0:
        print("⚠️ ATTENTION : Le masque prédit est vide. Aucun poil rouge ne sera affiché.")

    fig = go.Figure()

    # 1. Peau en Vert Fluo
    fig.add_trace(go.Surface(
        x=x_grid,
        y=y_grid,
        z=-Z_peau, 
        colorscale=[[0, '#39FF14'], [1, '#39FF14']], 
        opacity=0.6,
        showscale=False,
        name='Surface Peau',
        hoverinfo='skip',
        lighting=dict(ambient=0.8, diffuse=0.2, roughness=0.1, specular=0.0) # Éclairage aplati pour garder la couleur fluo
    ))

    # 2. Poils en Rouge
    if nb_poils > 0:
        fig.add_trace(go.Scatter3d(
            x=X_poils,
            y=Y_poils,
            z=-Z_poils,
            mode='markers',
            marker=dict(
                size=4, 
                color='red', 
                opacity=1.0, 
                symbol='square'
            ),
            name='Poils (Prédiction)'
        ))

    # Dimensionnement forcé
    fig.update_layout(
        title="Résultat PU Learning - Poils (Rouge) & Peau (Vert Fluo)",
        scene=dict(
            xaxis=dict(title='X', range=[0, nx]),
            yaxis=dict(title='Y', range=[0, ny]),
            zaxis=dict(title='Z (Profondeur)', range=[-nz, 0]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=ny/nx, z=nz/nx) # Force les proportions physiques exactes
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"💾 Sauvegarde dans : {output_file}...")
    fig.write_html(output_file)
    print("✅ TERMINE !")

if __name__ == "__main__":
    generate_prediction_html()