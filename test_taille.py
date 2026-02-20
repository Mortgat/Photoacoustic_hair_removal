import os
from PIL import Image

dossier = "frames_sam2"
# On trie exactement comme le fait SAM 2 en interne
fichiers = sorted([f for f in os.listdir(dossier) if f.endswith('.jpg')])

print(f"📁 Fichier maître (lu par SAM 2 pour fixer la taille) : {fichiers[0]}")
img_prems = Image.open(os.path.join(dossier, fichiers[0]))
print(f"📏 Taille de ce fichier : {img_prems.size} (Largeur x Hauteur)\n")

print("🔍 Analyse de toutes les images en cours...")
tailles_trouvees = set()
fichiers_anormaux = []

for f in fichiers:
    taille = Image.open(os.path.join(dossier, f)).size
    tailles_trouvees.add(taille)
    if taille != (1024, 1024):
        fichiers_anormaux.append(f)

print(f"📊 Résolutions uniques présentes dans le dossier : {tailles_trouvees}")
if fichiers_anormaux:
    print(f"🚨 ATTENTION ! {len(fichiers_anormaux)} fichier(s) n'est/ne sont PAS en 1024x1024 :")
    # On n'affiche que les 5 premiers pour ne pas polluer
    print(fichiers_anormaux[:5])
else:
    print("✅ TOUTES les images sont strictement en 1024x1024.")