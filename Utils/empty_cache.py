import torch
import gc
import os

def clean_gpu_memory():
    print("--- Tentative de nettoyage de la VRAM ---")
    
    # 1. Suppression des variables inutilisées dans Python
    # Cela force Python à libérer les objets qui ne sont plus référencés
    gc.collect()
    
    # 2. Nettoyage spécifique à PyTorch
    if torch.cuda.is_available():
        print(f"Avant : {torch.cuda.memory_allocated() / 1024**2:.2f} MB utilisés (PyTorch)")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("✓ Cache PyTorch vidé.")
        print(f"Après : {torch.cuda.memory_allocated() / 1024**2:.2f} MB utilisés (PyTorch)")
    else:
        print("! PyTorch n'utilise pas de GPU sur cette machine.")

    # 3. Nettoyage spécifique à TensorFlow
    #   try:
    #       import tensorflow as tf
    #       # Note : TF est "têtu". Il ne vide pas le cache dynamiquement comme PyTorch.
    #       # On peut cependant essayer de réinitialiser les limitations de mémoire.
    #       from tensorflow.keras import backend as K
    #       K.clear_session()
    #       print("✓ Session TensorFlow nettoyée.")
    #    except ImportError:
    #       print("! TensorFlow n'est pas installé, passage à la suite.")
#
    #   print("---------------------------------------")
    #   print("Nettoyage terminé. Si l'erreur OOM persiste, redémarrez votre terminal.")

if __name__ == "__main__":
    clean_gpu_memory()