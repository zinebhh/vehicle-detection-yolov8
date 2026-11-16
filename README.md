# Détection de véhicules en temps réel avec YOLOv8 (fine-tuning)

**Objectif** : identifier et encadrer automatiquement les véhicules (voiture, moto, bus, camion)
dans des vidéos de trafic routier, en effectuant le *fine-tuning* d'un modèle YOLOv8 pré-entraîné sur COCO.
Le projet ajoute le **suivi multi-objets (ByteTrack)** et le **comptage** des véhicules par franchissement de ligne.

## 1. Structure

```
vehicle_detection/
├── data/vehicles.yaml          # configuration du dataset
├── scripts/
│   ├── prepare_dataset.py      # téléchargement/conversion du dataset (COCO ou Roboflow)
│   ├── train.py                # fine-tuning YOLOv8
│   ├── evaluate.py             # mAP, précision, rappel, vitesse
│   ├── detect_video.py         # détection + tracking + comptage sur vidéo/webcam
│   └── export_model.py         # export ONNX / TensorRT / OpenVINO
├── videos/                     # placez ici vos vidéos de trafic
├── runs/                       # résultats (poids, courbes, vidéo annotée)
└── requirements.txt
```

## 2. Installation

```bash
python -m venv venv && source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```
Un GPU NVIDIA est fortement recommandé (sinon utilisez **Google Colab**, GPU T4 gratuit :
`!pip install ultralytics fiftyone` puis lancez les mêmes commandes avec `!python ...`).

## 3. Exécution pas à pas

```bash
# 1) Dataset (≈ 4 000 images d'entraînement, 4 classes de véhicules issues de COCO)
python scripts/prepare_dataset.py --source coco --max-train 4000 --max-val 800

# 2) Fine-tuning (yolov8n = rapide ; yolov8s = bon compromis ; yolov8m = plus précis)
python scripts/train.py --model yolov8s.pt --epochs 50 --imgsz 640 --batch 16

# 3) Évaluation
python scripts/evaluate.py --weights runs/yolov8_vehicles/weights/best.pt

# 4) Détection temps réel sur vidéo (résultat : runs/output.mp4)
python scripts/detect_video.py --source videos/trafic.mp4 --show

# 5) (optionnel) Export pour accélérer l'inférence
python scripts/export_model.py --format onnx
```

Pour de meilleurs résultats sur des scènes spécifiques (caméra fixe, vue aérienne, nuit), utilisez un dataset
dédié : **UA-DETRAC**, **Roboflow Universe** (« vehicle detection traffic »), ou annotez vos propres images
(Roboflow / CVAT / LabelImg), puis `--source roboflow`.

## 4. Méthodologie

| Étape | Choix | Justification |
|---|---|---|
| Modèle | YOLOv8s pré-entraîné COCO | détecteur one-stage, bon compromis vitesse/précision |
| Transfert | gel des 10 premières couches (backbone) | conserve les caractéristiques génériques, accélère la convergence, limite le surapprentissage |
| Classes | car, motorcycle, bus, truck | classes COCO de véhicules routiers |
| Optimiseur | AdamW, lr0=1e-3, cosinus, warm-up 3 ép. | stabilité du fine-tuning |
| Augmentation | mosaïque, mixup, flip horizontal, HSV, échelle | robustesse à la luminosité et aux tailles variées |
| Arrêt | early stopping (patience 15) | évite le surapprentissage |
| Suivi | ByteTrack (intégré à Ultralytics) | ID persistants entre les images |
| Comptage | franchissement d'une ligne virtuelle | chaque ID compté une seule fois |

## 5. Métriques d'évaluation

- **Précision / Rappel** : fiabilité des détections / proportion de véhicules retrouvés.
- **mAP@0.5** et **mAP@0.5:0.95** : métriques standard de détection d'objets.
- **Vitesse (ms/image) et FPS** : critère du « temps réel » (≥ 25 FPS visé).
- Courbes générées automatiquement dans `runs/yolov8_vehicles/` : `results.png`, `confusion_matrix.png`,
  `PR_curve.png`, `F1_curve.png`, `val_batch*_pred.jpg`.

Les chiffres exacts dépendent de votre dataset et de votre matériel : reportez ici vos mesures issues de
`runs/evaluation.json`. À titre indicatif, YOLOv8s tourne en temps réel sur un GPU récent, et il faut passer
à `yolov8n` ou à l'export ONNX pour approcher le temps réel sur CPU.

## 6. Pistes d'amélioration

- Dataset spécifique à la caméra cible, avec images de nuit/pluie.
- Résolution plus élevée (`--imgsz 960`) pour les petits véhicules lointains.
- Estimation de vitesse, lecture de plaques, comptage par voie.
- Déploiement : TensorRT (`--format engine --half`), API FastAPI, flux RTSP.

## 7. Dépannage

- *CUDA out of memory* → réduire `--batch` ou `--imgsz`, ou utiliser `yolov8n.pt`.
- *Pas de GPU* → `--device cpu` (lent), ou Google Colab.
- *Vidéo non lisible* → convertir en MP4 H.264 ; vérifier le chemin `--source`.

## 8. Site web (import / export de vidéos et de documents)

```bash
pip install -r requirements.txt -r webapp/requirements-web.txt
python webapp/app.py          # ou run_web.sh / run_web.bat  →  http://localhost:5000
```

- **Importer** : vidéo de trafic (mp4, avi, mov, mkv, webm) et, si besoin, vos poids `best.pt`.
- **Régler** : confiance, position de la ligne de comptage (aperçu sur la vidéo), IoU, taille d'image, tracker.
- **Exporter** : vidéo annotée (H.264), `comptage.csv`, `vehicules.csv`, `rapport.json`, `rapport.html` (imprimable), ZIP.
- **Documents d'entraînement** : courbes, matrice de confusion, `evaluation.json` et poids du dossier `runs/`.
- Les tâches sont conservées dans `webapp/jobs/` (historique). Un seul traitement à la fois (GPU).
- Pour l'ouvrir à d'autres machines du réseau : `HOST=0.0.0.0 python webapp/app.py` (pas d'authentification : ne pas exposer sur Internet).
