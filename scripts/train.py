"""
Fine-tuning de YOLOv8 pré-entraîné (COCO) sur le dataset de véhicules.

Exemple :
  python scripts/train.py --model yolov8s.pt --epochs 50 --imgsz 640 --batch 16
"""
import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/vehicles.yaml")
    p.add_argument("--model", default="yolov8s.pt", help="yolov8n/s/m/l/x.pt (poids pré-entraînés COCO)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="0 pour GPU, cpu pour CPU (auto par défaut)")
    p.add_argument("--freeze", type=int, default=10, help="nb de couches du backbone gelées (0 = aucune)")
    p.add_argument("--patience", type=int, default=15, help="early stopping")
    p.add_argument("--name", default="yolov8_vehicles")
    a = p.parse_args()

    model = YOLO(a.model)  # transfert d'apprentissage : on repart des poids COCO

    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        freeze=a.freeze if a.freeze > 0 else None,
        patience=a.patience,
        project="runs",
        name=a.name,
        exist_ok=True,
        # Optimisation
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        # Augmentations adaptées au trafic routier
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0, mixup=0.1,
        close_mosaic=10,
        plots=True,
        seed=42,
    )

    metrics = model.val()
    print(f"\nmAP@0.5      : {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.3f}")
    print(f"Meilleurs poids : runs/{a.name}/weights/best.pt")


if __name__ == "__main__":
    main()
