"""
Évaluation d'un modèle : mAP, précision, rappel par classe + vitesse d'inférence.
Exemple : python scripts/evaluate.py --weights runs/yolov8_vehicles/weights/best.pt
"""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/yolov8_vehicles/weights/best.pt")
    p.add_argument("--data", default="data/vehicles.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    model = YOLO(a.weights)
    m = model.val(data=a.data, imgsz=a.imgsz, conf=a.conf, device=a.device, plots=True)

    names = model.names
    per_class = {}
    print(f"\n{'Classe':<12}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for i, c in enumerate(m.box.ap_class_index):
        p_, r_, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = dict(precision=p_, recall=r_, map50=ap50, map50_95=ap)
        print(f"{names[int(c)]:<12}{p_:>8.3f}{r_:>8.3f}{ap50:>8.3f}{ap:>10.3f}")

    res = {
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "speed_ms": {k: float(v) for k, v in m.speed.items()},  # preprocess / inference / postprocess
        "per_class": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in per_class.items()},
    }
    out = Path("runs/evaluation.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nmAP50={res['mAP50']:.3f} | mAP50-95={res['mAP50-95']:.3f} | "
          f"inférence={res['speed_ms']['inference']:.1f} ms/img")
    print(f"[OK] Résultats sauvegardés : {out}")


if __name__ == "__main__":
    main()
