"""
Préparation du dataset de véhicules au format YOLO.

Deux sources possibles :
  1) coco     : sous-ensemble de COCO-2017 (car, motorcycle, bus, truck) via FiftyOne
  2) roboflow : dataset de trafic routier téléchargé depuis Roboflow Universe

Exemples :
  python scripts/prepare_dataset.py --source coco --max-train 4000 --max-val 800
  python scripts/prepare_dataset.py --source roboflow --api-key XXX --workspace WS --project PROJ --version 1

Résultat : data/vehicles.yaml (utilisable directement par train.py)
"""
import argparse
from pathlib import Path

CLASSES = ["car", "motorcycle", "bus", "truck"]
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def write_yaml(dataset_dir: Path, names):
    yaml_path = DATA_DIR / "vehicles.yaml"
    lines = [
        f"path: {dataset_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(names)}",
        "names:",
    ] + [f"  {i}: {n}" for i, n in enumerate(names)]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Fichier de configuration écrit : {yaml_path}")


def from_coco(max_train: int, max_val: int):
    import fiftyone as fo
    import fiftyone.zoo as foz
    from fiftyone import ViewField as F

    out = DATA_DIR / "coco_vehicles"
    for split, fo_split, n in [("train", "train", max_train), ("val", "validation", max_val)]:
        print(f"[INFO] Téléchargement COCO ({fo_split}) – max {n} images")
        ds = foz.load_zoo_dataset(
            "coco-2017",
            split=fo_split,
            label_types=["detections"],
            classes=CLASSES,
            max_samples=n,
            shuffle=True,
            seed=42,
        )
        # On ne garde que les 4 classes de véhicules
        view = ds.filter_labels("ground_truth", F("label").is_in(CLASSES), only_matches=True)
        view.export(
            export_dir=str(out),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            split=split,
            classes=CLASSES,
        )
    write_yaml(out, CLASSES)


def from_roboflow(api_key, workspace, project, version):
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(
        "yolov8", location=str(DATA_DIR / "roboflow_vehicles")
    )
    print(f"[OK] Dataset téléchargé dans {ds.location}")
    print("[INFO] Pointez --data de train.py vers le data.yaml présent dans ce dossier.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["coco", "roboflow"], default="coco")
    p.add_argument("--max-train", type=int, default=4000)
    p.add_argument("--max-val", type=int, default=800)
    p.add_argument("--api-key")
    p.add_argument("--workspace")
    p.add_argument("--project")
    p.add_argument("--version", type=int, default=1)
    a = p.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    if a.source == "coco":
        from_coco(a.max_train, a.max_val)
    else:
        from_roboflow(a.api_key, a.workspace, a.project, a.version)
