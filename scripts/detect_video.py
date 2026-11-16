"""
Détection + suivi (ByteTrack) + comptage de véhicules sur une vidéo ou une webcam.

Exemples :
  python scripts/detect_video.py --source videos/trafic.mp4 --weights runs/yolov8_vehicles/weights/best.pt
  python scripts/detect_video.py --source 0 --show            # webcam en temps réel
  python scripts/detect_video.py --source videos/trafic.mp4 --line 0.6   # ligne de comptage à 60 % de la hauteur

Touche : q = quitter
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

COLORS = {  # BGR
    "car": (0, 200, 0),
    "motorcycle": (0, 165, 255),
    "bus": (255, 120, 0),
    "truck": (0, 0, 230),
}


def draw_box(frame, xyxy, label, color):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_panel(frame, counts, fps):
    lines = [f"FPS: {fps:.1f}", "Vehicules comptes :"] + [f"  {k}: {v}" for k, v in counts.items()]
    lines.append(f"  TOTAL: {sum(counts.values())}")
    h = 22 * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (250, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (12, 27 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="videos/trafic.mp4", help="chemin vidéo ou index webcam (0)")
    p.add_argument("--weights", default="runs/yolov8_vehicles/weights/best.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml ou botsort.yaml")
    p.add_argument("--line", type=float, default=0.55, help="position de la ligne de comptage (0-1 de la hauteur)")
    p.add_argument("--output", default="runs/output.mp4")
    p.add_argument("--show", action="store_true", help="afficher la fenêtre en direct")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    source = int(a.source) if a.source.isdigit() else a.source
    model = YOLO(a.weights)
    names = model.names

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la source : {a.source}")
    W, H = int(cap.get(3)), int(cap.get(4))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))

    line_y = int(H * a.line)
    last_y = {}                     # id -> dernière position y du centre
    counted = set()                 # ids déjà comptés
    counts = defaultdict(int)       # classe -> total
    fps, t_prev = 0.0, time.time()

    stream = model.track(
        source=source, stream=True, persist=True, tracker=a.tracker,
        conf=a.conf, iou=a.iou, imgsz=a.imgsz, device=a.device, verbose=False,
    )

    for r in stream:
        frame = r.orig_img.copy()
        cv2.line(frame, (0, line_y), (W, line_y), (0, 255, 255), 2)

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()

            for box, tid, c, cf in zip(boxes, ids, clss, confs):
                name = names[c]
                cy = (box[1] + box[3]) / 2
                # Franchissement de la ligne (dans les deux sens)
                if tid in last_y and tid not in counted:
                    if (last_y[tid] - line_y) * (cy - line_y) <= 0 and last_y[tid] != cy:
                        counted.add(tid)
                        counts[name] += 1
                last_y[tid] = cy
                draw_box(frame, box, f"#{tid} {name} {cf:.2f}", COLORS.get(name, (255, 255, 255)))

        now = time.time()
        inst = 1.0 / max(now - t_prev, 1e-6)
        fps = 0.9 * fps + 0.1 * inst if fps else inst
        t_prev = now
        draw_panel(frame, dict(counts) if counts else {n: 0 for n in names.values()}, fps)

        writer.write(frame)
        if a.show:
            cv2.imshow("Detection de vehicules - YOLOv8", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    writer.release()
    cv2.destroyAllWindows()
    print(f"[OK] Vidéo annotée : {a.output}")
    print("Comptage final :", dict(counts), "| total =", sum(counts.values()))


if __name__ == "__main__":
    main()
