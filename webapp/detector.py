"""
Moteur de détection pour le site web : YOLOv8 + ByteTrack + comptage par ligne virtuelle.
Reprend la logique de scripts/detect_video.py, mais :
  - renvoie des résultats exploitables (JSON, CSV, rapport HTML) ;
  - vote majoritaire de la classe par véhicule (évite qu'un camion « devienne » une voiture) ;
  - compte les deux sens de circulation ;
  - ré-encode la vidéo en H.264 pour qu'elle soit lisible dans le navigateur.
"""
import csv
import html
import json
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2

VEHICLES = {"car", "motorcycle", "bus", "truck"}
COLORS = {"car": (0, 200, 0), "motorcycle": (0, 165, 255), "bus": (255, 120, 0), "truck": (0, 0, 230)}


class Cancelled(Exception):
    pass


def vehicle_class_ids(names):
    """Ids des classes véhicules, que le modèle soit fine-tuné (4 classes) ou COCO (80 classes)."""
    return [int(i) for i, n in names.items() if n in VEHICLES]


def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def to_h264(src: Path, dst: Path) -> bool:
    exe = _ffmpeg_exe()
    if exe:
        r = subprocess.run(
            [exe, "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
             "-crf", "23", "-movflags", "+faststart", "-an", str(dst)],
            capture_output=True,
        )
        if r.returncode == 0:
            src.unlink(missing_ok=True)
            return True
    shutil.move(str(src), str(dst))  # repli : mp4v (téléchargeable, parfois non lisible en ligne)
    return False


def _draw_box(frame, xyxy, label, color):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, max(y1, th + 6)), color, -1)
    cv2.putText(frame, label, (x1 + 2, max(y1 - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_panel(frame, counts, fps):
    lines = [f"FPS: {fps:.1f}", "Vehicules comptes :"] + [f"  {k}: {v}" for k, v in counts.items()]
    lines.append(f"  TOTAL: {sum(counts.values())}")
    h = 22 * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (250, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (12, 27 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def run(source, out_dir, weights, conf=0.35, iou=0.5, imgsz=640, line=0.55,
        tracker="bytetrack.yaml", device=None, progress=None, should_stop=None):
    from ultralytics import YOLO

    source, out_dir = Path(source), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Nouveau modèle à chaque vidéo : avec persist=True, le tracker garde son état entre deux appels,
    # donc réutiliser le même objet ferait « fuiter » les identifiants d'une vidéo à l'autre.
    model = YOLO(str(weights))
    names = model.names
    class_ids = vehicle_class_ids(names)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError("Impossible d'ouvrir la vidéo (format non pris en charge ou fichier corrompu).")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    raw = out_dir / "_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))
    line_y = int(H * line)

    last_y, counted = {}, set()
    votes = defaultdict(Counter)           # id -> votes de classe
    conf_sum, seen = defaultdict(float), defaultdict(int)
    first_f, last_f = {}, {}
    counts, directions = Counter(), Counter()
    events = []                            # franchissements de ligne
    track_dir = {}
    t0, fps, t_prev, idx = time.time(), 0.0, time.time(), -1

    stream = model.track(
        source=str(source), stream=True, persist=True, tracker=tracker, conf=conf, iou=iou,
        imgsz=imgsz, device=device or None, classes=class_ids or None, verbose=False,
    )
    try:
        for r in stream:
            idx += 1
            if should_stop and should_stop():
                raise Cancelled()
            frame = r.orig_img.copy()
            cv2.line(frame, (0, line_y), (W, line_y), (0, 255, 255), 2)

            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.int().cpu().tolist()
                clss = r.boxes.cls.int().cpu().tolist()
                confs = r.boxes.conf.cpu().tolist()
                for box, tid, c, cf in zip(boxes, ids, clss, confs):
                    name = names[c]
                    votes[tid][name] += 1
                    conf_sum[tid] += cf
                    seen[tid] += 1
                    first_f.setdefault(tid, idx)
                    last_f[tid] = idx
                    cy = (box[1] + box[3]) / 2
                    if tid in last_y and tid not in counted and (last_y[tid] - line_y) * (cy - line_y) <= 0 and last_y[tid] != cy:
                        counted.add(tid)
                        cls_name = votes[tid].most_common(1)[0][0]
                        sens = "descendant" if cy > last_y[tid] else "montant"
                        counts[cls_name] += 1
                        directions[sens] += 1
                        track_dir[tid] = sens
                        events.append({"image": idx, "temps_s": round(idx / src_fps, 2), "id": tid,
                                       "classe": cls_name, "sens": sens})
                    last_y[tid] = cy
                    _draw_box(frame, box, f"#{tid} {votes[tid].most_common(1)[0][0]} {cf:.2f}",
                              COLORS.get(name, (255, 255, 255)))

            now = time.time()
            inst = 1.0 / max(now - t_prev, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            t_prev = now
            _draw_panel(frame, dict(counts) if counts else {n: 0 for n in sorted(VEHICLES)}, fps)
            writer.write(frame)
            if progress and idx % 5 == 0:
                progress(idx + 1, total, dict(counts))
    finally:
        writer.release()
        try:
            stream.close()
        except Exception:
            pass

    to_h264(raw, out_dir / "video_annote.mp4")
    elapsed = time.time() - t0
    n_frames = idx + 1

    # --- Exports -----------------------------------------------------------------------------
    with open(out_dir / "comptage.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "temps_s", "id", "classe", "sens"], delimiter=";")
        w.writeheader()
        w.writerows(events)

    tracks = []
    for tid in sorted(seen):
        tracks.append({
            "id": tid, "classe": votes[tid].most_common(1)[0][0],
            "premiere_image": first_f[tid], "derniere_image": last_f[tid],
            "images_vues": seen[tid], "confiance_moyenne": round(conf_sum[tid] / seen[tid], 3),
            "compte": "oui" if tid in counted else "non", "sens": track_dir.get(tid, ""),
        })
    with open(out_dir / "vehicules.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(tracks[0].keys()) if tracks else ["id"], delimiter=";")
        w.writeheader()
        w.writerows(tracks)

    report = {
        "genere_le": datetime.now().isoformat(timespec="seconds"),
        "source": source.name,
        "poids": str(weights),
        "parametres": {"conf": conf, "iou": iou, "imgsz": imgsz, "ligne": line, "tracker": tracker},
        "video": {"largeur": W, "hauteur": H, "fps": round(src_fps, 2), "images": n_frames,
                  "duree_s": round(n_frames / src_fps, 1)},
        "comptage": {"par_classe": dict(counts), "total": int(sum(counts.values())),
                     "par_sens": dict(directions)},
        "vehicules_suivis": len(seen),
        "traitement": {"secondes": round(elapsed, 1), "fps_moyen": round(n_frames / max(elapsed, 1e-6), 1)},
    }
    (out_dir / "rapport.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "rapport.html").write_text(_html_report(report), encoding="utf-8")
    return report


def _html_report(r):
    e = html.escape
    rows = "".join(f"<tr><td>{e(k)}</td><td>{v}</td></tr>" for k, v in r["comptage"]["par_classe"].items())
    sens = "".join(f"<tr><td>{e(k)}</td><td>{v}</td></tr>" for k, v in r["comptage"]["par_sens"].items())
    p = r["parametres"]
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>Rapport de détection – {e(r['source'])}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:820px;margin:40px auto;color:#1f2937;padding:0 16px}}
h1{{color:#1d4ed8}}h2{{border-bottom:2px solid #e5e7eb;padding-bottom:4px;margin-top:28px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d1d5db;padding:6px 10px;text-align:left}}
th{{background:#eff6ff}}.big{{font-size:42px;font-weight:700;color:#1d4ed8}}</style></head><body>
<h1>Rapport de détection de véhicules</h1>
<p>Vidéo : <b>{e(r['source'])}</b> — généré le {e(r['genere_le'])}</p>
<h2>Résultat</h2><div class="big">{r['comptage']['total']} véhicules comptés</div>
<table><tr><th>Classe</th><th>Nombre</th></tr>{rows or '<tr><td colspan=2>Aucun</td></tr>'}</table>
<h2>Sens de circulation</h2><table><tr><th>Sens</th><th>Nombre</th></tr>{sens or '<tr><td colspan=2>–</td></tr>'}</table>
<h2>Vidéo</h2><table><tr><td>Résolution</td><td>{r['video']['largeur']}×{r['video']['hauteur']}</td></tr>
<tr><td>Durée</td><td>{r['video']['duree_s']} s ({r['video']['images']} images à {r['video']['fps']} i/s)</td></tr>
<tr><td>Véhicules suivis (IDs distincts)</td><td>{r['vehicules_suivis']}</td></tr></table>
<h2>Paramètres</h2><table><tr><td>Poids</td><td>{e(r['poids'])}</td></tr>
<tr><td>Confiance / IoU</td><td>{p['conf']} / {p['iou']}</td></tr><tr><td>Taille d'image</td><td>{p['imgsz']}</td></tr>
<tr><td>Ligne de comptage</td><td>{int(p['ligne']*100)} % de la hauteur</td></tr><tr><td>Tracker</td><td>{e(p['tracker'])}</td></tr></table>
<h2>Performance</h2><p>Traitement en {r['traitement']['secondes']} s — {r['traitement']['fps_moyen']} images/s en moyenne.</p>
</body></html>"""
