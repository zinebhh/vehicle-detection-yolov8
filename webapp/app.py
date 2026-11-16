"""
Site web : importer des vidéos (et des poids .pt), lancer la détection YOLOv8 et exporter
la vidéo annotée + les documents (CSV, JSON, rapport HTML, ZIP).

Lancement :  python webapp/app.py      puis ouvrir  http://localhost:5000
"""
import io
import json
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import detector

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
JOBS_DIR, MODELS_DIR, RUNS_DIR = BASE / "jobs", BASE / "models", ROOT / "runs"
for d in (JOBS_DIR, MODELS_DIR, RUNS_DIR):
    d.mkdir(exist_ok=True)

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
BUILTIN_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]       # COCO, téléchargés automatiquement
EXPORT_FILES = ["video_annote.mp4", "comptage.csv", "vehicules.csv", "rapport.json", "rapport.html"]
TRAIN_EXT = {".png", ".jpg", ".csv", ".json", ".yaml", ".pt"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024**3                    # 2 Go
pool = ThreadPoolExecutor(max_workers=1)                          # un traitement à la fois (GPU)
JOBS, LOCK = {}, threading.Lock()


# ------------------------------------------------------------------ persistance des tâches
def save(job):
    (JOBS_DIR / job["id"] / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")


def load_jobs():
    for f in JOBS_DIR.glob("*/job.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if j["status"] in ("queued", "running"):
            j.update(status="error", error="Interrompu (le serveur a été redémarré).")
        j.pop("cancel", None)
        JOBS[j["id"]] = j


def public(job):
    return {k: v for k, v in job.items() if k != "cancel"}


def available_models():
    out = []
    for p in sorted(RUNS_DIR.glob("*/weights/*.pt")):
        out.append({"id": p.relative_to(ROOT).as_posix(), "label": f"Fine-tuné · {p.parent.parent.name}/{p.name}"})
    for p in sorted(MODELS_DIR.glob("*.pt")):
        out.append({"id": p.relative_to(ROOT).as_posix(), "label": f"Importé · {p.name}"})
    out += [{"id": m, "label": f"COCO pré-entraîné · {m}"} for m in BUILTIN_MODELS]
    return out


# ------------------------------------------------------------------ traitement
def process(job_id):
    job = JOBS[job_id]
    jdir = JOBS_DIR / job_id
    if job["cancel"]:
        job.update(status="cancelled")
        return save(job)
    job.update(status="running", started=time.time())
    save(job)

    def progress(done, total, counts):
        job.update(frame=done, total_frames=total, counts=counts,
                   progress=round(done / total, 4) if total else 0)

    try:
        p = job["params"]
        weights = ROOT / job["weights"] if "/" in job["weights"] else job["weights"]
        report = detector.run(
            jdir / job["source_file"], jdir, weights, conf=p["conf"], iou=p["iou"], imgsz=p["imgsz"],
            line=p["line"], tracker=p["tracker"], device=p["device"] or None,
            progress=progress, should_stop=lambda: job["cancel"],
        )
        job.update(status="done", progress=1.0, report=report, counts=report["comptage"]["par_classe"],
                   files=[f for f in EXPORT_FILES if (jdir / f).exists()], finished=time.time())
    except detector.Cancelled:
        job.update(status="cancelled")
    except Exception as e:  # noqa: BLE001
        job.update(status="error", error=f"{type(e).__name__}: {e}")
    save(job)


# ------------------------------------------------------------------ pages & API
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def api_models():
    return jsonify(available_models())


@app.post("/api/models")
def api_upload_model():
    f = request.files.get("weights")
    if not f or not f.filename.lower().endswith(".pt"):
        return jsonify(error="Fichier .pt requis."), 400
    f.save(MODELS_DIR / secure_filename(f.filename))
    return jsonify(available_models())


@app.post("/api/jobs")
def api_create_job():
    v = request.files.get("video")
    if not v or Path(v.filename).suffix.lower() not in VIDEO_EXT:
        return jsonify(error="Vidéo requise (" + ", ".join(sorted(VIDEO_EXT)) + ")."), 400
    weights = request.form.get("weights", "yolov8s.pt")
    if weights not in {m["id"] for m in available_models()}:
        return jsonify(error="Modèle inconnu."), 400
    try:
        params = {
            "conf": min(max(float(request.form.get("conf", 0.35)), 0.01), 0.95),
            "iou": min(max(float(request.form.get("iou", 0.5)), 0.1), 0.95),
            "imgsz": int(request.form.get("imgsz", 640)),
            "line": min(max(float(request.form.get("line", 0.55)), 0.05), 0.95),
            "tracker": request.form.get("tracker", "bytetrack.yaml"),
            "device": request.form.get("device", "").strip(),
        }
    except ValueError:
        return jsonify(error="Paramètres invalides."), 400
    if params["tracker"] not in ("bytetrack.yaml", "botsort.yaml") or params["imgsz"] not in (320, 480, 640, 960, 1280):
        return jsonify(error="Paramètres invalides."), 400
    if params["device"] and params["device"] not in ("cpu", "mps") and not params["device"].replace(",", "").isdigit():
        return jsonify(error="Device invalide (cpu, 0, 0,1…)."), 400

    jid = uuid.uuid4().hex[:12]
    jdir = JOBS_DIR / jid
    jdir.mkdir()
    src = "source" + Path(v.filename).suffix.lower()
    v.save(jdir / src)
    job = {"id": jid, "video_name": secure_filename(v.filename) or src, "source_file": src, "weights": weights,
           "params": params, "status": "queued", "progress": 0, "frame": 0, "total_frames": 0, "counts": {},
           "created": time.time(), "cancel": False, "files": []}
    with LOCK:
        JOBS[jid] = job
    save(job)
    pool.submit(process, jid)
    return jsonify(public(job)), 201


@app.get("/api/jobs")
def api_jobs():
    return jsonify(sorted((public(j) for j in JOBS.values()), key=lambda j: -j["created"]))


def get_job(jid):
    job = JOBS.get(jid)
    if not job:
        abort(404)
    return job


@app.post("/api/jobs/<jid>/cancel")
def api_cancel(jid):
    get_job(jid)["cancel"] = True
    return jsonify(ok=True)


@app.delete("/api/jobs/<jid>")
def api_delete(jid):
    import shutil
    job = get_job(jid)
    if job["status"] in ("queued", "running"):
        return jsonify(error="Annulez d'abord la tâche."), 409
    shutil.rmtree(JOBS_DIR / jid, ignore_errors=True)
    JOBS.pop(jid, None)
    return jsonify(ok=True)


@app.get("/api/jobs/<jid>/files/<name>")
def api_file(jid, name):
    get_job(jid)
    if name not in EXPORT_FILES and not name.startswith("source."):
        abort(404)
    path = JOBS_DIR / jid / name
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=request.args.get("download") == "1",
                     download_name=f"{Path(JOBS[jid]['video_name']).stem}_{name}", conditional=True)


@app.get("/api/jobs/<jid>/bundle.zip")
def api_bundle(jid):
    job = get_job(jid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n in EXPORT_FILES:
            if (JOBS_DIR / jid / n).exists():
                z.write(JOBS_DIR / jid / n, n)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"{Path(job['video_name']).stem}_resultats.zip")


@app.get("/api/training")
def api_training():
    """Documents de l'entraînement (courbes, matrice de confusion, evaluation.json, poids)."""
    files = []
    for p in sorted(RUNS_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in TRAIN_EXT and len(p.relative_to(RUNS_DIR).parts) <= 3:
            files.append({"path": p.relative_to(RUNS_DIR).as_posix(), "size": p.stat().st_size,
                          "kind": "image" if p.suffix.lower() in (".png", ".jpg") else p.suffix.lower()[1:]})
    ev = RUNS_DIR / "evaluation.json"
    evaluation = json.loads(ev.read_text(encoding="utf-8")) if ev.exists() else None
    return jsonify(files=files, evaluation=evaluation)


@app.get("/api/training/<path:rel>")
def api_training_file(rel):
    p = (RUNS_DIR / rel).resolve()
    if not p.is_relative_to(RUNS_DIR.resolve()) or not p.is_file() or p.suffix.lower() not in TRAIN_EXT:
        abort(404)
    return send_file(p, as_attachment=request.args.get("download") == "1", conditional=True)


load_jobs()

if __name__ == "__main__":
    import os
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), threaded=True)
