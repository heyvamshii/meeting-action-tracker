"""
Serve the Meeting Action Tracker on Modal, with storage that survives restarts.

Uses the same Modal secret as voice-ops-agent (create it once):
    python -m modal secret create groq GROQ_API_KEY=gsk_...

Then:
    python -m modal serve modal_app.py     # temporary URL, live-reloads while you edit
    python -m modal deploy modal_app.py    # permanent URL

Storage: the SQLite database, uploaded audio and saved transcripts live on a
Modal Volume mounted at /data. The first container copies in the demo
database (data/seed/meetings.db); after that, meetings you upload are kept
across restarts and redeploys. Delete the volume to start from the demo again:
    python -m modal volume delete meeting-action-tracker-data
"""

import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

PROJECT_DIR = Path(__file__).parent
# Container paths are Linux paths. Kept as strings: on Windows, Path("/data")
# would render as "\data", which Modal rejects.
REMOTE_DIR = "/root/app"
VOLUME_DIR = "/data"
DB_PATH = f"{VOLUME_DIR}/db/meetings.db"
PORT = 8000
COMMIT_EVERY_SECONDS = 30

# Private or local-only material that must never reach the image. This mirrors
# .gitignore: data/audio, data/db and data/transcripts hold real recordings and
# the working database with a real meeting in it. Modal uploads from disk, so
# .gitignore alone would not keep them out.
_SKIP_PARTS = {".venv", "venv", "__pycache__", ".git", ".claude",
               ".pytest_cache", ".ruff_cache"}
_SKIP_DATA_DIRS = ("data/audio/", "data/db/", "data/transcripts/")


def _ignore(path: Path) -> bool:
    posix = path.as_posix()
    return (
        bool(_SKIP_PARTS.intersection(path.parts))
        or path.name == ".env"
        or any(folder in posix for folder in _SKIP_DATA_DIRS)
    )


volume = modal.Volume.from_name("meeting-action-tracker-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(PROJECT_DIR / "requirements.txt"))
    # Absolute path: config.py joins it onto the project root, and joining an
    # absolute path returns it unchanged.
    .env({"DB_PATH": DB_PATH})
    .add_local_dir(PROJECT_DIR, REMOTE_DIR, copy=True, ignore=_ignore)
)

app = modal.App("meeting-action-tracker", image=image)


def _prepare_storage() -> None:
    """Seed the volume on first run and point the app's data folders at it."""
    app_dir, volume_dir, db_path = Path(REMOTE_DIR), Path(VOLUME_DIR), Path(DB_PATH)
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(app_dir / "data" / "seed" / "meetings.db", db_path)

    # Uploads and transcripts are written under data/ by the app; replacing
    # those folders with links onto the volume keeps them too.
    for name in ("audio", "transcripts"):
        persistent = volume_dir / name
        persistent.mkdir(parents=True, exist_ok=True)
        local = app_dir / "data" / name
        if local.is_symlink():
            continue
        if local.exists():
            shutil.rmtree(local)
        local.symlink_to(persistent, target_is_directory=True)

    volume.commit()


def _commit_periodically() -> None:
    """Flush volume changes regularly, not only when the container stops."""
    while True:
        time.sleep(COMMIT_EVERY_SECONDS)
        try:
            volume.commit()
        except Exception as exc:  # keep serving even if one commit fails
            print(f"volume commit failed: {exc!r}", flush=True)


@app.function(
    secrets=[modal.Secret.from_name("groq")],
    volumes={VOLUME_DIR: volume},
    cpu=1.0,
    memory=1024,           # MiB: Streamlit + pandas/plotly + audio compression
    scaledown_window=300,  # keep a warm container 5 min after the last visitor
    # A single writer: SQLite on a volume must not be written by two
    # containers at once, and every visitor should see the same meetings.
    max_containers=1,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(PORT, startup_timeout=180)
def dashboard() -> None:
    _prepare_storage()
    threading.Thread(target=_commit_periodically, daemon=True).start()
    subprocess.Popen(
        [
            "streamlit", "run", f"{REMOTE_DIR}/app/main.py",
            "--server.port", str(PORT),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            # Default is 200 MB; Groq's transcription API accepts 25 MB after
            # the app's own compression, so allow room for a long recording.
            "--server.maxUploadSize", "500",
        ],
        cwd=REMOTE_DIR,
    )
