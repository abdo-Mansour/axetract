import os
import json
import tempfile
import shutil

def atomic_write(filepath: str, data: dict) -> None:
    """Write JSON data atomically to avoid corruption on crash."""
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirpath)
    try:
        with os.fdopen(fd, "w") as tmp_file:
            json.dump(data, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        shutil.move(tmp_path, filepath)
    except Exception:
        os.remove(tmp_path)
        raise
