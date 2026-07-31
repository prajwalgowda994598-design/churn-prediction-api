"""
run.py  -  launch the Churn Prediction API from the project root.

Usage:
    python run.py              # production mode
    python run.py --reload     # dev mode with auto-reload

This file exists solely to ensure the project root is on sys.path before
uvicorn imports api.main, so 'import api' resolves correctly regardless of
how or where the interpreter was launched.
"""

import os
import sys
from pathlib import Path

# Guarantee the project root (directory containing this file) is on sys.path.
# Inserting at index 0 takes priority over any site-packages 'api' namespace.
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    reload = "--reload" in sys.argv

    # --port NNNN  (default 8080)
    port = 8080
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    elif os.environ.get("PORT"):
        port = int(os.environ["PORT"])

    # --host X.X.X.X  (default 127.0.0.1; use 0.0.0.0 for LAN access)
    host = "127.0.0.1"
    if "--host" in sys.argv:
        host = sys.argv[sys.argv.index("--host") + 1]

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(ROOT)] if reload else None,
    )
