import os
import sys

# Ensure parent folder is importable when running in Vercel serverless runtime.
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from app import app

# Vercel looks for a variable named "app" for WSGI deployment.
