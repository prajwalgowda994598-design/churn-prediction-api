import sys
sys.path.insert(0, ".")
from api.main import app
print("app imported OK:", app.title)
