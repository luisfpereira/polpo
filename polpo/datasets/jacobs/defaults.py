from pathlib import Path

from polpo.utils import in_frank

if in_frank():
    DATA_DIR = Path("/scratch/data/maternal")
else:
    DATA_DIR = Path("~/data/maternal").expanduser()


MATERNAL_PROJECT_FOLDER = "maternal_brain_project"
PILOT_PROJECT_FOLDER = f"{MATERNAL_PROJECT_FOLDER}_pilot"
PILOT_DATA_DIR = DATA_DIR / PILOT_PROJECT_FOLDER
