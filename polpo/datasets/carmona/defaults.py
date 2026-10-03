from pathlib import Path

from polpo.utils import in_frank

if in_frank():
    DATA_DIR = Path("/scratch/data/maternal/neuromaternal_madrid_2021")
else:
    DATA_DIR = Path("~/data/maternal/neuromaternal_madrid_2021").expanduser()
