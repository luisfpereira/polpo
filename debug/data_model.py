import sys

# for project_pregnancy
sys.path.append("/home/luisfpereira/Repos/github/herbrain")

import hydra
import numpy as np

from polpo.config import load_data, load_models


@hydra.main(version_base=None, config_path="./config", config_name="data_model")
def my_app(cfg):
    print(cfg.keys())
    data = load_data(cfg.data, name="data")
    models = load_models(cfg.models)

    model = models["week_mesh_model"]

    print(model.predict(np.array([0.2])).shape)


if __name__ == "__main__":
    my_app()
