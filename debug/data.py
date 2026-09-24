import sys

# for project_pregnancy
sys.path.append("/home/luisfpereira/Repos/github/herbrain")

import hydra

from polpo.config import load_data


@hydra.main(version_base=None, config_path="./config", config_name="data")
def my_app(cfg):
    print(cfg.data.keys())

    data = load_data(cfg.data, name="data")

    # print(type(data["mesh"]))
    # print(data["mesh"].keys())

    # print(data["mesh_vertices"])
    print(data.keys())
    print(data["hormones"].keys())
    print(data["hormones_for_pred"].keys())

    print(data["hormones"])

    print("======")

    print(data["hormones_for_pred"])

    print(type(data["hormones_for_pred"]))


if __name__ == "__main__":
    my_app()
