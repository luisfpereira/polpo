import json
import logging
from pathlib import Path

import polpo.pipeline.dict as ppdict
import polpo.utils as putils
from polpo.pipeline.load.enigma import get_all_structs
from polpo.pipeline.load.pregnancy.carmona import MeshLoader
from polpo.protocol.pairwise_varifold import PairwiseVarifold


def protocol_per_struct(struct, data_dir, results_dir, derivative="enigma"):
    # for serialization
    params = dict(
        struct=struct,
        derivative=derivative,
        data_dir=data_dir,
    )
    with open(results_dir / "params.json", "w") as file:
        json.dump(params, file, indent=4)

    known_correspondences = True if derivative == "enigma" else False

    dataset = (
        MeshLoader(
            data_dir=data_dir,
            struct_subset=[struct],
            derivative=derivative,
            as_mesh=True,
        )
        + ppdict.ExtractUniqueKey(nested=True)
    )()

    protocol = PairwiseVarifold(known_correspondences, results_dir)

    protocol.run(dataset)


if __name__ == "__main__":
    logging.basicConfig(
        filename="run.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    structs = get_all_structs(order=True)
    derivative = "enigma"

    data_dir = (
        "/scratch/data/maternal/neuromaternal_madrid_2021"
        if putils.in_frank()
        else Path.home() / ".herbrain/data/maternal/neuromaternal_madrid_2021"
    )

    for struct in structs:
        results_dir = (
            putils.get_results_path()
            / "pairwise_varifold"
            / "carmona"
            / f"{struct}_{derivative}"
        )
        if results_dir.exists():
            logging.info(f"Skipping {struct} because folder already exists")
            continue

        results_dir.mkdir(parents=True, exist_ok=False)

        try:
            protocol_per_struct(
                struct,
                data_dir=data_dir,
                results_dir=results_dir,
            )
        except Exception as e:
            logging.warning(f"Oops, something went wrong for {struct}: {e}")
