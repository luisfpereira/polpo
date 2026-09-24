from polpo.hydra import (
    instantiate_dict_from_config,
    key_value_instantiate,
)


def load_variables(variables_cfg, name=None):
    # syntax sugar
    return instantiate_dict_from_config(
        variables_cfg,
        name=name,
        instantiate_func=lambda key, value: key_value_instantiate(
            key, value, key_name="id_", targets=("polpo.dash.variables.VarDef",)
        ),
    )
