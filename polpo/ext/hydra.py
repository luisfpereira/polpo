import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError, UnsupportedInterpolationType


def register_dict_lookup_resolver(var_name, dict_):
    OmegaConf.register_new_resolver(var_name, lambda key: dict_[key])


def key_value_instantiate(key, value, key_name=None, targets=()):
    # creates kwargs with key_name if _target_ in targets
    kwargs = {}
    if (
        key_name is not None
        and isinstance(value, DictConfig)
        and value.get("_target_", None) in targets
    ):
        kwargs = {key_name: key}

    return instantiate(value, **kwargs)


def operate_after_instantiate(value, operation):
    instance = instantiate(value)
    if hasattr(instance, operation):
        return getattr(instance, operation)()

    return instance


def instantiate_dict_from_config(cfg, name=None, instantiate_func=None):
    # NB: name controls register
    if instantiate_func is None:
        instantiate_func = key_value_instantiate

    dict_ = {}
    missing = {}
    for key, value in cfg.items():
        try:
            # TODO: avoid exception?
            new_value = (
                value if isinstance(value, bool) else instantiate_func(key, value)
            )
            dict_[key] = new_value

        except (UnsupportedInterpolationType, InterpolationResolutionError):
            missing[key] = value

    if name is None:
        return dict_

    register_dict_lookup_resolver(name, dict_)

    while len(set(cfg.keys()) - set(dict_.keys())):
        other = instantiate_dict_from_config(missing, instantiate_func=instantiate_func)
        dict_.update(other)

    return dict_


def load_data(data_cfg, name=None):
    # syntax sugar
    return instantiate_dict_from_config(
        data_cfg,
        name=name,
        instantiate_func=lambda key, value: operate_after_instantiate(
            value, operation="load"
        ),
    )


def load_models(model_cfg, name=None):
    # syntax sugar
    return instantiate_dict_from_config(
        model_cfg,
        name=name,
        instantiate_func=lambda key, value: operate_after_instantiate(
            value, operation="create"
        ),
    )


class DefaultsResolver:
    """Defaults resolver.

    Checks use of `${var}` in main config file.
    If file does not exists, overrides to `default`.

    Avoids defining empty yaml files.
    """

    def _load_config_defaults(self, config_path, config_name="config"):
        # loads default section of main config
        with open(config_path / f"{config_name}.yaml", "r") as f:
            raw = yaml.safe_load(f)

        return raw.get("defaults", [])

    def _parse_dict_for_deps(self, key, config_defaults):
        # finds key_value (e.g. data: {key_value})
        # finds dep vars (e.g. objs: ${data})
        key_spec = f"${{{key}}}"

        key_value = None
        dep_keys = []
        for entry in config_defaults:
            if not isinstance(entry, dict):
                continue

            if key in entry:
                key_value = entry[key]
            elif key_spec in entry.values():
                dep_keys.append(list(entry.keys())[0])

        if key_value is None:
            raise ValueError("Can't find data key")

        return key_value, dep_keys

    def _check_overrides(self, key, key_value, dep_keys, overrides):
        # checks if key or dep_vars are overriden
        for override in overrides:
            key_, value_ = override.split("=")
            if key == key_:
                key_value = value_
            elif key_ in dep_keys:
                dep_keys.remove(key_)

        return key_value, dep_keys

    def _update_overrides(self, config_path, key_value, dep_keys, overrides):
        # updates overrides
        for var_ in dep_keys:
            if not (config_path / var_ / f"{key_value}.yaml").exists():
                overrides.append(f"{var_}=default")

        return overrides

    def resolve(self, config_path, key, overrides, config_name="config"):
        # NB: overrides are updated
        defaults_ls = self._load_config_defaults(config_path, config_name)

        key_value, dep_keys = self._parse_dict_for_deps(key, defaults_ls)
        key_value, dep_keys = self._check_overrides(key, key_value, dep_keys, overrides)
        return self._update_overrides(config_path, key_value, dep_keys, overrides)
