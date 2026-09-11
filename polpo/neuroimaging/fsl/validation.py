from polpo.neuroi.validation import validate_struct as _validate_struct
from polpo.neuroi.validation import validate_structs as _validate_structs

from .naming import get_all_subcortical_structs


def validate_struct(struct, prefixed=True):
    return _validate_struct(struct, get_all_subcortical_structs(prefixed=prefixed))


def validate_structs(structs, prefixed=True):
    return _validate_structs(structs, get_all_subcortical_structs(prefixed=prefixed))
