def validate_struct(struct, all_structs):
    if struct not in all_structs:
        raise ValueError(
            f"Oops, `{struct}` is not available. Please, choose from: {','.join(all_structs)}"
        )


def validate_structs(structs, all_structs):
    for struct in structs:
        validate_struct(struct, all_structs)
