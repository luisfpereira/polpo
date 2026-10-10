def create_diretive_header(header_name, tab_size=3, **directive_params):
    tab = " " * tab_size
    return [
        f".. {header_name}::",
    ] + [
        f"{tab}:{key}: {value}" if value else f"{tab}:{key}:"
        for key, value in directive_params.items()
    ]


def create_directive_str(directive_name, filenames, tab_size=3, **directive_params):
    tab = " " * tab_size
    headers = create_diretive_header(
        directive_name, tab_size=tab_size, **directive_params
    )

    return "\n".join(headers) + f"\n\n{tab}" + f"\n{tab}".join(filenames)
