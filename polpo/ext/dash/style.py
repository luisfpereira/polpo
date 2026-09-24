class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


STYLE = AttrDict(
    text_fontsize="24px",
    text_fontfamily="Avenir",
)


def update_style(new_style):
    for key, value in new_style.items():
        STYLE[key] = value
