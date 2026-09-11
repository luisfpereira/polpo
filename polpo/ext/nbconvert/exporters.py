from nbconvert.exporters import Exporter

_base_init_preprocessors = Exporter._init_preprocessors


def _init_preprocessors(self):
    _base_init_preprocessors(self)

    self.register_preprocessor(
        "polpo.nbconvert.preprocessors.GifToPngMimeShim", enabled=True
    )
    self._preprocessors = [self._preprocessors.pop()] + self._preprocessors


Exporter._init_preprocessors = _init_preprocessors
