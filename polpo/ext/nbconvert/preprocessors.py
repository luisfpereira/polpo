from nbconvert.preprocessors import Preprocessor


class GifToPngMimeShim(Preprocessor):
    enabled = True

    def preprocess_cell(self, cell, resources, index):
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if isinstance(data, dict) and "image/gif" in data:
                data["image/png"] = data.pop("image/gif")
        return cell, resources
