from polpo.preprocessing.base import PreprocessingStep


class ApplyTransformation(PreprocessingStep):
    def __call__(self, data):
        # TODO: accept transformation at init?
        points, transformation = data

        rotation_matrix = transformation[:3, :3]
        translation = transformation[:3, 3]

        return (rotation_matrix @ points.T).T + translation
