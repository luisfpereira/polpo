import nibabel

from polpo.preprocessing.base import PreprocessingStep

read = nibabel.freesurfer.read_geometry


class FreeSurferReader(PreprocessingStep):
    def __init__(self, read_metadata=False, read_stamp=False):
        self.read_metadata = read_metadata
        self.read_stamp = read_stamp

    def __call__(self, filename):
        """Apply step.

        Parameters
        ----------
        filename : str
            File name.

        Returns
        -------
        vertices : np.array
            Mesh vertices.
        faces : np.array
            Mesh faces.
        """
        return read(
            filename,
            read_metadata=self.read_metadata,
            read_stamp=self.read_stamp,
        )
