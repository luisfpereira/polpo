import abc
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv


def _plot_index_to_shape(index, n_axis, rowise=False):
    a, b = index // n_axis, index % n_axis

    if rowise:
        return b, a

    return a, b


class _PvPlotter(abc.ABC):
    def __init__(self, **kwargs):
        self._pl = pv.Plotter(**kwargs)

    def __getattr__(self, name):
        """Get attribute.

        It is only called when ``__getattribute__`` fails.
        Delegates attribute calling to pl.
        """
        return getattr(self._pl, name)


class RegisteredMeshesColoredPlotter(_PvPlotter):
    # TODO: should this take a metric?
    # TODO: do geomfum version
    def _compute_vertex_dists(self, points_a, points_b):
        return np.linalg.norm(points_a - points_b, axis=-1)

    def add_meshes(
        self,
        mesh_a,
        mesh_b,
        ref_dist=None,
        name="vertex diffs",
        cmap="coolwarm",
        **kwargs,
    ):
        diffs = self._compute_vertex_dists(mesh_a.points, mesh_b.points)

        if ref_dist is not None:
            diffs = diffs / ref_dist

        vis_mesh = mesh_a.copy()
        vis_mesh[name] = diffs

        return self._pl.add_mesh(vis_mesh, cmap=cmap, **kwargs)


class RegisteredMeshesGifPlotter(_PvPlotter):
    # TODO: feature: two mesh sequences
    def __init__(
        self,
        gif_name=None,
        shape=(1, 1),
        fps=10,
        border=False,
        off_screen=True,
        notebook=False,
        subtitle=None,
        rowise=False,
        **kwargs,
    ):
        if gif_name is None:
            _, name = tempfile.mkstemp(suffix=".gif", dir=".")
            gif_name = Path(name).name

        if subtitle is True:
            subtitle = lambda x, y: str(y)

        self.fps = fps
        self.gif_name = gif_name
        self.subtitle = subtitle
        self.rowise = rowise

        self._gif = None

        self._subtitle_kwargs = {"font_size": 8}

        super().__init__(
            shape=shape,
            border=border,
            off_screen=off_screen,
            notebook=notebook,
            **kwargs,
        )

    def set_subtitle_kwargs(self, kwargs):
        self._subtitle_kwargs.update(kwargs)

        return self

    def _iter_meshes(self, meshes):
        if isinstance(meshes, dict):
            return meshes.items()

        return enumerate(meshes)

    def add_meshes(self, meshes, show_edges=True, **kwargs):
        """Add meshes to gif.

        Parameters
        ----------
        meshes : list[pv.Mesh] or list[list[pv.Mesh]]
            Axes: time, subplot.
        """
        # TODO: fix docstrings
        pl = self._pl

        if self._gif is None:
            self._gif = pl.open_gif(self.gif_name, fps=self.fps)

        if pl.shape[0] * pl.shape[1] == 1:
            if isinstance(meshes, dict):
                meshes = {key: [mesh] for key, mesh in meshes.items()}
            else:
                meshes = [[mesh] for mesh in meshes]

        subplot_axis = 0 if self.rowise else 1
        rendered_meshes = {}
        for time_index, (time_id, meshes_) in enumerate(self._iter_meshes(meshes)):
            for subplot_index, (comp_id, mesh) in enumerate(self._iter_meshes(meshes_)):
                pl.subplot(
                    *_plot_index_to_shape(
                        subplot_index, pl.shape[subplot_axis], rowise=self.rowise
                    )
                )

                if time_index:
                    rendered_meshes[subplot_index].points = mesh.points
                else:
                    rendered_meshes[subplot_index] = mesh_ = mesh.copy()
                    pl.add_mesh(mesh_, show_edges=show_edges, **kwargs)

                if callable(self.subtitle):
                    pl.add_title(
                        self.subtitle(time_id, comp_id),
                        **self._subtitle_kwargs,
                    )

            pl.write_frame()

    def close(self):
        self._pl.close()

    def show(self):
        # show in notebook
        from IPython.display import Image, display

        image = Image(open(self.gif_name, "rb").read())
        display(image)
