import abc

import numpy as np
import plotly.graph_objects as go

from polpo.utils import unnest_list

from .base import Plotter


class GoPlotter(Plotter, abc.ABC):
    @abc.abstractmethod
    def transform_data(self, data):
        pass

    def update(self, fig, data):
        fig.update(data=self.transform_data(data))
        return fig


class SlicePlotter(GoPlotter):
    def __init__(
        self, cmap="gray", title=None, x_label=None, y_label=None, layout=None
    ):
        self.cmap = cmap

        if layout is None:
            layout = go.Layout(
                title_x=0.5,
                uirevision="constant",
                margin=dict(l=20, r=20, t=40, b=20),
            )

        defaults = {
            **({"title_text": title} if title is not None else {}),
            **(
                {
                    f"{key}axis_title_text": value
                    for key, value in [("x", x_label), ("y", y_label)]
                    if value is not None
                }
            ),
        }
        layout.update(defaults)

        self.layout = layout

    def transform_data(self, data):
        return [go.Heatmap(z=data.T, colorscale=self.cmap, showscale=False)]

    def plot(self, data=None):
        # Create heatmap trace for the current slice
        if data is None:
            return go.Figure(
                layout=self.layout,
            )

        fig = go.Figure(data=self.transform_data(data), layout=self.layout)

        width = int(len(data[:, 0]) * 1.5)
        height = int(len(data[0]) * 1.5)

        fig.update_layout(
            width=width,
            height=height,
        )

        return fig


class BaseMeshPlotter(GoPlotter, abc.ABC):
    def __init__(self, layout=None):
        if layout is True:
            layout = go.Layout(
                margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                scene=dict(
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                    zaxis=dict(showgrid=True),
                    aspectmode="manual",
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="z",
                    aspectratio=dict(x=1, y=1, z=1),
                ),
                uirevision="constant",
            )

        self.layout = layout

    @property
    def n_graphs(self):
        return 1

    def update_layout(self, data):
        # TODO: pass current layout?
        return self.layout

    def update(self, fig, data):
        # TODO: check layout updates
        fig.update(data=self.transform_data(data), layout=self.update_layout(data))
        return fig

    def plot(self, data=None):
        if data is None:
            return go.Figure(layout=self.layout)

        # TODO: ensure right ordering
        # TODO: need to check whether layout=None works
        return go.Figure(
            data=self.transform_data(data),
            layout=self.update_layout(data),
        )


class StaticMeshPlotter(BaseMeshPlotter):
    # NB: only visibility changes

    def __init__(self, mesh, visible=False):
        super().__init__()
        self._visible = visible
        self._data = self._create_data(mesh)

    def _create_data(self, mesh):
        vertices = mesh.vertices
        faces = mesh.faces

        return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            colorbar_title="z",
            color="black",
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.2,
            visible=self._visible,
            flatshading=False,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5),
        )

    def transform_data(self, data):
        # NB: only updates visibility
        if isinstance(data, bool):
            self._data["visible"] = data

        return [self._data]


class MeshPlotter(BaseMeshPlotter):
    def __init__(self):
        super().__init__()
        self.data_ = None

    def transform_data(self, data):
        """Convert multiple meshes into Plotly-compatible data."""
        if data is None:
            return self.data_

        if isinstance(data, bool):
            for datum in self.data_:
                datum["visible"] = data

            return self.data_

        mesh_pred = data.vertices
        faces = data.faces
        vertex_colors = data.visual.vertex_colors

        self.data_ = [
            go.Mesh3d(
                x=mesh_pred[:, 0],
                y=mesh_pred[:, 1],
                z=mesh_pred[:, 2],
                colorbar_title="z",
                vertexcolor=vertex_colors,
                # i, j and k give the vertices of triangles
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name="y",
            )
        ]
        return self.data_


class MeshesPlotter(BaseMeshPlotter):
    def __init__(
        self, plotters, overlay_plotter=None, bounds=None, overlay_bounds=None
    ):
        # NB: overlay_bounds are given priority if overlay is active

        super().__init__(layout=True)
        # TODO: warn if layout is not None?
        self.plotters = plotters
        self.overlay_plotter = overlay_plotter

        self.bounds = bounds
        self.overlay_bounds = overlay_bounds

        if self.bounds is not None:
            mins, maxs = self.bounds
            self.layout.scene.xaxis.range = [mins[0], maxs[0]]
            self.layout.scene.yaxis.range = [mins[1], maxs[1]]
            self.layout.scene.zaxis.range = [mins[2], maxs[2]]

    @property
    def n_graphs(self):
        return len(self.plotters) + self.has_overlay

    @property
    def has_overlay(self):
        return self.overlay_plotter is not None

    def transform_data(self, data):
        if not isinstance(data, list):
            # TODO: consider tuple
            data = [data]

        if self.has_overlay and len(data) < self.n_graphs:
            data = data + [None]

        out_data = unnest_list(
            [
                plotter.transform_data(data_)
                for plotter, data_ in zip(self.plotters + [self.overlay_plotter], data)
            ]
        )
        return out_data

    def update_layout(self, data):
        overlay_vis = self.has_overlay and data[-1]
        if (
            not isinstance(data, list)
            or not isinstance(data[0], bool)
            or (self.bounds is None and self.overlay_bounds is None)
            or (overlay_vis and self.overlay_bounds is None)
            or (not overlay_vis and self.bounds is None)
        ):
            return self.layout

        mins, maxs = self.overlay_bounds if overlay_vis else self.bounds
        self.layout.scene.xaxis.range = [mins[0], maxs[0]]
        self.layout.scene.yaxis.range = [mins[1], maxs[1]]
        self.layout.scene.zaxis.range = [mins[2], maxs[2]]

        return self.layout
