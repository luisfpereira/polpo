"""Package agnostic imports for MRI plotters."""

import abc

import numpy as np

try:
    from .pyplot import SlicesPlotter  # noqa:F401
except ImportError:
    pass

try:
    from .plotly import SlicePlotter  # noqa:F401
except ImportError:
    pass

from .base import Plotter


class Slicer(abc.ABC):
    @staticmethod
    def _resize(slices):
        # TODO: function instead?
        common_width = max([len(slice_[:, 0]) for slice_ in slices])
        common_height = max([len(slice_[0]) for slice_ in slices])

        for i_slice, slice_ in enumerate(slices):
            if len(slice_[:, 0]) < common_width:
                diff = common_width - len(slice_[:, 0])
                slice_ = np.pad(
                    slice_, ((diff // 2, diff // 2), (0, 0)), mode="constant"
                )
                slices[i_slice] = slice_
            if len(slice_[0]) < common_height:
                diff = common_height - len(slice_[0])
                slice_ = np.pad(
                    slice_, ((0, 0), (diff // 2, diff // 2)), mode="constant"
                )
                slices[i_slice] = slice_

        return slices


class MriSlicer(Slicer):
    def __init__(self, index_ordering=(0, 1, 2), common_size=True):
        self.index_ordering = index_ordering
        self.common_size = common_size

    def slice(self, img_fdata, slice_indices):
        slices = []
        for index, slice_index in zip(self.index_ordering, slice_indices):
            slicing_indices = [slice(None)] * 3
            slicing_indices[index] = slice_index
            slices.append(img_fdata[tuple(slicing_indices)])

        if self.common_size:
            slices = self._resize(slices)

        return slices


class SingleSliceMriSlicer(Slicer):
    def __init__(self, axis=0, common_size=True):
        # common_size to work properly with switching
        self.axis = axis
        self.common_size = common_size

    def slice(self, img_fdata, slice_index):
        slicing_indices = [slice(None)] * 3
        slicing_indices[self.axis] = slice_index
        img = img_fdata[tuple(slicing_indices)]

        slices = [img]
        if self.common_size:
            other_axes = [axis for axis in (0, 1, 2) if axis != self.axis]
            for axis in other_axes:
                slicing_indices = [slice(None)] * 3
                slicing_indices[axis] = slice_index
                slices.append(img_fdata[tuple(slicing_indices)])

            slices = self._resize(slices)

        return slices[0]


class MriPlotter(Plotter):
    def __init__(self, plotter=None, slicer=None, label_axes=True, add_title=True):
        if slicer is None:
            slicer = MriSlicer()

        if plotter is None:
            plotter = SlicesPlotter()

        self.slicer = slicer
        self.plotter = plotter
        self.label_axes = label_axes
        self.add_title = add_title

        self._default_labels = {0: ("Y", "Z"), 1: ("X", "Z"), 2: ("X", "Y")}
        self._default_titles = {0: "Side View", 1: "Front View", 2: " Top View"}

    @classmethod
    def build(
        cls,
        *args,
        slicer=None,
        index_ordering=(0, 1, 2),
        common_size=None,
        plotter=None,
        cmap="gray",
        **kwargs,
    ):
        # convenient instantiation
        if slicer is None:
            slicer = MriSlicer(index_ordering, common_size)

        if plotter is None:
            plotter = SlicesPlotter(cmap)

        return cls(*args, slicer=slicer, plotter=plotter, **kwargs)

    def plot(self, img_fdata, slice_indices):
        fig, axes = self.plotter.plot(self.slicer.slice(img_fdata, slice_indices))

        if self.label_axes:
            for ax, index in zip(axes, self.slicer.index_ordering):
                x_label, y_label = self._default_labels[index]
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

        if self.add_title:
            for ax, index in zip(axes, self.slicer.index_ordering):
                ax.set_title(self._default_titles[index])

        return fig, axes
