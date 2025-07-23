import abc

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib.colors import TwoSlopeNorm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from polpo.plot.mri import MriSlicer, SingleSliceMriSlicer
from polpo.preprocessing import IdentityStep
from polpo.sklearn.adapter import AdapterPipeline, MapTransformer
from polpo.sklearn.base import GetParamsMixin
from polpo.sklearn.compose import ObjectBasedTransformedTargetRegressor
from polpo.sklearn.dict import BiDictToValuesList
from polpo.sklearn.mesh import BiMeshesToVertices
from polpo.sklearn.np import BiFlattenButFirst, BiHstack
from polpo.sklearn.point_cloud import (
    FittableRegisteredPointCloudSmoothing,
)


def _to_list_with_false(obj):
    return [obj] if obj else []


def _clone_repeat(obj, n_reps):
    if isinstance(obj, (list, tuple)):
        if len(obj) != n_reps:
            raise ValueError(f"Inconsistent size for obj: {n_reps} != {len(obj)}")

        return obj

    if isinstance(obj, sklearn.base.BaseEstimator):
        return [sklearn.base.clone(obj) for _ in range(n_reps)]

    return [obj] * n_reps


class Model(abc.ABC):
    @abc.abstractmethod
    def predict(self, X):
        pass


class ModelFactory(abc.ABC):
    @abc.abstractmethod
    def create(self):
        # returns a `Model`
        pass


class ConstantOutput(Model):
    # for debugging
    def __init__(self, value):
        super().__init__()
        self.value = value

    def predict(self, X=None):
        return self.value


class ListLookup(Model):
    def __init__(self, data, tar=0):
        super().__init__()
        self.data = data
        self.tar = tar

    def predict(self, X):
        # NB: expects a (int,)
        return self.data[X[0] - self.tar]


class PdDfLookup(Model):
    def __init__(self, df, output_keys, tar=0):
        super().__init__()
        self.df = df
        self.output_keys = output_keys
        self.tar = tar

    def predict(self, X):
        # NB: expects a (int,)

        # TODO: make it input key based instead?
        df_session = self.df.iloc[X[0] - self.tar]
        df_session.columns = self.df.columns

        return [df_session.get(key) for key in self.output_keys]


class MriSlicesLookup(Model):
    def __init__(self, data, index_tar=1, slicer=None):
        if slicer is None:
            slicer = MriSlicer()

        self.data = data
        self.index_tar = index_tar
        self.slicer = slicer

    @classmethod
    def from_index_ordering(cls, data, index_tar=1, index_ordering=(0, 1, 2)):
        slicer = MriSlicer(index_ordering=index_ordering)
        return cls(data, index_tar, slicer)

    def predict(self, X):
        if len(X) == 2:  # to account for single index
            index, slice_indices = X
        else:
            index, *slice_indices = X

        datum = self.data[index - self.index_tar]
        return self.slicer.slice(datum, slice_indices)


class SwitchableMriSlicesLookup(Model):
    def __init__(self, data, index_tar=1, slicer=None):
        if slicer is None:
            slicer = SingleSliceMriSlicer()

        self.data = data
        self.index_tar = index_tar
        self.slicer = slicer

    @classmethod
    def from_axis(cls, data, index_tar=1, axis=0):
        slicer = SingleSliceMriSlicer(axis=axis)
        return cls(data, index_tar, slicer)

    def predict(self, X):
        if len(X) == 3:  # to account for single index
            axis, index, slice_indices = X
        else:
            axis, index, *slice_indices = X

        self.slicer.axis = axis

        datum = self.data[index - self.index_tar]
        return self.slicer.slice(datum, slice_indices)


def X2xPipeline(scaler=None, as_pipe=True):
    # useful for object regressor
    # NB: sklearn compatible pipelines
    steps = [np.asarray, np.atleast_2d]

    if scaler is not None:
        steps.append(scaler)

    if as_pipe:
        return AdapterPipeline(steps=steps)

    return steps


def Meshes2FlatVertices(smoother=False, as_pipe=True):
    if smoother is None:
        smoother = FittableRegisteredPointCloudSmoothing(n_neighbors=10)

    steps = (
        [BiMeshesToVertices()]
        + _to_list_with_false(smoother)
        + [
            FunctionTransformer(func=np.stack),
            BiFlattenButFirst(),
        ]
    )

    if as_pipe:
        return AdapterPipeline(steps=steps)

    return steps


def Meshes2Comps(
    scaler=None,
    dim_reduction=None,
    smoother=False,
    mesh_transform=None,
    as_pipe=True,
):
    # mesh_transform (e.g. AffineTransformation)
    if scaler is None:
        scaler = StandardScaler(with_std=False)

    if dim_reduction is None:
        dim_reduction = PCA()

    if smoother is None:
        smoother = FittableRegisteredPointCloudSmoothing(n_neighbors=10)

    if mesh_transform is not None:
        mesh_transform = FunctionTransformer(inverse_func=mesh_transform)

    steps = (
        _to_list_with_false(mesh_transform)
        + [BiMeshesToVertices(), FunctionTransformer(func=np.stack)]
        + _to_list_with_false(smoother)
        + [BiFlattenButFirst()]
        + _to_list_with_false(scaler)
        + [dim_reduction]
    )

    if as_pipe:
        return AdapterPipeline(steps=steps)

    return steps


def DictMeshes2y(pipes, as_pipe=True):
    steps = [
        BiDictToValuesList(),
        MapTransformer(pipes),
        BiHstack(),
    ]
    if as_pipe:
        return AdapterPipeline(steps=steps)

    return steps


def DictMeshes2FlatVertices(n_pipes, smoother=False, as_pipe=True):
    pipes = [
        Meshes2FlatVertices(smoother=smoother_)
        for smoother_ in _clone_repeat(smoother, n_reps=n_pipes)
    ]

    return DictMeshes2y(pipes, as_pipe=as_pipe)


def DictMeshes2Comps(
    n_pipes,
    scaler=None,
    dim_reduction=None,
    smoother=False,
    mesh_transform=None,
    as_pipe=True,
):
    # syntax sugar
    pipes = [
        Meshes2Comps(
            scaler=scaler_,
            dim_reduction=dim_reduction_,
            smoother=smoother_,
            mesh_transform=mesh_transform_,
        )
        for scaler_, dim_reduction_, smoother_, mesh_transform_ in zip(
            _clone_repeat(scaler, n_reps=n_pipes),
            _clone_repeat(dim_reduction, n_reps=n_pipes),
            _clone_repeat(smoother, n_reps=n_pipes),
            _clone_repeat(mesh_transform, n_reps=n_pipes),
        )
    ]

    return DictMeshes2y(pipes, as_pipe=as_pipe)


class ObjectRegressor(GetParamsMixin, AdapterPipeline):
    # just syntax sugar for sklearn.Pipeline

    def __init__(self, model, objs2y, x2x=None):
        # NB: recall sklearn.MultiOutputRegressor
        if model is None:
            model = LinearRegression()

        if x2x is None:
            x2x = X2xPipeline()

        tmodel = ObjectBasedTransformedTargetRegressor(
            regressor=model,
            transformer=objs2y,
            check_inverse=False,
        )

        super().__init__(
            steps=[
                ("pre", x2x),
                ("model", tmodel),
            ]
        )

    @property
    def objs2y(self):
        # TODO: check clone, because it seems to be fit after fit
        return self["model"].transformer

    @property
    def objs2y_(self):
        return self["model"].transformer_

    @property
    def regressor(self):
        return self["model"].regressor

    @property
    def regressor_(self):
        return self["model"].regressor_


class MeshColorizer:
    def __init__(self, x_ref=None, delta_lim=None, scaling_factor=1.0, cmap="coolwarm"):
        # uses mean if x_ref is None
        # uses max changes in data if delta_lim is None
        self.x_ref = x_ref
        self.delta_lim = delta_lim
        self.scaling_factor = scaling_factor

        self.ref_vertices_ = None
        self.color_norm_ = None

        self.cmap = plt.get_cmap(cmap)

    def _magnitude2color(self, value):
        return self.cmap(self.color_norm_(value * self.scaling_factor), bytes=True)

    def _computed_velocity_norm(self, meshes, ref_vertices, signed=True):
        # meshes: list[Trimesh]
        vel = np.array([mesh.vertices - ref_vertices for mesh in meshes])
        vel_norm = np.linalg.norm(vel, axis=-1)

        if not signed:
            return vel_norm

        signed_vel_norm = []
        for mesh, vel_, vel_norm_ in zip(meshes, vel, vel_norm):
            signed_vel_norm.append(
                np.sign(np.vecdot(mesh.vertex_normals, vel_, axis=-1)) * vel_norm_
            )

        return np.stack(signed_vel_norm)

    def fit(self, model, X=None, y=None):
        # y is ignored if delta_lim
        # X is ignored if x_ref
        # model: BaseEstimator

        x_ref = self.x_ref if self.x_ref is not None else np.mean(X, axis=0)
        ref_mesh = model.predict(x_ref)[0]
        self.ref_vertices_ = ref_mesh.vertices

        if self.delta_lim:
            lim_meshes = model.predict(x_ref + self.delta_lim)
        else:
            lim_meshes = y

        vel_norms = self._computed_velocity_norm(
            lim_meshes, self.ref_vertices_, signed=False
        )
        vmax = np.amax(vel_norms)

        self.color_norm_ = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        return self

    def __call__(self, meshes):
        # meshes: list[Trimesh]
        signed_vel_norms = self._computed_velocity_norm(
            meshes, self.ref_vertices_, signed=True
        )

        for mesh, signed_vel_norm in zip(meshes, signed_vel_norms):
            colors = self._magnitude2color(signed_vel_norm)

            mesh.visual.vertex_colors = colors.astype(np.uint8)

        return meshes


class DictMeshColorizer(MeshColorizer):
    def fit(self, model, X=None, y=None):
        # y is ignored if delta_lim
        # X is ignored if x_ref
        # model: BaseEstimator

        x_ref = self.x_ref if self.x_ref is not None else np.mean(X, axis=0)
        ref_mesh = model.predict(x_ref)
        self.ref_vertices_ = {
            key: meshes[0].vertices for key, meshes in ref_mesh.items()
        }

        if self.delta_lim:
            lim_meshes = model.predict(x_ref + self.delta_lim)
        else:
            lim_meshes = y

        self.color_norm_ = {}
        for key, lim_meshes_ in lim_meshes.items():
            vel_norms = self._computed_velocity_norm(
                lim_meshes_, self.ref_vertices_[key], signed=False
            )
            vmax = np.amax(vel_norms)
            self.color_norm_[key] = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        return self

    def __call__(self, meshes):
        # meshes: list[Trimesh]

        for key, meshes_ in meshes.items():
            signed_vel_norms = self._computed_velocity_norm(
                meshes_, self.ref_vertices_[key], signed=True
            )

            for mesh, signed_vel_norm in zip(meshes_, signed_vel_norms):
                colors = self._magnitude2color(signed_vel_norm, key)

                mesh.visual.vertex_colors = colors.astype(np.uint8)

        return meshes

    def _magnitude2color(self, value, key):
        return self.cmap(self.color_norm_[key](value * self.scaling_factor), bytes=True)


class SupervisedEmbeddingRegressor(BaseEstimator, RegressorMixin):
    # TODO: how shaky is this?

    def __init__(self, encoder, regressor):
        self.encoder = encoder
        self.regressor = regressor

        self.encoder_ = None
        self.regressor_ = None

    def fit(self, X, y):
        self.encoder_ = clone(self.encoder)
        self.regressor_ = clone(self.regressor)

        self.encoder_.fit(y, X)

        z = self.encoder_.transform(y)
        self.regressor_.fit(X, z)
        return self

    def predict(self, X):
        check_is_fitted(self)

        z_pred = self.regressor_.predict(X)
        return self.encoder_.inverse_transform(z_pred)

    def predict_eval(self, X, y):
        check_is_fitted(self)

        if hasattr(self.encoder_, "transform_eval"):
            yt = self.encoder_.transform_eval(y)
        else:
            yt = self.encoder_.transform(y)

        if hasattr(self.regressor_, "predict_eval"):
            z_pred = self.regressor_.predict_eval(X, yt)
        else:
            z_pred = self.regressor_.predict(X)

        return self.encoder_.inverse_transform(z_pred)


class SklearnLikeModelFactory(ModelFactory):
    # TODO: inherit from some pipeline based factory?

    def __init__(self, model, data, pipeline=None):
        self.model = model
        self.data = data

        if pipeline is None:
            pipeline = IdentityStep()

        self.pipeline = pipeline

    def create(self):
        X, y = self.pipeline(self.data)
        return self.model.fit(X=X, y=y)
