import dash_bootstrap_components as dbc
import numpy as np
from dash import Dash
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from polpo.dash.components import (
    ComponentGroup,
    Graph,
    MultiModelsMeshExplorer,
    Slider,
)
from polpo.dash.style import update_style
from polpo.dash.variables import VarDef
from polpo.models import (
    DictMeshColorizer,
    DictMeshes2Comps,
    MeshColorizer,
    Meshes2Comps,
    ObjectRegressor,
)
from polpo.plot.mesh import MeshesPlotter, MeshPlotter, StaticMeshPlotter
from polpo.preprocessing import (
    IndexMap,
    ListSqueeze,
    Map,
    NestingSwapper,
    PartiallyInitializedStep,
    Pipeline,
    WrapInList,
)
from polpo.preprocessing import (
    dict as ppdict,
)
from polpo.preprocessing import pd as ppd
from polpo.preprocessing.load.pregnancy import (
    DenseMaternalCsvDataLoader,
    DenseMaternalMeshLoader,
    DenseMaternalSegmentationsLoader,
    PregnancyPilotMriLoader,
    PregnancyPilotRegisteredMeshesLoader,
    PregnancyPilotSegmentationsLoader,
)
from polpo.preprocessing.mesh.conversion import (
    TrimeshFromData,
    TrimeshFromPvMesh,
)
from polpo.preprocessing.mesh.io import PvReader, TrimeshReader
from polpo.preprocessing.mesh.registration import PvAlign
from polpo.preprocessing.mesh.transform import AffineTransformation
from polpo.preprocessing.mri import (
    LocalToTemplateTransform,
    MriImageLoader,
    SkimageMarchingCubes,
    segmtool2encoding,
)
from polpo.sklearn.compose import PostTransformingEstimator


def _load_homornes_df():
    return Pipeline(
        steps=[
            DenseMaternalCsvDataLoader(pilot=True),
            ppd.Drop(labels=27),
        ]
    )()


def _load_session_week(hormones_df):
    session_week_dict = Pipeline(
        steps=[
            ppd.ColumnsSelector(column_names="gestWeek"),
            ppd.SeriesToDict(),
        ],
    )(hormones_df)

    return session_week_dict


def _load_hormones_ordering():
    return ["estro", "prog", "lh"]


def _load_session_hormones(hormones_df, hormones_ordering):
    return Pipeline(
        steps=[
            ppd.ColumnsSelector(column_names=hormones_ordering),
            ppd.Dropna(),
            ppd.DfToDict(orient="index"),
        ]
    )(hormones_df)


def _load_hipp_meshes():
    return Pipeline(
        steps=[
            PregnancyPilotRegisteredMeshesLoader(method="deformetrica", as_dict=True),
            ppdict.DictMap(step=TrimeshReader()),
        ]
    )()


def _load_overlay_image():
    image_loader = Pipeline(
        steps=[
            PregnancyPilotMriLoader(subset=[1]),
            ListSqueeze(),
            MriImageLoader(return_affine=True),
        ]
    )

    return image_loader()


def _load_reference_image(data="hipp"):
    if data == "hipp":
        return Pipeline(
            steps=[
                PregnancyPilotSegmentationsLoader(subset=[1]),
                ListSqueeze(),
                MriImageLoader(return_affine=True),
            ]
        )()

    # FIXME: this leads to identity
    return Pipeline(
        steps=[
            DenseMaternalSegmentationsLoader(subset=[1]),
            ListSqueeze(),
            MriImageLoader(return_affine=True),
        ]
    )()


def _load_overlay_mesh(overlay_image):
    img2mesh = Pipeline(
        steps=[
            SkimageMarchingCubes(return_values=False),
            TrimeshFromData(),
        ]
    )
    return img2mesh(overlay_image)


def _load_maternal_pilot():
    return Pipeline(
        steps=[
            DenseMaternalMeshLoader(subject_id=None, as_dict=True),
            ppdict.DictMap(step=PvReader()),
            PartiallyInitializedStep(
                Step=lambda target, max_iterations: ppdict.DictMap(
                    step=PvAlign(target=target, max_iterations=max_iterations)
                ),
                _target=lambda meshes: meshes[1],  # template mesh
                max_iterations=500,
            ),
            ppdict.DictMap(step=TrimeshFromPvMesh()),
        ]
    )()


def _load_maternal_multiple(subject_id="01", tool="fsl"):
    encoding = segmtool2encoding(tool)

    struct_keys = encoding.structs

    mesh_loader = ppdict.HashWithIncoming(
        Map(
            PartiallyInitializedStep(
                Step=DenseMaternalMeshLoader,
                pass_data=False,
                subject_id=subject_id,
                _struct=lambda name: name.split("_")[-1],
                _left=lambda name: name.split("_")[0] == "L",
                as_dict=True,
            )
            + ppdict.DictMap(PvReader())
        )
    )

    prep_pipe = PartiallyInitializedStep(
        Step=lambda **kwargs: ppdict.DictMap(PvAlign(**kwargs) + TrimeshFromPvMesh()),
        _target=lambda meshes: meshes[list(meshes.keys())[0]],
        max_iterations=500,
    )

    pipe = mesh_loader + ppdict.DictMap(prep_pipe)

    meshes = pipe(struct_keys)

    return meshes


def _merge_session_week_meshes(session_week, registered_meshes):
    pipeline = Pipeline(
        steps=[
            ppdict.DictMerger(),
            NestingSwapper(),
            IndexMap(index=0, step=Map(step=WrapInList())),
        ]
    )

    X, meshes = pipeline((session_week, registered_meshes))

    return X, meshes


def _instantiate_mesh_model(X, y, mesh_transform=None, colorizer=False):
    model = ObjectRegressor(
        model=LinearRegression(),
        objs2y=Meshes2Comps(
            dim_reduction=PCA(n_components=4),
            smoother=False,
            mesh_transform=mesh_transform,
        ),
    )

    if colorizer:
        model = PostTransformingEstimator(model, colorizer)

    model.fit(X, y)

    return model


def _instantiate_week_mesh_model(
    session_week, registered_meshes, mesh_transform=None, colorized=False
):
    X, y = _merge_session_week_meshes(session_week, registered_meshes)

    colorizer = (
        MeshColorizer(
            x_ref=np.array(0.0),
            delta_lim=np.array(15.0),
        )
        if colorized
        else None
    )

    return _instantiate_mesh_model(
        X, y, mesh_transform=mesh_transform, colorizer=colorizer
    )


def _merge_session_week_multi_meshes(X, registered_meshes):
    dict_pipe = (
        IndexMap(ppdict.NestedDictSwapper(), index=1)
        + ppdict.DictMerger()
        + NestingSwapper()
        + IndexMap(lambda x: np.asarray(x)[:, None], index=0)
        + IndexMap(ppdict.ListDictSwapper(), index=1)
    )

    # meshes_ : dict[list]
    X, meshes_ = dict_pipe([X, registered_meshes])

    return X, meshes_


def _instantiate_multi_mesh_model(X, y, mesh_transform=None, colorizer=None):
    n_structs = len(y)
    pca = PCA(n_components=4)
    objs2y = DictMeshes2Comps(
        n_pipes=n_structs, dim_reduction=pca, mesh_transform=mesh_transform
    )
    inner_model = LinearRegression(fit_intercept=True)
    model = ObjectRegressor(inner_model, objs2y=objs2y)

    if colorizer:
        model = PostTransformingEstimator(model, colorizer)

    model.fit(X, y)

    return model


def _instantiate_week_multi_mesh_model(
    session_week, registered_meshes, mesh_transform=None, colorized=False
):
    X, y = _merge_session_week_multi_meshes(session_week, registered_meshes)

    colorizer = (
        DictMeshColorizer(
            x_ref=np.array(0.0),
            delta_lim=np.array(15.0),
        )
        if colorized
        else None
    )

    return _instantiate_multi_mesh_model(
        X, y, mesh_transform=mesh_transform, colorizer=colorizer
    )


def _merge_session_hormones_meshes(session_hormones, registered_meshes):
    X, meshes = Pipeline(
        steps=[
            ppdict.DictMerger(),
            NestingSwapper(),
            IndexMap(index=0, step=Map(step=ppdict.DictToValuesList())),
        ]
    )((session_hormones, registered_meshes))

    return X, meshes


def _instantiate_hormones_mesh_model(
    session_hormones,
    registered_meshes,
    mesh_transform=None,
    colorized=False,
):
    X, y = _merge_session_hormones_meshes(session_hormones, registered_meshes)

    colorizer = (
        MeshColorizer(
            delta_lim=None,
            scaling_factor=2.0,
        )
        if colorized
        else None
    )

    return _instantiate_mesh_model(
        X, y, mesh_transform=mesh_transform, colorizer=colorizer
    )


def _merge_session_hormones_multi_meshes(X, registered_meshes):
    dict_pipe = (
        IndexMap(ppdict.NestedDictSwapper(), index=1)
        + ppdict.DictMerger()
        + NestingSwapper()
        + IndexMap(index=0, step=Map(step=ppdict.DictToValuesList()))
        + IndexMap(lambda x: np.asarray(x), index=0)
        + IndexMap(ppdict.ListDictSwapper(), index=1)
    )

    # meshes_ : dict[list]
    X, meshes_ = dict_pipe([X, registered_meshes])

    return X, meshes_


def _instantiate_hormones_multi_mesh_model(
    session_hormones, registered_meshes, mesh_transform=None, colorized=False
):
    X, y = _merge_session_hormones_multi_meshes(session_hormones, registered_meshes)

    colorizer = (
        DictMeshColorizer(
            scaling_factor=2.0,
        )
        if colorized
        else None
    )

    return _instantiate_multi_mesh_model(
        X, y, mesh_transform=mesh_transform, colorizer=colorizer
    )


def _create_week_inputs():
    gest_week = VarDef(
        id_="GestWeek",
        name="Gestational Week",
        min_value=0,
        max_value=36,
        default_value=15,
    )

    return Slider(gest_week)


def _create_hormones_inputs(hormones_ordering):
    estro = VarDef(
        id_="estro",
        name="Estrogen",
        unit="pg/ml",
        min_value=4100,
        max_value=12400,
    )
    prog = VarDef(
        id_="prog",
        name="Progesterone",
        unit="ng/ml",
        min_value=54,
        max_value=103,
    )
    lh = VarDef(
        id_="lh",
        name="LH",
        unit="ng/ml",
        min_value=0.59,
        max_value=1.45,
    )
    return ComponentGroup(
        ordering=hormones_ordering,
        components=[
            Slider(
                var_def=estro,
                step=500,
                label_style={"fontSize": 30, "display": "block"},
            ),
            Slider(
                var_def=prog,
                step=3,
                label_style={"fontSize": 30, "display": "block"},
            ),
            Slider(
                var_def=lh,
                step=0.05,
                label_style={"fontSize": 30, "display": "block"},
            ),
        ],
    )


Key2MeshLoader = {
    "hipp": _load_hipp_meshes,
    "maternal": _load_maternal_pilot,
    "multiple": _load_maternal_multiple,
}

Key2WeekModelInstantiator = {
    "hipp": _instantiate_week_mesh_model,
    "maternal": _instantiate_week_mesh_model,
    "multiple": _instantiate_week_multi_mesh_model,
}

Key2HormonesModelInstantiator = {
    "hipp": _instantiate_hormones_mesh_model,
    "maternal": _instantiate_hormones_mesh_model,
    "multiple": _instantiate_hormones_multi_mesh_model,
}


def _create_layout(
    data="hipp",
    hideable=False,
    overlay=False,
    week=True,
    hormones=True,
    colorized=False,
):
    # hideable only applies to multiple

    if not (week or hormones):
        raise ValueError("At least week or hormones")

    inputs = []
    models = []

    registered_meshes = Key2MeshLoader[data]()

    postproc_pred = None
    checkbox_labels = []
    overlay_plotter = None

    affine_transform = None
    if overlay:
        overlay_image, overlay_affine = _load_overlay_image()
        overlay_mesh = _load_overlay_mesh(overlay_image)
        _, reference_affine = _load_reference_image(data)

        # NB: identity if not pilot
        affine_mat = LocalToTemplateTransform(template_affine=overlay_affine)(
            reference_affine
        )
        affine_transform = Map(AffineTransformation(affine_mat))

        overlay_plotter = StaticMeshPlotter(overlay_mesh)
        checkbox_labels = [(-1, "Show Full Brain", False)]

    hormones_df = _load_homornes_df()

    if week:
        session_week_data = _load_session_week(hormones_df)

        inputs.append(_create_week_inputs())

        models.append(
            Key2WeekModelInstantiator[data](
                session_week_data,
                registered_meshes,
                mesh_transform=affine_transform,
                colorized=colorized,
            )
        )

    if hormones:
        hormones_ordering = _load_hormones_ordering()

        session_hormones_data = _load_session_hormones(hormones_df, hormones_ordering)

        inputs.append(_create_hormones_inputs(hormones_ordering))

        models.append(
            Key2HormonesModelInstantiator[data](
                session_hormones_data,
                registered_meshes,
                mesh_transform=affine_transform,
                colorized=colorized,
            )
        )

    if data == "multiple":
        plotter = MeshesPlotter(
            [MeshPlotter() for _ in range(len(registered_meshes))],
            overlay_plotter=overlay_plotter,
        )

        postproc_pred = ppdict.DictMap(ListSqueeze()) + ppdict.DictToValuesList()

        if hideable:
            checkbox_labels.extend(
                [
                    (index, key, True)
                    for index, key in enumerate(registered_meshes.keys())
                ]
            )
    else:
        if overlay:
            plotter = MeshesPlotter(
                plotters=[MeshPlotter()],
                overlay_plotter=overlay_plotter,
            )
        else:
            plotter = MeshPlotter()
            postproc_pred = ListSqueeze()

    mesh_explorer = MultiModelsMeshExplorer(
        models=models,
        inputs=inputs,
        graph=Graph(id_="mesh-plot", plotter=plotter),
        postproc_pred=postproc_pred,
        checkbox_labels=checkbox_labels,
    )

    return dbc.Container(mesh_explorer.to_dash())


def my_app(
    data="hipp",
    hideable=False,
    overlay=False,
    week=True,
    hormones=True,
    colorized=False,
):
    style = {
        "margin_side": "20px",
        "text_fontsize": "24px",
        "text_fontfamily": "Avenir",
        "title_fontsize": "40px",
        "space_between_sections": "70px",
        "space_between_title_and_content": "30px",
    }
    update_style(style)

    layout = _create_layout(
        data=data,
        hideable=hideable,
        overlay=overlay,
        week=week,
        hormones=hormones,
        colorized=colorized,
    )

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    app.layout = layout

    app.run(
        debug=True,
        use_reloader=False,
        host="0.0.0.0",
        port="8050",
    )
