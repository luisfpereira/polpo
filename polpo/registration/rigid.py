import numpy as np

from .base import BaseRegistration


def kabsch(source, target, as_homogeneous=True):
    """Estimate a rigid transformation between corresponding point clouds.

    Parameters
    ----------
    source : array-like, shape (n_points, 3)
        Source point cloud.
    target : array-like, shape (n_points, 3)
        Target point cloud with pointwise correspondences to `source`.
    as_homogeneous : bool
        If True, return the transformation as a homogeneous matrix.

    Returns
    -------
    transform : ndarray, shape (4, 4)
        Homogeneous rigid transformation mapping `source` to `target`.
        Returned when `as_homogeneous=True`.
    rotation : ndarray, shape (3, 3)
        Rotation matrix mapping `source` to `target`.
        Returned when `as_homogeneous=False`.
    translation : ndarray, shape (3,)
        Translation vector mapping `source` to `target`.
        Returned when `as_homogeneous=False`.
    """
    centroid_A = source.mean(axis=0)
    centroid_B = target.mean(axis=0)

    source_ = source - centroid_A
    target_ = target - centroid_B

    H = source_.T @ target_
    U, S, Vt = np.linalg.svd(H)
    rotation = Vt.T @ U.T

    # reflection fix
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    translation = centroid_B - rotation @ centroid_A

    if as_homogeneous:
        return np.vstack(
            [
                np.hstack([rotation, translation[:, None]]),
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    return rotation, translation


class RigidRegistration(BaseRegistration):
    """Rigid registration of corresponding point sets.

    The rigid transformation is estimated from corresponding source and
    target points using the Kabsch algorithm. It acts on a point ``x`` as

    .. math::

        x' = R x + t,

    where ``R`` is a rotation matrix and ``t`` is a translation vector.

    The estimated transformation is stored as a homogeneous matrix in
    ``transform_``.
    """

    def fit(self, source, target):
        """Estimate the rigid transformation from source to target.

        Parameters
        ----------
        source : array-like, shape (n_points, 3)
            Source points.
        target : array-like, shape (n_points, 3)
            Target points with pointwise correspondences to `source`.

        Returns
        -------
        self : RigidRegistration
            Fitted registration.
        """
        self.transform_ = kabsch(source, target)
        return self

    def transform(self, points):
        """Apply the estimated rigid transformation.

        Parameters
        ----------
        points : array-like, shape (n_points, 3)
            Points to transform.

        Returns
        -------
        transformed : ndarray, shape (n_points, 3)
            Transformed points.
        """
        rotation = self.transform_[:3, :3]
        translation = self.transform_[:3, 3]

        return points @ rotation.T + translation
