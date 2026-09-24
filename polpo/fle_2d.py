# inspired by https://github.com/nmarshallf/fle_2d
import math
import os

import finufft
import numpy as np
import scipy

from polpo.cache import LruCache

# TODO: add references


class BasisBuilder:
    def __init__(self, image_size, eps=1e-7):
        self.image_size = image_size
        self.eps = eps

        # to instantiate with defaults
        self.set_basis()

    def set_basis(self, spectrum_size=None, bandlimit=None, use_cached_roots=True):
        self.spectrum_size = spectrum_size
        self.bandlimit = bandlimit
        self.use_cached_roots = use_cached_roots

        return self

    def build(self):
        spectrum_size = (
            self.spectrum_size
            if self.spectrum_size is not None
            else int(self.image_size**2 * np.pi / 4)
        )
        basis = LaplacianDiskSpectrumFinder(
            spectrum_size,
            bandlimit=self.bandlimit,
            use_cached_roots=self.use_cached_roots,
        )()

        params = NumericsDefaults(self.image_size, basis, eps=self.eps)

        # TODO: allow for other operators?
        grid = PiConjSymCartesianizedPolarGrid(
            params.n_radial,
            params.n_angular,
            bounds=(basis.lmbd_min, basis.lmbd_max),
            h=params.h,
        )
        nufft = Nufft(grid, self.image_size, self.eps)

        step_2 = Step2(np.max(np.abs(basis.ns)), grid)

        upsampler = Dct()
        interp = SparseInterpolator(
            n_interp=2 * params.n_radial,
            numsparse=params.numsparse,
            basis=basis,
            grid=grid,
        )

        step_3 = Step3(interp, upsampler=upsampler)

        steps = [nufft, step_2, step_3]

        fo = FastOperator(self.image_size, basis, steps, max_iter=params.max_iter)

        return basis.set_operator(fo)


class DiskBasis:
    # TODO: map single to double script with dict?

    def __init__(self, psi, ns, ks, lmbds, cs):
        # TODO: rename arguments

        # ns, ks are index lists
        # lmbds, cs are flatten

        # TODO: psi should be defined here?
        self.psi = psi
        self.ns = ns
        self.ks = ks
        self.lmbds = lmbds
        self.cs = cs

        self.n_elems = len(self.lmbds)

        ns_to_indices = {}
        for index, n in enumerate(ns):
            indices = ns_to_indices.get(n, [])
            indices.append(index)
            ns_to_indices[n] = indices

        # given an n, returns all the positions where it appears
        self.ns_to_indices = ns_to_indices

        # TODO: required?
        self.n_ns = len(self.ns_to_indices)

        # TODO: remove?
        ndx = 2 * np.abs(ns) - (ns < 0)
        self.ndmax = np.max(ndx)

        # TODO: remove
        # temporary for compatibility with fle_2d
        self.ns_to_fle = {n: 2 * abs(n) - int(n < 0) for n in self.ns_to_indices}

        # TODO: remove?
        self._c2r = precomp_transform_complex_to_real(ns)
        self._r2c = scipy.sparse.csr_matrix(self._c2r.transpose().conj())

        self._operator = None

    def set_operator(self, operator):
        self._operator = operator
        return self

    @property
    def lmbd_min(self):
        return self.lmbds[0]

    @property
    def lmbd_max(self):
        return self.lmbds[-1]

    def r2c(self, a):
        # TODO: improve docstrings
        return matvecmul(self._r2c, a)

    def c2r(self, a):
        # TODO: improve docstrings
        return matvecmul(self._c2r, a)

    def expand(self, image):
        return self._operator(image)

    def evaluate(self, coeffs):
        return self._operator.inverse(coeffs)


class LaplacianDiskSpectrumFinder:
    # follows geomfum
    def __init__(self, spectrum_size, bandlimit=None, use_cached_roots=True):
        self.spectrum_size = spectrum_size
        self.bandlimit = bandlimit

        self._roots_table = self._load_roots() if use_cached_roots else None

    def _load_roots(self):
        path_to_module = os.path.dirname(__file__)
        zeros_path = os.path.join(path_to_module, "jn_zeros_n=3000_nt=2500.mat")
        data = scipy.io.loadmat(zeros_path)
        roots_table = data["roots_table"]
        return roots_table

    def _jn_zeros(self, n, n_t):
        if self._roots_table is not None:
            return self._roots_table[n, :n_t]

        return scipy.special.jn_zeros(n, n_t)

    def __call__(self):
        ne = self.spectrum_size
        bandlimit = self.bandlimit

        # number of roots to check
        nc = int(3 * np.sqrt(ne))
        nd = int(2 * np.sqrt(ne))

        # preallocate
        nn = 1 + 2 * nc
        ns = np.zeros((nn, nd), dtype=int, order="F")
        ks = np.zeros((nn, nd), dtype=int, order="F")
        lmds = np.ones((nn, nd), dtype=np.float64) * np.inf

        ns[0, :] = 0
        lmds[0, :] = self._jn_zeros(0, nd)
        ks[0, :] = np.arange(nd) + 1

        # add roots of J_n for n > 0 twice with +k and -k
        # the square of the roots are eigenvalues of the Laplacian (with
        # Dirichlet boundary conditions
        # see {eq:eigenfun}
        for i in range(nc):
            n = i + 1
            ns[2 * n - 1, :] = -n
            ks[2 * n - 1, :] = np.arange(nd) + 1

            lmds[2 * n - 1, :nd] = self._jn_zeros(n, nd)

            ns[2 * n, :] = n
            ks[2 * n, :] = ks[2 * n - 1, :]
            lmds[2 * n, :] = lmds[2 * n - 1, :]

        # flatten
        ns = ns.flatten()
        ks = ks.flatten()
        lmds = lmds.flatten()

        # sort by lmds
        idx = np.argsort(lmds)
        ns = ns[idx]
        ks = ks[idx]
        lmds = lmds[idx]

        # sort complex conjugate pairs: -n first, +n second
        idx = np.arange(ne + 1)
        for i in range(ne + 1):
            if ns[i] >= 0:
                continue
            if np.abs(lmds[i] - lmds[i + 1]) < 1e-14:
                continue
            idx[i - 1] = i
            idx[i] = i - 1

        ns = ns[idx]
        ks = ks[idx]
        lmds = lmds[idx]

        # {sec:bandlimit}
        if bandlimit:
            for i in range(len(lmds)):
                if lmds[ne] / (np.pi) >= (bandlimit - 1) // 2:
                    ne = ne - 1

        # potentially subtract 1 from ne to keep complex conj pairs
        if ns[ne - 1] < 0:
            ne = ne - 1

        # make sure that ne is always at least 1
        if ne <= 1:
            ne = 1

        # take top ne values
        ns = ns[:ne]
        ks = ks[:ne]
        lmds = lmds[:ne]

        cs = np.zeros(ne)

        # TODO: can exploit scipy.special.jv
        psi = [None] * ne
        for i in range(ne):
            n = ns[i]
            lmd = lmds[i]
            # see {eq:eigenfun_const}
            c = 1 / np.sqrt(np.pi * scipy.special.jv(ns[i] + 1, lmds[i]) ** 2)
            if ns[i] == 0:
                psi[i] = (
                    lambda r, t, n=n, c=c, lmd=lmd: c
                    * scipy.special.jv(n, lmd * r)
                    * (r <= 1)
                )
            else:
                # see {eq:eigenfun} and {eq:eigenfun_extend}
                psi[i] = (
                    lambda r, t, c=c, n=n, lmd=lmd: c
                    * scipy.special.jv(n, lmd * r)
                    * np.exp(1j * n * t)
                    * (r <= 1)
                )
            cs[i] = c

        return DiskBasis(psi, ns, ks, lmds, cs)


class DenseOperator:
    def __init__(self, mat):
        self.mat = mat

        image_size = int(math.sqrt(mat.shape[0]))
        self._image_shape = (image_size, image_size)

    @classmethod
    def from_basis(cls, L, basis, complexmode=True):
        return cls(create_dense_operator(L, basis, complexmode=complexmode))

    def __call__(self, image):
        return matvecmul(self.mat.T, image.reshape(image.shape[:-2] + (-1,)))

    def inverse(self, coeffs):
        return matvecmul(self.mat, coeffs).reshape(
            coeffs.shape[:-1] + self._image_shape
        )


class FastOperator:
    def __init__(self, image_size, basis, steps, max_iter=0):
        self.basis = basis
        self._image_shape = (image_size, image_size)
        self.max_iter = max_iter

        self.steps = steps

        self._image_cropper = ImageCropper(image_size)

    def _call_t(self, image):
        out = image
        for step in self.steps:
            out = step(out)

        return out

    def __call__(self, image):
        image = self._image_cropper(image.copy())

        b = self._call_t(image)

        # TODO: add other stopping criteria?
        a0 = b
        for i in range(self.max_iter):
            a0 = a0 - self._call_t(self.inverse(a0)) + b
        return a0

    def inverse(self, coeffs):
        out = coeffs
        for step in reversed(self.steps):
            out = step.inverse(out)

        return out


class PiConjSymCartesianizedPolarGrid:
    def __init__(self, n_radial, n_angular, bounds, h):
        # TODO: find a good name for h and bounds

        self.n_radial = n_radial
        self.n_angular = n_angular
        self.bounds = bounds
        self.h = h

        self.xs, self.ys = self._create_grid(full=False)
        self._full_coords = None

    @property
    def coords(self):
        return np.stack([self.xs, self.ys], axis=-1)

    @property
    def full_coords(self):
        if self._full_coords is None:
            self._full_coords = np.stack(self._create_grid(full=True), axis=-1)

        return self._full_coords

    def _create_grid(self, full=False):
        xs = 1 - (2 * np.arange(self.n_radial) + 1) / (2 * self.n_radial)
        xs = np.cos(np.pi * xs)
        pts = (xs + 1) / 2
        pts = (self.bounds[1] - self.bounds[0]) * pts + self.bounds[0]
        pts = pts.reshape(-1, 1)

        n_angular_ = self.n_angular if full else self.n_angular // 2
        phi = 2 * np.pi * np.arange(n_angular_) / self.n_angular
        x = np.cos(phi)
        x = x.reshape(1, -1)
        y = np.sin(phi)
        y = y.reshape(1, -1)
        x = x * pts * self.h
        y = y * pts * self.h

        x = x.flatten()
        y = y.flatten()

        return x, y

    def to_full(self, f):
        # [..., partial_dim]
        # TODO: handle vectorization
        # TODO: improve notation
        batch_shape = f.shape[:-1]

        z = np.zeros(batch_shape + (self.n_radial, self.n_angular), dtype=f.dtype)

        z0 = f.reshape(batch_shape + (self.n_radial, self.n_angular // 2))

        z[..., :, : self.n_angular // 2] = z0
        z[..., :, self.n_angular // 2 :] = np.conj(z0)
        return z.reshape(batch_shape + (self.n_angular * self.n_radial,))

    def to_partial(self, f):
        # [..., dim]
        f = self.unflatten_scalar(f)[..., :, : self.n_angular // 2]
        return self.flatten_scalar(f)

    def unflatten_scalar(self, scalar):
        # -1 allows for full and partial (n_angular)
        return scalar.reshape(scalar.shape[:-1] + (self.n_radial, -1))

    def flatten_scalar(self, scalar):
        # -1 allows for full and partial (n_radial*n_angular)
        return scalar.reshape(scalar.shape[:-2] + (-1,))


class NumericsDefaults:
    def __init__(self, image_size, basis, eps=1e-7, n_radial=None, n_angular=None):
        self.image_size = image_size
        self.eps = eps
        # TODO: allow to pass parameters and override?

        # Heuristics for choosing numsparse and maxitr
        max_iter = 1 + int(3 * np.log2(image_size))
        numsparse = 32
        if eps >= 1e-10:
            numsparse = 22
            max_iter = 1 + int(2 * np.log2(image_size))
        if eps >= 1e-7:
            numsparse = 16
            max_iter = 1 + int(np.log2(image_size))
        if eps >= 1e-4:
            numsparse = 8
            max_iter = 1 + int(np.log2(image_size)) // 2

        self.max_iter = max_iter
        self.numsparse = numsparse

        self.n_radial = self.default_n_radial() if n_radial is None else n_radial
        self.n_angular = (
            self.default_n_angular(basis) if n_angular is None else n_angular
        )

        # TODO: n_interp and numsparse are particular?
        # TODO: currently has to be 2*n_radial due to Dct (easy fix)
        self.n_interp = 2 * self.n_radial

        self.h = 1 / (self.image_size // 2)

    def default_n_radial(self):
        L = self.image_size
        eps = self.eps

        Q = int(np.ceil(2.4 * L))
        n_radial = Q
        tmp = 1 / (np.sqrt(np.pi))
        for q in range(1, Q + 1):
            tmp = tmp / q * (np.sqrt(np.pi) * L / 4)
            if tmp <= eps:
                n_radial = int(max(q, np.log2(1 / eps)))
                break
        n_radial = max(n_radial, int(np.ceil(np.log2(1 / eps))))

        return n_radial

    def default_n_angular(self, basis):
        L = self.image_size
        eps = self.eps
        lmbd_max = basis.lmbd_max
        ndmax = basis.ndmax

        S = int(max(7.08 * L, -np.log2(eps) + 2 * np.log2(L)))
        n_angular = S
        for svar in range(int(lmbd_max + ndmax) + 1, S + 1):
            tmp = L**2 * ((lmbd_max + ndmax) / svar) ** svar
            if tmp <= eps:
                n_angular = int(max(int(svar), np.log2(1 / eps)))
                break

        if n_angular % 2 == 1:
            n_angular += 1

        return n_angular


class ImageCropper:
    def __init__(self, image_size, tol=1e-13):
        R = image_size // 2
        x = np.arange(-R, R)
        y = np.arange(-R, R)
        xs, ys = np.meshgrid(x, y)
        xs = xs / R
        ys = ys / R
        rs = np.sqrt(xs**2 + ys**2)

        self.zeros_idx = rs > 1 + 1e-13

    def __call__(self, image):
        # NB: acts in place
        image[..., self.zeros_idx] = 0.0
        return image


class Nufft:
    # TODO: better naming
    # TODO: try other libraries

    # TODO: test with vectorization cache (maybe setpts is slow)

    def __init__(self, grid, image_size, eps=1e-7, cache_size=10):
        self.grid = grid
        self._image_shape = (image_size, image_size)

        def make_plan(n_trans, nufft_type, isign):
            if isinstance(n_trans, tuple):
                n_trans = 1 if len(n_trans) == 0 else n_trans[0]

            plan = finufft.Plan(
                nufft_type=nufft_type,
                n_modes_or_dim=self._image_shape,
                n_trans=n_trans,
                isign=isign,
                eps=eps,
            )
            # NUFFT has opposite meshgrid ordering
            plan.setpts(grid.ys, grid.xs)
            return plan

        self._plan = LruCache(
            cache_size, lambda n_trans: make_plan(n_trans, nufft_type=2, isign=-1)
        )

        self._plan_inv = LruCache(
            cache_size, lambda n_trans: make_plan(n_trans, nufft_type=1, isign=1)
        )

        self._image_cropper = ImageCropper(image_size)

    def __call__(self, f, full=True):
        # required by plan
        f = f.astype(np.complex128)
        z0 = self._plan.get(f.shape[:-2]).execute(f) * self.grid.h**2

        if full:
            return self.grid.to_full(z0)

        return z0

    def inverse(self, z):
        # TODO: check if already partial?
        # assumes full
        batch_shape = z.shape[:-1]
        z = self.grid.to_partial(z)

        f = self._plan_inv.get(batch_shape).execute(z)
        f = f + np.conj(f)
        f = np.real(f)
        f = f.reshape(batch_shape + self._image_shape)

        return self._image_cropper(f)


class Step2:
    def __init__(self, n_max, grid):
        # TODO: is grid needed? maybe reshape before?
        self.grid = grid

        nus = np.zeros(1 + 2 * n_max, dtype=int)
        nus[0] = 0
        for i in range(1, n_max + 1):
            nus[2 * i - 1] = -i
            nus[2 * i] = i

        # [0, -1, 1, -2, 2, ...]
        self.nus = nus
        self._pow_nus = np.pow((1j), nus)
        self._ipow_nus = np.pow((-1j), nus)

        self.c2r_nus = precomp_transform_complex_to_real(nus)
        self.r2c_nus = scipy.sparse.csr_matrix(self.c2r_nus.transpose().conj())

    def __call__(self, z):
        # TODO: improve naming; improve dim info

        # [..., dim]
        z = self.grid.unflatten_scalar(z)
        n_angular = z.shape[-1]

        # TODO: can use a truncated z?
        b = np.fft.fft(z) / n_angular
        b = b[..., :, self.nus] * self._pow_nus
        b = matvecmul(self.c2r_nus, b)

        # [..., dim_a, dim_b]
        return np.real(b)

    def inverse(self, b):
        # TODO: handle vectorization (easy)

        tmp = np.zeros(b.shape[:-1] + (self.grid.n_angular,), dtype=np.complex128)

        b = matvecmul(self.r2c_nus, b)

        tmp[..., :, self.nus] = b * self._ipow_nus

        z = np.fft.ifft(tmp, axis=-1)

        return self.grid.flatten_scalar(z)


class IdentityTransform:
    def __call__(self, a):
        return a

    def inverse(self, a):
        return a


class Dct:
    def __call__(self, b):
        # NB: this ratio is hardcoded in fle_2d
        # TODO: controls bz; also impacts b access in SparseInterpolator
        ratio = 2

        n_interp = ratio * b.shape[-2]

        b = scipy.fft.dct(b, axis=-2, type=2) / n_interp
        bz = np.zeros_like(b)
        b = np.concatenate((b, bz), axis=-2)
        # TODO: is is proper to multiply by 2?
        b = scipy.fft.idct(b, axis=-2, type=2) * (2 * n_interp)

        return b

    def inverse(self, b):
        b = scipy.fft.dct(b, axis=-2, type=2)
        # TODO: fix when adapting ratio
        b = b[..., : b.shape[-2] // 2, :]
        b = scipy.fft.idct(b, axis=-2, type=2)
        return b


class SparseInterpolator:
    def __init__(self, n_interp, numsparse, basis, grid):
        self.basis = basis
        self.grid = grid

        # TODO: currently has to be 2*n_radial due to Dct
        self.n_interp = n_interp
        self.numsparse = numsparse

        self.A3, self.A3_T = self._build_matrices(basis)

    def _build_matrices(self, basis):
        basis = self.basis

        # Source points
        xs = 1 - (2 * np.arange(self.n_interp) + 1) / (2 * self.n_interp)
        xs = np.cos(np.pi * xs)

        lbmd_diff = basis.lmbd_max - basis.lmbd_min

        A3 = {}
        A3_T = {}
        for n, indices in basis.ns_to_indices.items():
            # Target points
            # NB: this is where lambdas are selected
            lmbds_n = basis.lmbds[indices]

            x = 2 * (lmbds_n - basis.lmbd_min) / lbmd_diff - 1
            vals, x_ind, xs_ind = np.intersect1d(x, xs, return_indices=True)
            x[x_ind] = x[x_ind] + 2e-16

            A3[n], A3_T[n] = barycentric_interp_sparse(x, xs, self.numsparse)

        return A3, A3_T

    def __call__(self, b):
        a = np.zeros(b.shape[:-2] + (self.basis.n_elems,), dtype=np.float64)
        for n, indices in self.basis.ns_to_indices.items():
            a[..., indices] = matvecmul(self.A3[n], b[..., self.basis.ns_to_fle[n]])

        a = a * self.basis.cs / self.grid.h

        return self.basis.r2c(a)

    def inverse(self, a):
        a = self.basis.c2r(a)

        # TODO: confirm it is not a mistake, i.e. divide by cs?
        a = a * self.grid.h * self.basis.cs

        b = np.zeros(
            a.shape[:-1] + (self.n_interp, self.basis.ndmax + 1),
            dtype=np.float64,
            order="F",
        )
        for n, indices in self.basis.ns_to_indices.items():
            b[..., self.basis.ns_to_fle[n]] = matvecmul(self.A3_T[n], a[..., indices])

        return b


class Step3:
    def __init__(self, interp, upsampler=True):
        self.interp = interp

        if upsampler is None:
            upsampler = IdentityTransform()
        elif upsampler is True:
            upsampler = Dct()

        # NB: must be compatible with interp
        self.upsampler = upsampler

    def __call__(self, b):
        b = self.upsampler(b)
        return self.interp(b)

    def inverse(self, a):
        b = self.interp.inverse(a)
        return self.upsampler.inverse(b)


def matvecmul(mat, vec):
    """Matrix vector multiplication.

    Parameters
    ----------
    mat : array-like, shape=[..., m, n]
        Matrix.
    vec : array-like, shape=[..., n]
        Vector.

    Returns
    -------
    matvec : array-like, shape=[..., m]
        Matrix vector multiplication.
    """
    # TODO: in geomfum
    # TODO: move to proper place: geomstats?
    if vec.ndim == 1:
        return mat @ vec

    if mat.ndim == 2:
        reshape_out = False
        if vec.ndim > 2:  # to handle sparse matrices
            reshape_out = True
            batch_shape = vec.shape[:-1]
            vec = vec.reshape(-1, vec.shape[-1])

        out = (mat @ vec.T).T
        if reshape_out:
            return out.reshape(batch_shape + mat.shape[:1])

        return out

    return np.einsum("...ij,...j->...i", mat, vec)


def barycentric_interp_sparse(x, xs, s):
    # https://people.maths.ox.ac.uk/trefethen/barycentric.pdf

    # TODO: kind of builds the stencil?

    n = len(x)
    m = len(xs)

    # Modify points by 2e-16 to avoid division by zero
    vals, x_ind, xs_ind = np.intersect1d(x, xs, return_indices=True, assume_unique=True)
    x[x_ind] = x[x_ind] + 2e-16

    idx = np.zeros((n, s))
    jdx = np.zeros((n, s))
    vals = np.zeros((n, s))
    xss = np.zeros((n, s))
    denom = np.zeros((n, 1))
    temp = np.zeros((n, 1))
    ws = np.zeros((n, s))
    xdiff = np.zeros(n)
    for i in range(n):
        # get a kind of balanced interval around our point
        k = np.searchsorted(x[i] < xs, True)

        idp = np.arange(k - s // 2, k + (s + 1) // 2)
        if idp[0] < 0:
            idp = np.arange(s)
        if idp[-1] >= m:
            idp = np.arange(m - s, m)
        xss[i, :] = xs[idp]
        jdx[i, :] = idp
        idx[i, :] = i

    x = x.reshape(-1, 1)
    Iw = np.ones(s, dtype=bool)
    ew = np.zeros((n, 1))
    xtw = np.zeros((n, s - 1))

    Iw[0] = False
    const = np.zeros((n, 1))
    for j in range(s):
        ew = np.sum(-np.log(np.abs(xss[:, 0].reshape(-1, 1) - xss[:, Iw])), axis=1)
        constw = np.exp(ew / s)
        constw = constw.reshape(-1, 1)
        const += constw
    const = const / s

    for j in range(s):
        Iw[j] = False
        xtw = const * (xss[:, j].reshape(-1, 1) - xss[:, Iw])
        ws[:, j] = 1 / np.prod(xtw, axis=1)
        Iw[j] = True

    xdiff = xdiff.flatten()
    x = x.flatten()
    temp = temp.flatten()
    denom = denom.flatten()
    for j in range(s):
        xdiff = x - xss[:, j]
        temp = ws[:, j] / xdiff
        vals[:, j] = vals[:, j] + temp
        denom = denom + temp
    vals = vals / denom.reshape(-1, 1)

    vals = vals.flatten()
    idx = idx.flatten()
    jdx = jdx.flatten()
    A = scipy.sparse.csr_matrix((vals, (idx, jdx)), shape=(n, m), dtype=np.float64)
    A_T = scipy.sparse.csr_matrix((vals, (jdx, idx)), shape=(m, n), dtype=np.float64)

    return A, A_T


def precomp_transform_complex_to_real(ns):
    ne = len(ns)
    nnz = np.sum(ns == 0) + 2 * np.sum(ns != 0)
    idx = np.zeros(nnz, dtype=int)
    jdx = np.zeros(nnz, dtype=int)
    vals = np.zeros(nnz, dtype=np.complex128)

    k = 0
    for i in range(ne):
        n = ns[i]
        if n == 0:
            vals[k] = 1
            idx[k] = i
            jdx[k] = i
            k = k + 1
        if n < 0:
            s = (-1) ** np.abs(n)

            vals[k] = s / (1j * np.sqrt(2))
            idx[k] = i
            jdx[k] = i
            k = k + 1

            vals[k] = -1 / (1j * np.sqrt(2))
            idx[k] = i
            jdx[k] = i + 1
            k = k + 1

            vals[k] = s / np.sqrt(2)
            idx[k] = i + 1
            jdx[k] = i
            k = k + 1

            vals[k] = 1 / np.sqrt(2)
            idx[k] = i + 1
            jdx[k] = i + 1
            k = k + 1

    A = scipy.sparse.csr_matrix((vals, (idx, jdx)), shape=(ne, ne), dtype=np.complex128)

    return A


def create_dense_operator(L, basis, complexmode=False):
    # TODO: recover parallel behavior?
    # TODO: considers L to be even

    # aka create_denseB
    # see {eq:operator_B} and {eq:operator_B^*}

    # Evaluate eigenfunctions
    R = L // 2
    h = 1 / R
    x = np.arange(-R, R)
    y = np.arange(-R, R)
    xs, ys = np.meshgrid(x, y)
    xs = xs / R
    ys = ys / R
    rs = np.sqrt(xs**2 + ys**2)
    ts = np.arctan2(ys, xs)

    B = np.zeros((L, L, basis.n_elems), dtype=np.complex128, order="F")
    for i in range(basis.n_elems):
        B[:, :, i] = basis.psi[i](rs, ts)
    B = h * B

    B = B.reshape(L**2, basis.n_elems)

    if not complexmode:
        B = transform_complex_to_real(B, basis.ns)

    return B.reshape(L**2, basis.n_elems)


def transform_complex_to_real(Z, ns):
    ne = Z.shape[1]
    X = np.zeros(Z.shape, dtype=np.float64)

    for i in range(ne):
        n = ns[i]
        if n == 0:
            X[:, i] = np.real(Z[:, i])
        if n < 0:
            s = (-1) ** np.abs(n)
            x0 = (-s * Z[:, i] + Z[:, i + 1]) / (1j * np.sqrt(2))
            x1 = (s * Z[:, i] + Z[:, i + 1]) / np.sqrt(2)
            X[:, i] = np.real(x0)
            X[:, i + 1] = np.real(x1)

    return X
