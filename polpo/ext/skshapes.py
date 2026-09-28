import skshapes as sks


class SksFromPv:
    def __call__(self, poly_data):
        return sks.PolyData(poly_data)


class PvFromSks:
    def __call__(self, poly_data):
        return poly_data.to_pyvista()


class SksRigidRegistration:
    def __init__(self, loss=None, n_iter=2, verbose=False):
        if loss is None:
            loss = sks.NearestNeighborsLoss()

        self.registration = sks.Registration(
            model=sks.RigidMotion(),
            loss=loss,
            n_iter=n_iter,
            verbose=verbose,
        )

    def __call__(self, data):
        source, target = data
        self.registration.fit(source=source, target=target)
        return self.registration.transform(source=source)
