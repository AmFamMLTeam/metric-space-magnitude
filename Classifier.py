import numpy as np
from scipy.spatial import distance_matrix


def schur_comp(Z, B, C, D):
    return D - C.dot(Z).dot(B)


def cdf(pt, dist):
    return ((dist < pt).sum())/dist.shape[0]


def abs(pt, dist):
    return float(np.abs(pt))


class WeightClassifier():
    def __init__(
        self,
        wt_fn=cdf,
        magn_scale=None,
        class_ts=None
    ):
        self.wt_fn = wt_fn
        self.magn_scale = magn_scale
        self.class_ts = class_ts

    def _setup_classes(self, y):
        _classes = np.unique(y)
        _classes.sort()
        self._classes = _classes
        if self.class_ts is None:
            self.class_ts = np.ones(shape=self._classes.shape, dtype='float')
            err_msg = 'class_ts.shape does not match _classes.shape'
            assert self.class_ts.shape == self._classes.shape, err_msg

    def _setup_info(self, X, y):
        if self.magn_scale is None:
            self._info = {}
            for c in self._classes:
                d = {}
                d['X'] = X[y == c]
                class_index = np.argwhere(self._classes == c)[0][0]
                class_t = self.class_ts[class_index]
                dist_mtx = distance_matrix(d['X'], d['X'])
                if dist_mtx.shape[0] >= 1000:
                    inv_fn = np.linalg.pinv
                else:
                    inv_fn = np.linalg.inv
                try:
                    d['Z'] = inv_fn(np.exp(-class_t*dist_mtx))
                except Exception as e:
                    print(f'Exception {e} for class {c} t value {class_t}')
                    D = (
                        np.exp(-class_t*dist_mtx)
                        + 0.01 * np.identity(
                            n=dist_mtx.shape[0]
                        )  # perturb sim mtx to invert
                    )
                    Z = inv_fn(D)
                    d['Z'] = Z
                d['wts'] = d['Z'].sum(axis=1)
                d['t'] = class_t
                self._info[c] = d
        else:
            self._info = {}
            for c in self._classes:
                d = {}
                d['X'] = X[y == c]
                dist_mtx = distance_matrix(d['X'], d['X'])
                if dist_mtx.shape[0] >= 1000:
                    inv_fn = np.linalg.pinv
                else:
                    inv_fn = np.linalg.inv
                ts = np.linspace(0.1, 10., 30)
                Zs = []
                for t in ts:
                    try:
                        Z = inv_fn(np.exp(-t*dist_mtx))
                        Zs.append(Z)
                    except Exception as e:
                        print(f'Exception: {e} for t: {t} perturbing matrix')
                        D = (
                            np.exp(-t*dist_mtx)
                            +
                            0.01 * np.identity(
                                n=dist_mtx.shape[0]
                            )  # perturb similarity mtx to invert
                        )
                        Z = inv_fn(D)
                        Zs.append(Z)

                magnitudes = np.array([Z.sum() for Z in Zs])
                index = np.argmin(np.abs(magnitudes - self.magn_scale))
                t = ts[index]
                Zt = Zs[index]
                wts = Zt.sum(axis=1)

                d['ts'] = ts
                d['Zs'] = Zs
                d['magnitudes'] = magnitudes
                d['t'] = t
                d['Z'] = Zt
                d['wts'] = wts
                self._info[c] = d

    def fit(self, X, y):
        self._setup_classes(y)
        self._setup_info(X, y)

    def predict(self, new_points):
        res = []
        for cls in self._classes:
            X = self._info[cls]['X']
            Z = self._info[cls]['Z']
            wts = self._info[cls]['wts']
            t = self._info[cls]['t']
            Cs = np.exp(-t*distance_matrix(new_points, X))
            pred = []
            for c_i in Cs:
                C = c_i[np.newaxis]
                B = C.T
                schur = schur_comp(Z, B, C, 1).ravel()  # 1-dimensional
                wt = ((-1/(schur)).dot(C).dot(Z).sum() + (1/schur))
                pred.append(self.wt_fn(wt, wts))
            res.append(np.array(pred))
        preds = np.vstack(res)
        preds = np.argmin(preds, axis=0)
        pred_class = np.array([self._classes[_] for _ in preds])
        return pred_class

    def predict_proba(self, new_points):
        res = []
        for cls in self._classes:
            X = self._info[cls]['X']
            Z = self._info[cls]['Z']
            wts = self._info[cls]['wts']
            t = self._info[cls]['t']
            Cs = np.exp(-t*distance_matrix(new_points, X))
            pred = []
            for c_i in Cs:
                C = c_i[np.newaxis]
                B = C.T
                schur = schur_comp(Z, B, C, 1).ravel()  # 1-dimensional
                wt = ((-1/(schur)).dot(C).dot(Z).sum() + (1/schur))
                pred.append(self.wt_fn(wt, wts))
            res.append(np.array(pred))
        preds = np.vstack(res)
        return preds


class WeightClassifierCDF(WeightClassifier):
    def __init__(self, magn_scale=None, class_ts=None):
        super().__init__(wt_fn=cdf, magn_scale=magn_scale, class_ts=class_ts)


class WeightClassifierABS(WeightClassifier):
    def __init__(self, magn_scale=None, class_ts=None):
        super().__init__(wt_fn=abs, magn_scale=magn_scale, class_ts=class_ts)


class PowerClassifier(WeightClassifier):
    def __init__(self, wt_fn=cdf, tol=1e-3, t_max=10.):
        self.tol = tol
        self.t_max = t_max
        super().__init__(wt_fn)

    def _setup_info(self, X, y):
        self._info = {}
        for c in self._classes:
            d = {}
            d['X'] = X[y == c]
            dist_mtx = distance_matrix(d['X'], d['X'])
            # mn = dist_mtx[np.nonzero(dist_mtx)].min()
            # d['ts'] = np.linspace(self.tol,-1.5*np.log(self.tol)/mn,100)
            d['ts'] = np.linspace(self.tol, self.t_max, 20)
            if dist_mtx.shape[0] >= 10000:
                inv_fn = np.linalg.pinv
            else:
                inv_fn = np.linalg.inv
            d['Zs'] = [inv_fn(np.exp(-t*dist_mtx)) for t in d['ts']]
            wts = [
                np.exp(-t)*Z.sum(axis=1)
                for t, Z
                in zip(d['ts'], d['Zs'])
            ]
            wt_mtx = np.vstack(wts)
            d['powers'] = wt_mtx.sum(axis=0)*(d['ts'][1]-d['ts'][0])
            self._info[c] = d

    def predict(self, new_points):
        res = []
        for cls in self._classes:
            X = self._info[cls]['X']
            ts = self._info[cls]['ts']
            Zs = self._info[cls]['Zs']
            powers = self._info[cls]['powers']
            Cs = np.exp(-distance_matrix(new_points, X))
            pred = []
            for c_i in Cs:
                power = 0
                C = c_i[np.newaxis]
                B = C.T
                for Z, t in zip(Zs, ts):
                    schur = schur_comp(Z, B, C, 1).ravel()  # 1-dimensional
                    wt = ((-1/(schur)).dot(C).dot(Z).sum() + (1/schur))
                    power += (wt*np.exp(-t))*(ts[1]-ts[0])
                pred.append(self.wt_fn(power, powers))
            res.append(np.array(pred))
        preds = np.vstack(res)
        preds = np.argmin(preds, axis=0)
        pred_class = np.array([self._classes[_] for _ in preds])
        return pred_class


class PowerClassifierCDF(PowerClassifier):
    def __init__(self, tol=1e-3):
        super().__init__(cdf, tol)


class PowerClassifierABS(PowerClassifier):
    def __init__(self, tol=1e-3):
        super().__init__(abs, tol)
