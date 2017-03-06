import numpy as np
from sklearn.cross_validation import cross_val_predict as cvp


class MultiGrainedScaner():
    def __init__(self, base_estimator, params_list, sliding_ratio=0.25, k_fold=3):
        if k_fold > 1:  # use cv
            self.params_list = params_list
        else:  # use oob
            self.params_list = [params.update({'oob_score': True}) or params for params in params_list]
        self.sliding_ratio = sliding_ratio
        self.k_fold = k_fold
        self.base_estimator = base_estimator
        klass = self.base_estimator.__class__
        self.estimators = [klass(**params) for params in self.params_list]

    # generate scaned samples, X is not None, X[0] is no more than 3d
    def _sample_slicer(self, X, y):
        data_shape = X[0].shape
        window_shape = [max(int(data_size * self.sliding_ratio), 1) for data_size in data_shape]
        scan_round_axis = [data_shape[i] - window_shape[i] + 1 for i in range(len(data_shape))]
        scan_round_total = reduce(lambda acc, x: acc * x, scan_round_axis)
        if len(data_shape) == 1:
            newX = np.array([x[beg:beg + window_shape[0]]
                             for x in X
                             for beg in range(scan_round_axis[0])])
        elif len(data_shape) == 2:
            newX = np.array([x[beg0:beg0 + window_shape[0], beg1:beg1 + window_shape[1]].ravel()
                             for x in X
                             for beg0 in range(scan_round_axis[0])
                             for beg1 in range(scan_round_axis[1])])
        elif len(data_shape) == 3:
            newX = np.array(
                [x[beg0:beg0 + window_shape[0], beg1:beg1 + window_shape[1], beg2:beg2 + window_shape[2]].ravel()
                 for x in X
                 for beg0 in range(scan_round_axis[0])
                 for beg1 in range(scan_round_axis[1])
                 for beg2 in range(scan_round_axis[2])])
        newy = y.repeat(scan_round_total)
        return newX, newy, scan_round_total

    # generate new sample vectors
    def scan_fit(self, X, y):
        self.n_classes = len(np.unique(y))
        newX, newy, scan_round_total = self._sample_slicer(X, y)
        sample_vector_list = []
        for estimator in self.estimators:
            estimator.fit(newX, newy)
            if self.k_fold > 1:  # use cv
                predict_ = cvp(estimator, newX, newy, cv=self.k_fold, n_jobs=-1)
            else:  # use oob
                predict_ = estimator.oob_decision_function_
                # fill default value if meet nan
                inds = np.where(np.isnan(predict_))
                predict_[inds] = 1. / self.n_classes
            sample_vector = predict_.reshape((len(X), scan_round_total * self.n_classes))
            sample_vector_list.append(sample_vector)
        return np.hstack(sample_vector_list)

    def scan_predict(self, X):
        newX, newy, scan_round_total = self._sample_slicer(X, np.zeros(len(X)))
        sample_vector_list = []
        for estimator in self.estimators:
            predict_ = estimator.predict(newX)
            sample_vector = predict_.reshape((len(X), scan_round_total * self.n_classes))
            sample_vector_list.append(sample_vector)
        return np.hstack(sample_vector_list)