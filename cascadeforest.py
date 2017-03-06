import numpy as np
from sklearn.cross_validation import cross_val_predict as cvp

class CascadeForest():
    def __init__(self, base_estimator, params_list, k_fold=3, evaluate=lambda pre, y: float(sum(pre == y)) / len(y)):
        if k_fold > 1:  # use cv
            self.params_list = params_list
        else:  # use oob
            self.params_list = [params.update({'oob_score': True}) or params for params in params_list]
        self.k_fold = k_fold
        self.evaluate = evaluate
        self.base_estimator = base_estimator

    #         base_class = base_estimator.__class__
    #         global prob_class
    #         class prob_class(base_class): #to use cross_val_predict, estimator's predict method should be predict_prob
    #             def predict(self, X):
    #                 return base_class.predict_proba(self, X)
    #         self.base_estimator = prob_class()

    def fit(self, X_train, y_train):
        self.n_classes = len(np.unique(y_train))
        self.estimators_levels = []
        klass = self.base_estimator.__class__
        predictions_levels = []
        self.classes = np.unique(y_train)

        # first level
        estimators = [klass(**params) for params in self.params_list]
        self.estimators_levels.append(estimators)
        predictions = []
        for estimator in estimators:
            estimator.fit(X_train, y_train)
            if self.k_fold > 1:  # use cv
                predict_ = cvp(estimator, X_train, y_train, cv=self.k_fold, n_jobs=-1)
            else:  # use oob
                predict_ = estimator.oob_decision_function_
                # fill default value if meet nan
                inds = np.where(np.isnan(predict_))
                predict_[inds] = 1. / self.n_classes
            predictions.append(predict_)
        attr_to_next_level = np.hstack(predictions)
        y_pre = self.classes.take(np.argmax(np.array(predictions).mean(axis=0), axis=1), axis=0)
        self.max_accuracy = self.evaluate(y_pre, y_train)

        # cascade step
        while True:
            print
            'level {}, CV accuracy: {}'.format(len(self.estimators_levels), self.max_accuracy)
            estimators = [klass(**params) for params in self.params_list]
            self.estimators_levels.append(estimators)
            predictions = []
            X_train_step = np.hstack((attr_to_next_level, X_train))
            for estimator in estimators:
                estimator.fit(X_train_step, y_train)
                if self.k_fold > 1:  # use cv
                    predict_ = cvp(estimator, X_train_step, y_train, cv=self.k_fold, n_jobs=-1)
                else:  # use oob
                    predict_ = estimator.oob_decision_function_
                    # fill default value if meet nan
                    inds = np.where(np.isnan(predict_))
                    predict_[inds] = 1. / self.n_classes
                predictions.append(predict_)
            attr_to_next_level = np.hstack(predictions)
            y_pre = self.classes.take(np.argmax(np.array(predictions).mean(axis=0), axis=1), axis=0)
            accuracy = self.evaluate(y_pre, y_train)
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
            else:
                self.estimators_levels.pop()
                break

    def predict_proba(self, X):
        # first level
        estimators = self.estimators_levels[0]
        predictions = []
        for estimator in estimators:
            predict_ = estimator.predict(X)
            predictions.append(predict_)
        attr_to_next_level = np.hstack(predictions)

        # cascade step
        for i in range(1, len(self.estimators_levels)):
            estimators = self.estimators_levels[i]
            predictions = []
            X_step = np.hstack((attr_to_next_level, X))
            for estimator in estimators:
                predict_ = estimator.predict(X_step)
                predictions.append(predict_)
            attr_to_next_level = np.hstack(predictions)

        # final step, calc prob
        prediction_final = np.array(predictions).mean(axis=0)  # 不同estimator求平均
        return prediction_final

    def predict(self, X):
        proba = self.predict_proba(X)
        predictions = self.classes.take(np.argmax(proba, axis=1), axis=0)  # 平均值最大的index对应的class
        return predictions