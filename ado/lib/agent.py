
from sklearn.metrics import roc_curve, auc
from sklearn import linear_model
import numpy as np


class Agent:
    def __init__(self, df_train, df_test, costs, bootstrapping=True):
        self.x_train = df_train.drop(columns="Activity")
        self.y_train = df_train.Activity

        self.x_test = df_test.drop(columns="Activity")
        self.y_test = df_test.Activity

        self.y_hat_train = []
        self.y_hat_test = []
        self.costs = costs
        self.bootstrapping = bootstrapping
        self.model = None
        self.best_cost = None

    def leave_one_all(self, cost):
        all_index = self.x_train.index.unique() if self.bootstrapping else self.x_train.index
        for idx in all_index:
            self.fit(idx, cost)
            y_hat = self.model.predict_proba(self.x_train.ix[idx, ].values.reshape(-1, self.x_train.shape[1]))[:, 1]
            y_hat = y_hat.flatten()
            self.y_hat_train.append(y_hat)
        self.y_hat_train = np.concatenate(self.y_hat_train)
        auc_roc = self.calculate_auc_roc(self.y_train, self.y_hat_train)
        return auc_roc

    def fit(self, idx, cost):
        model = linear_model.LogisticRegression(penalty='l1', C=cost)
        model.fit(self.x_train.drop(idx, axis=0), self.y_train.drop(idx, axis=0))
        self.model = model

    @staticmethod
    def calculate_auc_roc(truth, pred):
        fpr, tpr, _ = roc_curve(truth, pred)
        auc_roc = auc(fpr, tpr)
        return auc_roc

    def optimize_cost(self):
        auc_rocs = []
        for cost in self.costs:
            auc_roc = self.leave_one_all(cost)
            auc_rocs.append(auc_roc)
            self.y_hat_train = []
        best_idx = auc_rocs.index(max(auc_rocs))
        self.best_cost = self.costs[best_idx]
        return auc_rocs

    def predict_test(self):
        model = linear_model.LogisticRegression(penalty='l1', C=15)
        model.fit(self.x_train, self.y_train)
        self.model = model
        self.y_hat_test = model.predict_proba(self.x_test)[:, 1]

    def run(self):
        self.optimize_cost()
        self.predict_test()
        self.leave_one_all(self.best_cost)
