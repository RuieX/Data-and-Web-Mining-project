import os
import numpy as np
import pandas as pd
from math import sqrt
# import project.src.utils as utils
# import project.src.training as tr
from operator import itemgetter
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from category_encoders import LeaveOneOutEncoder
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Evaluation(object):
    def __init__(self, y_real: np.ndarray, y_pred: np.ndarray):
        self.y_pred = y_pred
        self.y_real = y_real

        self.mae = mean_absolute_error(y_real, y_pred)
        self.mse = mean_squared_error(y_real, y_pred)
        self.rmse = sqrt(mean_squared_error(y_real, y_pred))

    def print_eval(self):
        print("--------------Model Evaluations:--------------")
        print('Mean Absolute Error : {}'.format(self.mae))
        print('Mean Squared Error : {}'.format(self.mse))
        print('Root Mean Squared Error : {}'.format(self.rmse))
        print()


class EvaluatedModel(object):
    def __init__(self, model, train_eval: Evaluation, test_eval: Evaluation):
        self.model = model
        self.train_eval = train_eval
        self.test_eval = test_eval
