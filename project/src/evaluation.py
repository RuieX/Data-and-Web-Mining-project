import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Evaluation:
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


class EvaluatedModel:
    def __init__(self, model, train_eval: Evaluation, test_eval: Evaluation):
        self.model = model
        self.train_eval = train_eval
        self.test_eval = test_eval
