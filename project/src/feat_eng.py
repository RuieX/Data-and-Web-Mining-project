import numpy as np
import pandas as pd
# import project.src.utils as utils
# import project.src.training as tr
from operator import itemgetter
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from category_encoders import LeaveOneOutEncoder
from sklearn.feature_selection import RFECV, RFE


class TrainTestSplit(object):
    def __init__(self, x_train: pd.DataFrame, x_test: pd.DataFrame,
                 y_train: np.ndarray, y_test: np.ndarray):
        self.x_train: pd.DataFrame = x_train
        self.x_test: pd.DataFrame = x_test
        self.y_train: np.ndarray = y_train
        self.y_test: np.ndarray = y_test


def get_features_from_name(df: pd.DataFrame, identifiers: list[str]) -> pd.DataFrame:
    """
    Selects features based on their names.
    If they contain at least one of the strings provided, then the feature
    is selected, exception being if its name is contained in the exclusion list.

    :param df: dataframe to extract the categorical features from
    :param identifiers: strings contained in the name of a feature.
        If a feature name contains at least one of these, then the feature is selected.
    :return: dataframe containing the features
    """

    features = pd.DataFrame()
    for col in df:
        if any(id_ in col for id_ in identifiers):
            features[col] = df[col]

    return features


def _prepare_for_encoding(features: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all categorical features to string because:
    1. encoders can handle only either str or numerical values
    2. this way we make sure that the encoders correctly detect
        the categorical features to encode them

    :param features: dataframe containing the features to prepare for encoding
    :return: features ready to be encoded
    """
    for col in features:
        features.loc[:, col] = features[col].astype(str)

    return features
