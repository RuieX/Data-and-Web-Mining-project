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


def ohe_fit(train_cat_data: pd.DataFrame) -> OneHotEncoder:
    """
    Returns a fitted One Hot Encoder based on the provided categorical data

    :param train_cat_data: dataframe containing the categorical features
    :return: fitted encoder
    """

    # - sparse=False means the method transform returns an array and not a sparse matrix
    #       e.g. if the categorical data is a dataframe, it returns an array of lists, where there is a list
    #       for each column in the dataframe
    # - "ignore" means it does not raise error if the encoder encounters an unknown category,
    #       but it instead sets all the value of the known categories to 0
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

    train_cat_data = _prepare_for_encoding(train_cat_data)
    ohe.fit(train_cat_data)

    return ohe


def ohe_transform(df: pd.DataFrame,
                  cat_features: list[str],
                  ohe: OneHotEncoder) -> pd.DataFrame:
    """
    Replaces the categorical features in the dataframe with
    their one-hot-encoded version.

    :param df: data to perform the operation on
    :param cat_features: list containing the names of the features to apply ohe to
    :param ohe: previously fitted One Hot Encoder.
        Useful when you want to use the same encoder for different dataframes
    :return: dataframe with encoded features
    """

    categorical_data = df[cat_features]
    categorical_data = _prepare_for_encoding(categorical_data)

    encoded_df = df.drop(columns=categorical_data.columns)
    encoded_cat_data = ohe.transform(categorical_data)
    for i, col in enumerate(ohe.get_feature_names_out()):
        encoded_df.loc[:, col] = encoded_cat_data[:, i]

    return encoded_df


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
