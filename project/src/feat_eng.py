import os
import numpy as np
import pandas as pd
# import project.src.utils as utils
# import project.src.training as tr
from operator import itemgetter
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from category_encoders import LeaveOneOutEncoder
from sklearn.feature_selection import RFECV, RFE


class TrainTestSplit():
    def __init__(self, x_train: pd.DataFrame, x_test: pd.DataFrame,
                 y_train: np.ndarray, y_test: np.ndarray):
        self.x_train: pd.DataFrame = x_train
        self.x_test: pd.DataFrame = x_test
        self.y_train: np.ndarray = y_train
        self.y_test: np.ndarray = y_test

    def to_csv(self, dir_path: str):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        self.x_train.to_csv(f'{dir_path}/x_train.csv', index=False)
        self.x_test.to_csv(f'{dir_path}/x_test.csv', index=False)
        pd.DataFrame(self.y_train).to_csv(f'{dir_path}/y_train.csv', index=False)
        pd.DataFrame(self.y_test).to_csv(f'{dir_path}/y_test.csv', index=False)

    def drop_features(self, features_to_drop: list[str]) -> "TrainTestSplit":
        x_train = self.x_train.drop(columns=features_to_drop)
        x_test = self.x_test.drop(columns=features_to_drop)

        return TrainTestSplit(x_train, x_test, self.y_train, self.y_test)

    @staticmethod
    def from_csv_directory(dir_path: str) -> "TrainTestSplit":
        x_train = pd.read_csv(f'{dir_path}/x_train.csv')
        x_test = pd.read_csv(f'{dir_path}/x_test.csv')

        # The y datasets are only one column
        y_train = pd.read_csv(f'{dir_path}/y_train.csv',).iloc[:, 0].values
        y_test = pd.read_csv(f'{dir_path}/y_test.csv').iloc[:, 0].values

        return TrainTestSplit(x_train, x_test, y_train, y_test)


def apply_log_scale(data: TrainTestSplit,
                    features_to_scale) -> TrainTestSplit:
    to_scale_train = data.x_train[features_to_scale]
    to_scale_test = data.x_test[features_to_scale]

    # Replace 0 with np.nan so the number is skipped because log(0) = undefined
    # They will be filled back
    to_scale_train = to_scale_train.replace(0, np.nan)
    to_scale_test = to_scale_test.replace(0, np.nan)

    log_scaled_train = pd.DataFrame()
    log_scaled_test = pd.DataFrame()
    for f in features_to_scale:
        log_scaled_train[f] = np.log10(to_scale_train[f])
        log_scaled_test[f] = np.log10(to_scale_test[f])

    # Replace nan back with 0s
    log_scaled_train = log_scaled_train.fillna(value=0)
    log_scaled_test = log_scaled_test.fillna(value=0)

    # Remove old features and add scaled ones
    log_scaled = data.drop_features(features_to_drop=features_to_scale)
    log_scaled.x_train = pd.concat([log_scaled.x_train, log_scaled_train], axis="columns")
    log_scaled.x_test = pd.concat([log_scaled.x_test, log_scaled_test], axis="columns")

    return log_scaled


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
