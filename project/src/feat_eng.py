import numpy as np
import pandas as pd
# import project.src.utils as utils
# import project.src.training as tr
from operator import itemgetter
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from category_encoders import LeaveOneOutEncoder
from sklearn.feature_selection import RFECV, RFE

def get_missing_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe containing information about the presence of missing values
    in the columns of the provided dataframe.

    :param df: dataframe to retrieve the info for
    :return: Dataframe with columns "column", "dtype", "missing count", "missing %"
    """

    columns = []
    dtypes = []
    missing_pct = []
    missing_num = []
    for col in df:
        columns.append(col)
        dtypes.append(df[col].dtype)

        n = df[col].isnull().sum()
        missing_num.append(n)

        pct = (n / len(df)) * 100
        missing_pct.append(pct)

    return pd.DataFrame(data={
        "column": columns,
        "dtype": dtypes,
        "missing count": missing_num,
        "missing %": missing_pct
    })