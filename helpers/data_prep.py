import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """Returns outlier threshold limits: min and max

    Args:
        dataframe (pd.DataFrame): A dataframe
        col_name (str): The column name for creating threshold values
        q1 (float, optional): Q1 value for setting the minimum threshold. Defaults to 0.25.
        q3 (float, optional): Q3 value for setting the maximum thresold. Defaults to 0.75.

    Returns:
        int, int: min threshold and max threshold
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def grab_outliers(dataframe, col_name, index=False, q1=0.25, q3=0.75):
    """Prints the outlier observations

    Args:
        dataframe (pd.DataFrame): A dataframe
        col_name (str): The column name for seacrhing outlier values
        index (bool, optional): If it's true, returns the indices of outliers. Defaults to False.

    Returns:
        list: array of outliers' indices
    """
    low, up = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low)
              | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low)
              | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[(
            (dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    """Is there any outlier in the column

    Args:
        dataframe (pd.DataFrame): A dataframe
        col_name (str): The column name for checking the outliers

    Returns:
        bool: Returns True if there is outlier in the column name else False
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    """Replace outliers by thresholds

    Args:
        dataframe (pd.DataFrame): A dataframe
        variable (str): The column name for applying the thresold process
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def remove_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    """Remove the outlier observations

    Args:
        dataframe (pd.DataFrame): A dataframe
        col_name (str): The column name for removing the outliers

    Returns:
        pd.DataFrame: The dataframe which removed the outliers
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    df_without_outliers = dataframe[~(
        (dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def missing_values_table(dataframe, na_name=False):
    """Shows the missing value counts and ratios of columns

    Args:
        dataframe (pd.DataFrame): A dataframe
        na_name (bool, optional): If it's true, it returns the column name of missing values. Defaults to False.

    Returns:
        list: Missing value column names
    """
    na_columns = [
        col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() /
             dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)],
                           axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    """It prints a summary of between target column and na columns

    Args:
        dataframe (pd.DataFrame): A dataframe
        target (str): The target variable column name
        na_columns (list): Column names of missing values
    """
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    """Kategorik değişkenlerdeki rare olma durumunu verir

    Args:
        dataframe (pd.DataFraöe): Dataframe
        target (str): Hedef değişken ismi
        cat_cols (list): Kategorik değişken isimlerinin bulunduğu liste
    """
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, cat_cols, rare_perc):
    """
    1'den fazla rare varsa düzeltme yap. durumu göz önünde bulunduruldu.
    rare sınıf sorgusu 0.01'e göre yapıldıktan sonra gelen true'ların sum'ı alınıyor.
    eğer 1'den büyük ise rare cols listesine alınıyor.

    Args:
        dataframe (pd.DataFrame): Dataframe
        cat_cols (list): Kategorik değişkenlerin isimlerinin bulunduğu liste
        rare_perc (float): Yüzde kaç oranının altı RARE olarak etiketlensin

    Returns:
        pd.DataFrame: rare_perc altında kalan sınıf oranları RARE olarak etiketlenmiş veri seti
    """

    temp_df = dataframe.copy()
    rare_columns = [col for col in cat_cols if (
        temp_df[col].value_counts() / len(temp_df) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = temp_df[col].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[col] = np.where(temp_df[col].isin(
            rare_labels), 'Rare', temp_df[col])

    return temp_df


def plot_importance(model, features, num=5, save=False):
    """Modeli etkileyen değişkenlerin etkisini görselleştirir

    Parametre olarak aldığı modeli etkileyen değişkenleri görselleştirir

    Args:
        model (model): Makine öğrenmesi modeli
        features ([type]): features dataframe'i (X)
        num (int, optional): Grafikte ilk kaç değişken gösterilecek. Default 5
        save (bool, optional): Grafik lokale kaydedilecek mi? Default False
    """
    num = len(features)
    feature_imp = pd.DataFrame(
        {'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

