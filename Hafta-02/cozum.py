import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Ödev 1 Çözüm:

def cat_summary(dataframe, col_name, plot=False):
    if type(col_name) == list:
        for col in col_name:
            cat_summary(dataframe, col)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


df = sns.load_dataset('tips')
cat_summary(df, ['sex', 'day'])


###############################################

# Ödev 2 Çözüm:

def cat_summary(dataframe, col_name, plot=False):
    """Summarize categorical column(s)

    This function will summarize the categorical column(s). It can take a list as 2nd param.
    Returns the ratio for each unique variable.

    Parameters
    ----------
    dataframe : DataFrame
         Pandas DataFrame for getting categorical variables
    col_name: str or list
         Column name for creating summary, If it's list, function will return summary for each element of list
    plot: bool
        Will it draw a plot?

    Returns
    ----------
        There is no return value for creating a new variable
        It will print the summary of column(s)

    Examples
    ----------
    import seaborn as sns
    df = sns.load_dataset('tips')
    cat_summary(df, ['sex', 'day'])
    """
    if type(col_name) == list:
        for col in col_name:
            cat_summary(dataframe, col)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


help(cat_summary)
