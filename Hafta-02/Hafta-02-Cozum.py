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

# Ödev 3 (PROJE) Çözüm
df = pd.read_csv(
    '/Users/mebaysan/Desktop/VBO-VeriBilimiKampi-Donem6/Hafta-02/persona.csv')

# Görev 1


def get_summary_about_df(dataframe, head=5):  # Soru 1
    print(f"############### HEAD {head} ###################")
    print(dataframe.head(5))
    print("##################################")

    print(f"############### Describe ###################")
    print(dataframe.describe().T)
    print("##################################")

    print(f"############### Info ###################")
    print(dataframe.info())
    print("##################################")


get_summary_about_df(df)


def get_col_summary(df, col):
    print(f"Total count of unique values: {df[col].nunique()}")
    print(f"Total value counts for each value\n{df[col].value_counts()}")


get_col_summary(df, 'SOURCE')  # Soru 2

get_col_summary(df, 'PRICE')  # Soru 3 - 4

get_col_summary(df, 'COUNTRY')  # Soru 5

df.groupby('COUNTRY').sum()[['PRICE']]  # Soru 6

df.groupby('SOURCE').count()  # Soru 7

df.groupby('COUNTRY').mean()[['PRICE']]  # Soru 8

df.groupby('SOURCE').mean()[['PRICE']]  # Soru 9

df.groupby(['COUNTRY', 'SOURCE']).mean()[['PRICE']]  # Soru 10

# Görev 2
df.groupby(['COUNTRY', 'SOURCE', 'SEX', 'AGE']).mean()[['PRICE']]

# Görev 3
df.groupby(['COUNTRY', 'SOURCE', 'SEX', 'AGE']).mean()[
    ['PRICE']].sort_values('PRICE', ascending=False)  # Soru 1

agg_df = df.groupby(['COUNTRY', 'SOURCE', 'SEX', 'AGE']).mean()[
    ['PRICE']].sort_values('PRICE', ascending=False)  # Soru 2

# Görev 4
agg_df.reset_index(inplace=True)

# Görev 5
agg_df.loc[:, 'AGE_CAT'] = pd.cut(agg_df['AGE'], [0, 18, 23, 30, 41, 70], labels=[
                                  '0_18', '19_23', '24_30', '31_40', '41_70'])

# Görev 6
agg_df.loc[:, 'customers_level_based'] = agg_df.apply(lambda x: f"{x['COUNTRY'].upper()}_{x['SOURCE'].upper()}_{x['SEX'].upper()}_{x['AGE_CAT'].upper()}", axis=1)
agg_df.head()

level_df = agg_df[['customers_level_based','PRICE']].groupby('customers_level_based').mean()
# PRICE_LEVELED değişkenine aynı tipteki sınıflanmış müşterilerin ortalamalarını atıyoruz
agg_df.loc[:,'PRICE_LEVELED'] = agg_df['customers_level_based'].apply(lambda x: level_df[level_df.index == x]['PRICE'][0])

agg_df.head()

# Görev 7
agg_df.loc[:,'SEGMENT'] = pd.qcut(agg_df['PRICE'], 4, labels=['D', 'C', 'B', 'A'])

agg_df[agg_df['SEGMENT'] == 'C']

new_user = 'TUR_ANDROID_FEMALE_31_40'
new_user2 = 'FRA_IOS_FEMALE_31_40'

agg_df[agg_df['customers_level_based'] == new_user]
agg_df[agg_df['customers_level_based'] == new_user2]


