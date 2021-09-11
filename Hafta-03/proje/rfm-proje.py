import pandas as pd
from datetime import datetime
# Dataset Link: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# SORU 1
# ---------------------------------------------
# Görev 1
raw_data = pd.read_excel(
    '/Users/mebaysan/Desktop/VBO-VeriBilimiKampi-Donem6/Datasets/online_retail_II.xlsx', sheet_name='Year 2010-2011')

df = raw_data.copy()

# Görev 2
df.describe().T

# Görev 3
df.isna().sum()

# Görev 4
df.dropna(inplace=True)

# Görev 5
df.StockCode.nunique()

# GÖrev 6
df.groupby('StockCode').agg('count')

# Görev 7
df.groupby('StockCode').agg({'Invoice': 'count'}).sort_values(
    'Invoice', ascending=False).head(5)

# Görev 8
df = df[~df['Invoice'].str.contains('C', na=False)]

# Görev 9
df.loc[:, 'TotalPrice'] = df['Quantity'] * df['Price']
df.head()

# SORU 2
# ---------------------------------------------
today_date = datetime(2011, 12, 11)

rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (today_date - x.max()).days,
    'Invoice': lambda x: x.nunique(),
    'TotalPrice': lambda x: x.sum()
})

rfm.columns = ['recency', 'frequency', 'monetary']

rfm = rfm[rfm['monetary'] > 0]


# SORU 3
# ---------------------------------------------
# qcut küçükten büyüğe sıralıyor ve sonra çeyrekliklere göre labelları atıyor. Bu değişken için en küçük değer en iyi olduğundan labelları  5 > 1 verdik
rfm.loc[:, 'recency_score'] = pd.qcut(rfm['recency'], 5, [5, 4, 3, 2, 1])
rfm.loc[:, 'frequency_score'] = pd.qcut(
    rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm.loc[:, 'RFM_SCORE'] = rfm['recency_score'].astype(
    str) + rfm['frequency_score'].astype(str)

# SORU 4
# ---------------------------------------------
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm.head()

# SORU 5
# ---------------------------------------------
rfm.groupby('segment').agg({
    'recency':['min','max','mean'],
    'frequency':['count','min','max','mean'],
    'monetary':['min','max','sum','mean']
})

"""
Önemli Gördüğüm 3 Segment

1. Segment -> at_Rist
2. Segment -> cant_loose
3. Segment -> need_attention


"""

rfm[rfm['segment'] == 'loyal_customers'].to_excel('loyal_customers.xlsx')