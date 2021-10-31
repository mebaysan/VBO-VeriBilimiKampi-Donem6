import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
# Dataset Link: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# * Veriyi Okuma
# ---------------------------------------------
raw_data = pd.read_excel(
    '/Users/mebaysan/Desktop/VBO-VeriBilimiKampi-Donem6/Datasets/online_retail_II.xlsx', sheet_name='Year 2010-2011')

df = raw_data.copy()

df.head()

# * RFM'e Uygun Veri Seti Hazırlama
# ---------------------------------------------
df = df[~df['Invoice'].str.contains('C', na=False)]
df.loc[:, 'TotalPrice'] = df['Quantity'] * df['Price']

df.head()

# * RFM Metrikleri Hesaplama
# ---------------------------------------------
today_date = datetime(2011, 12, 11)

rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (today_date - x.max()).days,
    'Invoice': lambda x: x.nunique(),
    'TotalPrice': lambda x: x.sum()
})

rfm.columns = ['recency', 'frequency', 'monetary']

rfm = rfm[rfm['monetary'] > 0]

rfm.head()

# * K-Means & Optimum Küme Sayısı Bulmak
# ---------------------------------------------
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(rfm)
# ---------------------------------------------
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()
elbow.elbow_value_

# * Final K-Means
# ---------------------------------------------
final_kmeans = KMeans(n_clusters=elbow.elbow_value_)
final_kmeans.fit(df)
final_kmeans.labels_

# * Label'ları Eşleştirme
# ---------------------------------------------
rfm['Cluster_Label'] = final_kmeans.labels_
rfm.head()

# * RFM Skorlama
# ---------------------------------------------
rfm.loc[:, 'recency_score'] = pd.qcut(rfm['recency'], 5, [5, 4, 3, 2, 1])
rfm.loc[:, 'frequency_score'] = pd.qcut(
    rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm.loc[:, 'RFM_SCORE'] = rfm['recency_score'].astype(
    str) + rfm['frequency_score'].astype(str)

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


# * İnceleme
# ---------------------------------------------
pd.crosstab(rfm['segment'],rfm['Cluster_Label']) # Hangi segmentte kaç tane hangi cluster'dan var)