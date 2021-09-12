import pandas as pd
import datetime as dt
from sqlalchemy import create_engine
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

raw_data = pd.read_excel(
    '/Users/mebaysan/Desktop/VBO-VeriBilimiKampi-Donem6/Datasets/online_retail_II.xlsx', sheet_name='Year 2010-2011')

df = raw_data.copy()

df.head()

# Ön işleme
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df['Country'] == 'United Kingdom']
df['TotalPrice'] = df['Price'] * df['Quantity']

# Lifetime Hazırlama
today_date = dt.datetime(2011, 12, 11)

cltv = df.groupby('Customer ID').agg({
    'InvoiceDate': [
        lambda x: (x.max() - x.min()).days,  # tx - recency
        lambda x: (today_date - x.min()).days  # T
    ],
    'Invoice': lambda x: x.nunique(),  # frequency
    'TotalPrice': lambda x: x.sum()  # monetary
})

cltv.columns = cltv.columns.droplevel(0)

cltv.columns = ['recency', 'T', 'frequency', 'monetary']

cltv = cltv[cltv['monetary'] > 0]

# monetary değerini satın alma başına ortalama kazanç olarak ifade ediyoruz
cltv['monetary'] = cltv['monetary'] / cltv['frequency']

# recency ve T değişkenlerini günlük'ten haftalık'a çeviriyoruz. Model haftalık olarak çalışıyor
cltv['recency'] = cltv['recency'] / 7
cltv['T'] = cltv['T'] / 7

cltv = cltv[(cltv['frequency'] > 1)]

# BG-NBD Model Kurma
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv['frequency'], cltv['recency'], cltv['T'])

# Örnek: 1 hafta içinde en çok satın alım yapması beklenen müşteriler
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv['frequency'],
                                                        cltv['recency'],
                                                        cltv['T']).sort_values(ascending=False).head(10)


# Gamma Gamma Model Kurma
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv['frequency'], cltv['monetary'])

# SORU 1: 6 aylık cltv hesaplayın
cltv['6_month_cltv_pred'] = ggf.customer_lifetime_value(
    bgf, cltv['frequency'], cltv['recency'], cltv['T'], cltv['monetary'], time=6, freq='W')

# SORU 2: 1 aylık ve 12 aylık CLTV hesaplayın
cltv['1_month_cltv_pred'] = ggf.customer_lifetime_value(
    bgf, cltv['frequency'], cltv['recency'], cltv['T'], cltv['monetary'], time=1, freq='W')

cltv['12_month_cltv_pred'] = ggf.customer_lifetime_value(
    bgf, cltv['frequency'], cltv['recency'], cltv['T'], cltv['monetary'], time=12, freq='W')

cltv.sort_values('1_month_cltv_pred', ascending=False).head(
    10).index == cltv.sort_values('12_month_cltv_pred', ascending=False).head(10).index  # 1 aylık ve 12 aylık CLTV tahminlerindeki en değerli 10 müşteri aynıdır


# SORU 3: 6 aylık CLTV'ye göre tüm müşterileri 4 gruba ayırın ve grup isimlerini veri setine ekleyin
cltv['6_month_cltv_pred_segment'] = pd.qcut(
    cltv['6_month_cltv_pred'], 4, ['D', 'C', 'B', 'A'])

cltv.groupby('6_month_cltv_pred_segment').agg({'count', 'sum', 'mean'})
"""
- C ve A segmentlerinin ortalama recency değerlerinin neredeyse aynı olduğu gözlemlenmiştir
- C segmentindeki müşterilerin 

"""


# Soru 4
final_cltv = cltv.reset_index(
)[['Customer ID', 'recency', 'T', 'frequency', 'monetary']]
final_cltv['expected_purch_1_week'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    1, final_cltv['frequency'], final_cltv['recency'], final_cltv['T'])
final_cltv['expected_purch_1_month'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    4, final_cltv['frequency'], final_cltv['recency'], final_cltv['T'])
final_cltv['expected_average_profit_clv'] = ggf.customer_lifetime_value(
    bgf, final_cltv['frequency'], final_cltv['recency'], final_cltv['T'], final_cltv['monetary'], time=6, freq='W')
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(final_cltv[["expected_average_profit_clv"]])
final_cltv["scaled_clv"] = scaler.transform(
    final_cltv[["expected_average_profit_clv"]])
final_cltv['segment'] = pd.qcut(
    final_cltv['expected_average_profit_clv'], 4, ['D', 'C', 'B', 'A'])


creds = {'user': 'group_06',
         'passwd': 'hayatguzelkodlarucuyor',
         'host': '34.88.156.118',
         'port': 3306,
         'db': 'group_06'}

connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

conn = create_engine(connstr.format(**creds))

final_cltv.to_sql(name='enes_baysan', con=conn,
                  if_exists='replace', index=False)

pd.read_sql("show tables", conn)
