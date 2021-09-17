############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################

# Amacımız online retail II veri setine birliktelik analizi uygularak kullanıcılara ürün satın alma sürecinde
# ürün önermek

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

############################################
# Veri Ön İşleme
############################################

# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("/Users/mebaysan/Desktop/VBO-VeriBilimiKampi-Donem6/Datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


df = retail_data_prep(df)

############################################
# ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################

# Verinin gelmesini istediğimiz durum:

# Satırlar her bir işlemi (sepeti) temsil ederken, sütunlar ürünleri temsil eder. Hangi sepette hangi ürün var binary olarak temsil edilir. Ürün sepette varsa keişim 1 olur, yoksa 0 olur.

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1


df_fr = df[df['Country'] == "France"] # işlemleri Fransa özeline indiriyoruz

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20) # hangi sepette (Invoice) hangi üründen (Description) kaç adet var

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5] # istediğim veri yapısını hazırlıyorum, ürün sepette yoksa NaN geliyor. unstack sayesinde pivot yapıyorum

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5] # veri yapımda NaN gelmesin diye eğer sepette ürün yoksa 0 yazıyorum. Hala bir problem var, ürünün sepette kaç adet olduğunu  gözlemliyorum çünkü agg yaptık ve miktarın sum'ını aldık.


df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(
    lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5] # yukarıdaki veri yapısını bir adım daha atarak tam istediğim hale getiriyorum. Eğer ürün sepette geçiyorsa (toplam quantity'i 1 veya daha büyük ise) 1, geçmiyorsa 0 yazıyorum. applymap fonksiyonu tüm hücrelere uygulanır

df_fr.groupby(['Invoice', 'Description']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


def create_invoice_product_df(dataframe, id=False): # ister id'e göre ister ürün adına (description) göre veri yapısını oluştururum
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


fr_inv_pro_df = create_invoice_product_df(df_fr)
fr_inv_pro_df.head()

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)
fr_inv_pro_df.head()

def check_id(dataframe, stock_code): # id'e göre arl için oluşturduğum veri yapısında id'nin adını öğrenmek için bir fonksiyon
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_fr, 10002) # 10002 id'li ürünün adı ne?

############################################
# Birliktelik Kurallarının Çıkarılması
############################################

# Tüm olası ürün birlikteliklerinin olasılıkları
frequent_itemsets = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True) # apriori algoritmasını uyguluyoruz. min 0.01 olasılıkla birlikte satılabilecek ürünler gelsin diyoruz. her bir ürünün birbirleriyle beraber satılma olasılıkları
frequent_itemsets.sort_values("support", ascending=False)


# Birliktelik kurallarının çıkarılması:
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01) # apriori algoritmasını uyguladığımız verisetinden support metriğini kullanarak birliktelik kurallarını çıkartıyoruz
rules.sort_values("support", ascending=False).head()
# antecedents -> ilk ürün
# consequents -> sonraki ürün(ler)
# antecedent support -> ilk ürünün tek bağına gözlenme olasılığı
# consequent support -> sonraki ürün(ler) ün gözlenme olasılığı
# support -> ilk ve sonraki ürün(ler) beraber gözlenme olasılığı
# confidence -> apriori algoritmasındaki confidence değeri, ilk ürün satıldığında sonraki ürün(ler) satılma olasılığı
# lift -> apriori algoritmasındaki lift değeri, ilk ürün satıldığında sonraki ürün(ler) satılma olasılığı lift kat kadar artar
# leverage -> kaldıraç demektir. lift'e benzer fakat support'u yüksek değerlere öncelik verme eğilimindedir. Bir tık yanlı bir metriktir.
# conviction -> consequents olmadan antecedents'in olması
rules.sort_values("lift", ascending=False).head(500)

# support: İkisinin birlikte görülme olasılığı
# confidence: X alındığında Y alınma olasılığı.
# lift: X alındığında Y alınma olasılığı şu kadar kat artıyor.

############################################
# Çalışmanın Scriptini Hazırlama
############################################

import pandas as pd

pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

# örnek olarak başka bir ülkeye göre yapalım.
rules_grm = create_rules(df, country="Germany")
rules_grm.sort_values("lift", ascending=False).head(50)


############################################
# Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################

# Örnek:
# Kullanıcı örnek ürün id: 22492

product_id = 22492
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)


recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]): # ilk ürün (antecedents) sütununda geziyoruz
    for j in list(product): # ilgili ürünü(ler) listeye çeviriyorum
        if j == product_id: # eğer antecedents hücresinde sepetteki ürünüm varsa, ilgili satırın consequents hücresine gidiyorum ve ordaki ürünleri listeye atıyorum
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])


recommendation_list[0:2]

check_id(df, 22556)

check_id(df, recommendation_list[0])


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]


arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)
arl_recommender(rules, 22492, 3)

