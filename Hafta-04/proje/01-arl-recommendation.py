import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


df_ = pd.read_excel(
    "/Users/mebaysan/Desktop/VBO-VeriBilimiKampi-Donem6/Datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")


def prepare_retail(dataframe):
    # veri setini hazırlama
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe


def create_apriori_datastructure(dataframe, id=False):
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
    if id:
        grouped = germany_df.groupby(
            ['Invoice', 'StockCode'], as_index=False).agg({'Quantity': 'sum'})
        apriori_datastructure = pd.pivot(data=grouped, index='Invoice', columns='StockCode', values='Quantity').fillna(
            0).applymap(lambda x: 1 if x > 0 else 0)
        return apriori_datastructure
    else:
        grouped = germany_df.groupby(
            ['Invoice', 'Description'], as_index=False).agg({'Quantity': 'sum'})
        apriori_datastructure = pd.pivot(data=grouped, index='Invoice', columns='Description', values='Quantity').fillna(
            0).applymap(lambda x: 1 if x > 0 else 0)
        return apriori_datastructure


def get_item_name(dataframe, stock_code):
    # id'e göre arl için oluşturduğum veri yapısında id'nin adını öğrenmek için bir fonksiyon
    # liste de alabilir, gelen id'lerin adını verir
    if type(stock_code) != list:
        product_name = dataframe[dataframe["StockCode"] ==
                                 stock_code][["Description"]].values[0].tolist()
        return product_name
    else:
        product_names = [dataframe[dataframe["StockCode"] == product][[
            "Description"]].values[0].tolist()[0] for product in stock_code]
        return product_names


def get_rules(apriori_df, min_support=0.01):
    ############################################
    # Birliktelik Kurallarının Çıkarılması
    ############################################
    # Tüm olası ürün birlikteliklerinin olasılıkları
    # apriori algoritmasını uyguluyoruz. min 0.01 olasılıkla birlikte satılabilecek ürünler gelsin diyoruz. her bir ürünün birbirleriyle beraber satılma olasılıkları
    frequent_itemsets = apriori(
        apriori_df, min_support=min_support, use_colnames=True)
    # Birliktelik kurallarının çıkarılması:
    # apriori algoritmasını uyguladığımız verisetinden support metriğini kullanarak birliktelik kurallarını çıkartıyoruz
    rules = association_rules(
        frequent_itemsets, metric="support", min_threshold=min_support)
    # antecedents -> ilk ürün
    # consequents -> sonraki ürün(ler)
    # antecedent support -> ilk ürünün tek bağına gözlenme olasılığı
    # consequent support -> sonraki ürün(ler) ün gözlenme olasılığı
    # support -> ilk ve sonraki ürün(ler) beraber gözlenme olasılığı
    # confidence -> apriori algoritmasındaki confidence değeri, ilk ürün satıldığında sonraki ürün(ler) satılma olasılığı
    # lift -> apriori algoritmasındaki lift değeri, ilk ürün satıldığında sonraki ürün(ler) satılma olasılığı lift kat kadar artar
    # leverage -> kaldıraç demektir. lift'e benzer fakat support'u yüksek değerlere öncelik verme eğilimindedir. Bir tık yanlı bir metriktir.
    # conviction -> consequents olmadan antecedents'in olması
    return rules


def recommend_products(rules_df, product_id, rec_count=5):
    # rules_df -> kuralların oluşturulduğu dataframe
    # product_id -> şu an sepete eklenen ürün
    # rec_count -> tavsiye edilecek ürün adedi
    # lift metriğine göre kuralların tutulduğu veri setini sırala
    sorted_rules = rules_df.sort_values('lift', ascending=False)
    recommended_products = []  # tavsiye edilen ürünleri tutmak için bir liste oluştur

    # satın alınması muhtemel ilk ürünleri gez
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):  # her ürünü veya ürün grubunu bir listeye ata
            if j == product_id:  # eğer döndüğün liste product_id'e eşitse ki bu sepetteki ürün id'si demek
                # tavsiye edilen ürün listesine consequents sütunundaki ürünü ekle
                recommended_products.append(
                    list(sorted_rules.iloc[i]["consequents"]))

    # aynı üründen birden fazla olabilir diye bunu set veri yapısına çevir ve her biri unique olsun
    recommended_products = list(
        {item for item_list in recommended_products for item in item_list})
    # tavsiye edilen ürün listesinin baştan rec_count kadarını döndür
    return recommended_products[:rec_count]



df = df_.copy()
df = prepare_retail(df)  # veri setini hazırlıyorum
# sadece Germany işlemlerini seçiyorum
germany_df = df[df['Country'] == 'Germany']

# apriori algoritmasına uygun veri yapısını hazırlıyorum
germany_apriori_df = create_apriori_datastructure(germany_df, True)
rules = get_rules(germany_apriori_df)  # kuralları öğreniyorum

TARGET_PRODUCT_ID_1 = 21987
TARGET_PRODUCT_ID_2 = 23235
TARGET_PRODUCT_ID_3 = 22747

# hedef productların isimleri
get_item_name(germany_df, [TARGET_PRODUCT_ID_1,TARGET_PRODUCT_ID_2, TARGET_PRODUCT_ID_3])

def get_golden_shot(target_id,dataframe,rules):
    target_product = get_item_name(dataframe,target_id)[0]
    recomended_product_ids = recommend_products(rules, target_id)
    recomended_product_names = get_item_name(dataframe,recommend_products(rules, target_id))
    print(f'Sepetteki Ürün ID: {target_id}\nÜrün Adı: {target_product}')
    print(f'Önerilen Ürünler: {recomended_product_ids}\nÜrün İsimleri: {recomended_product_names}')

get_golden_shot(TARGET_PRODUCT_ID_1,germany_df,rules)

get_golden_shot(TARGET_PRODUCT_ID_2,germany_df,rules)

get_golden_shot(TARGET_PRODUCT_ID_3,germany_df,rules)