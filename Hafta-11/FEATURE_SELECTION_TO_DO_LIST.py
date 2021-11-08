# Feature Selection Yöntemleri Genel Özet

# Filter Methods (Statistical methods: korelasyon, ki-kare, Cramer’s V, vs)
# Wrapper Methods (backward selection, forward selection, stepwise, vs)
# Embeded (Tree Based Methods, Ridge, Lasso, vs)

# Bu yöntemler arasında çok değişkenli etkiler ve hesaplama maliyetleri
# açısından öne çıkan yöntemler: Tree Based Method'lardır.

############################
# TREE BASED SELECTION
############################

# TODO: tum değişkenleri, sayısal değişkenleri, kategorik değişkenleri,
#  ve yeni türetilen değişkenlerin isimlerini ayrı ayrı listelerde tut.

all_cols = []  # target burada olmamalı
num_cols = [col for col in df.columns if df[col].dtypes != 'O']
cat_cols = []
new_cols = []
target = []

# TODO: random forests, lightgbm, xgboost ve catboost modelleri geliştir.
#  Bu modellere orta şekerli hiperparametre optimizasyonu yap. Final modelleri kur.
#  Bu modellerin her birisine feature importance sor. Gelen feature importance'ların hepsini bir df'te topla.
# Gerekirse skorlar için standartlaştırma yap ki gerekecektir.
#  Bu df'in sütunları aşağıdaki şekilde olsun:

# model_name feature_name feature_importance

# TODO: oluşacak df'i analiz et. Grupby ile importance'in ortalamasını alıp, değişken önemlerini küçükten büyüğe sırala.
#  En önemli değişkenleri bul. Sıfırdan küçük olan importance'a sahip değişkenleri sil.
#  Nihayi olarak karar verdiğin değişkenlerin adını aşağıdaki şekilde sakla:

tree_based_features = []

# TODO: Önemli not. Yukarıdaki işlemler neticesinde catboost'un sonuçlarına özellikle odaklanıp
#  kategorik değişkenlerin incelenmesi gerekmektedir.
#  Çalışmanın başında tutulmuş olan cat_cols listesini kullanarak
#  sadece categorik değişkenler için hangi ağacın nasıl bir önem düzeyi verdiğini inceleyiniz
#  ve diğer algoritmalarca önemsiz fakat catboost tarafından önemli olan değerlendirilen değişkenleri bulunuz
#  ve aşağıdaki şekilde kaydediniz:

catboost_cat_features = []

# TODO: tree_based_features listesinde yer ALMAYIP catboost_cat_features listesinde YER ALAN değişkenleri bulunuz
#  ve bu değişkenleri tree_based_features listesine ekleyiniz.




############################
# STATISTICAL SELECTION
############################


# TODO sayısal bağımsız değişkenlerin birbiri arasındaki korelasyonlarına bakıp birbiri ile
#  yüzde 75 üzeri korelasyonlu olan değişkenler arasından 1 tane değişkeni rastgele seçiniz
#  ve değişkenlerin isimlerini aşağıdaki gibi kaydediniz:
#  elenen değişkenlerin isimlerini de aşağıdaki gibi kaydediniz:

correlation_based_features = []
correlation_based_dropped_features = []


# TODO: tree_based_features listesinde olup aynı anda correlation_based_dropped_features listesinde olan feature'lara
#  odaklanarak inceleme yapınız ve gerekli gördüğünüz değişkenleri tree_based_features listesinden siliniz ya da
#  drop listesinden agaç listesine taşıyınız.

# TODO: veri setindeki kategorik değişkenler ile bağımlı değişken arasında chi-squared testi uygulayınız
#  ve bu test sonucuna göre target ile dependency'si bulunan yani test sonucuna göre anlamlı olan
#  değişkenleri aşağıdaki şekilde saklayınız:

cat_cols_chi = []

# TODO: yukarıdan gelecek olan değişkenler ile tree_based_features listelerini karşılaştırınız. Durumu analiz ediniz.
#  cat_cols_chi listesinde olup tree_based_features listesinde olmayan değişkenleri eklemeyi değerlendiriniz.
#  ya da cat_cols_chi'de olmayıp tree_based_features'de olan değişkenleri çıkarmayı değerlendiriniz.
#  Değerlendirmekten kastedilen yoruma açık olarak istediğiniz şekilde değerlendirebilecek olmanızdır.


# TODO: netice olarak en sonda aşağıdaki isimlendirme ile seçilmis feature'ları kaydediniz:


selected_features = []

# TODO: seçilmiş feature'lar ile model tuning yaparak lightgbm için hiperparametre optimizasyonu yapınız.
# TODO: yeni hiperparametrelerle final modeli oluşturunuz.
