##################################################
# * Churn Prediction using PySpark
##################################################

# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirebilir misiniz?
# Amaç bir bankanın müşterilerinin bankayı terk etme ya da terk etmeme durumunun tahmin edilmesidir.

# 1. Kurulum
# 2. Exploratory Data Analysis
# 3. SQL Sorguları
# 4. Data Preprocessing & Feature Engineering
# 5. Modeling

##################################################
# * Kurulum
##################################################

# https://spark.apache.org/downloads.html
# username/spark dizinin altına indir.

# pyarrow: Farklı veri yapıları arasında çalışma kolaylığı sağlayan bir modul.

# pip install pyspark
# pip install findspark
# pip install PyArrow


import warnings
import findspark
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

findspark.init("/Users/mebaysan/spark/spark-3.2.0-bin-hadoop3.2") # download linkinden spark'ı indirdim ve ilgili path altına çıkarttım

##################################################
# * Spark Giriş
##################################################

# Spark session ayağa kaldırıyorum
# master => hangi server
# appName => Spark uygulama adı
# getOrCreate => ilgili master'da ilgili isimde varsa onu getir yoksa oluştur
spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_giris") \
    .getOrCreate()

sc = spark.sparkContext

# localhost:4040/jobs/ # => Spark arayüzü
# sc.stop() # durdurmak istersek

##################################################
# * Exploratory Data Analysis
##################################################


############################
# Pandas df ile Spark df farkını anlamak.
############################

spark_df = spark.read.csv("datasets/churn.csv", header=True, inferSchema=True) # oluşturduğumuz spark instance üzerinden veriyi okuyorum
type(spark_df)


df = sns.load_dataset("diamonds") # klasik bildiğimiz pandas veri seti
type(df)


df.head()
spark_df.head()

df.dtypes
spark_df.dtypes

df.ndim
# spark_df.ndim # böyle bir attr yok

########################################################
# Reading a json file
# df = spark.read.json(json_file_path)
#
# # Reading a text file
# df = spark.read.text(text_file_path)

# # Reading a parquet file
# df = spark.read.load(parquet_file_path) # or
# df = spark.read.parquet(parquet_file_path)
########################################################

############################
# * Exploratory Data Analysis
############################

# Gözlem ve değişken sayısı
print("Shape: ", (spark_df.count(), len(spark_df.columns)))

# Değişken tipleri
spark_df.printSchema()
spark_df.dtypes

# Değişken seçme
spark_df.Age


# Bir değişkeni görmek
spark_df.select(spark_df.Age).show() # değişken göstermek için show metodu kullanırız
spark_df.take(5)
spark_df.head()
spark_df.show()

# Değişken isimlerinin küçültülmesi
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])


# özet istatistikler
spark_df.describe().show()


# sadece belirli değişkenler için özet istatistikler
spark_df.describe(["age", "churn"]).show()


# Kategorik değişken sınıf istatistikleri
spark_df.groupby("churn").count().show()


# Eşsiz sınıflar
spark_df.select("churn").distinct().show()


# select(): Değişken seçimi
spark_df.select("age", "names").show(5)


# filter(): Gözlem seçimi / filtreleme
spark_df.filter(spark_df.age > 40).show()
spark_df.filter(spark_df.age > 40).count()


# groupby işlemleri
spark_df.groupby("churn").count().show()
spark_df.groupby("churn").agg({"age": "mean"}).show()


# Tüm numerik değişkenlerin seçimi ve özet istatistikleri
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
spark_df.select(num_cols).describe().show()

spark_df.select(num_cols).describe().toPandas().transpose()

# Tüm kategorik değişkenlerin seçimi ve özeti
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']

for col in cat_cols:
    spark_df.select(col).distinct().show()

# Churn'e göre sayısal değişkenlerin özet istatistikleri
for col in num_cols:
    spark_df.groupby("churn").agg({col: "mean"}).show()


##################################################
# * SQL Sorguları
##################################################


spark_df.createOrReplaceTempView("spark_df") # spark_df adında bir tablo oluşturuyorum eğer yoksa, varsa onunla değiştiriyorum

spark.sql("show databases").show() # mevcut session'daki veri tabanları

spark.sql("show tables").show() # mevcut session'daki tablolar

spark.sql("select age from spark_df limit 5").show() # mevcut session'da sql sorgusu çalıştırıyorum

spark.sql("select churn, avg(age) from spark_df group by Churn").show()


spark.sql('select * from spark_df').toPandas().head() # toPandas yardımı ile spark nesnelerini pandas nesnelerine çevirebiliriz
##################################################
# * Data Preprocessing & Feature Engineering
##################################################

############################
# * Missing Values
############################

from pyspark.sql.functions import when, count, col
spark_df.select(
    [count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]
    ).toPandas().T


# eksik değere sahip satırları silmek
spark_df.dropna().show() # missing values kaldırıldı


# tüm veri setindeki eksiklikleri belirli bir değerle doldurmak
spark_df.fillna(50).show() # missing values 50 ile dolduruldu


# eksik değerleri değişkenlere göre doldurmak
spark_df.na.fill({'age': 50, 'names': 'unknown'}).show()




############################
# * Feature Interaction
############################

spark_df = spark_df.withColumn('age_total_purchase', # withColumn ile yeni değişken oluştururuz, olmayan bir değişken adını yazarsak o değişkeni oluşturur
                                spark_df.age / spark_df.total_purchase)
spark_df.show(5)


############################
# * Bucketization / Bining / Num to Cat
############################

spark_df.select('age').describe().toPandas().transpose()


bucketizer = Bucketizer(splits=[0, 35, 45, 65], inputCol="age", outputCol="age_cat") # splitleri ver, hangi değişkeni kullanmak istiyorsun, bunu hangi değişken olarak çıkarmak istiyorsun

spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df) # oluşturduğumuz nesneyi kullanarak transform (dönüşüm) işlemini uyguluyoruz

spark_df.show(20)


spark_df = spark_df.withColumn('age_cat', spark_df.age_cat + 1) # withColumn ile aynı zamanda mevcut değişkeni güncelleyebiliriz


spark_df.groupby("age_cat").count().show()


spark_df.groupby("age_cat").agg({'churn': "mean"}).show()


spark_df = spark_df.withColumn("age_cat", spark_df["age_cat"].cast("integer"))


spark_df.groupby("age_cat").agg({'churn': "mean"}).show()


############################
# * when ile Değişken Türetmek (segment)
############################
# from pyspark.sql.functions import when

spark_df = spark_df.withColumn('segment', when(spark_df['years'] < 5, "segment_b").otherwise("segment_a"))


############################
# * when ile Değişken Türetmek (age_cat_2)
############################

spark_df.withColumn('age_cat_2',
                    when(spark_df['age'] < 36, "young"). # koşullar when() ile zincirleme birbirine bağlanır
                    when((35 < spark_df['age']) & (spark_df['age'] < 46), "mature").
                    otherwise("senior")).show()



############################
# * Label Encoding
############################


spark_df.show(5)


indexer = StringIndexer(inputCol="segment", outputCol="segment_label")

indexer.fit(spark_df).transform(spark_df).show(5)


temp_sdf = indexer.fit(spark_df).transform(spark_df)

spark_df = temp_sdf.withColumn("segment_label", temp_sdf["segment_label"].cast("integer")) # float olan mevcut değişkeni integer olarak cast ediyorum

spark_df = spark_df.drop('segment')

############################
# * One Hot Encoding
############################
# * ** eğer one-hot encoding yapacaksak bile önce label encoding yapmalıyız (Pyspark içinde)
encoder = OneHotEncoder(inputCols=["age_cat"], outputCols=["age_cat_ohe"])
spark_df = encoder.fit(spark_df).transform(spark_df)

spark_df.show(5)


############################
# * TARGET'ın Tanımlanması
############################


stringIndexer = StringIndexer(inputCol='churn', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))
spark_df.show(5)


############################
# * Feature'ların Tanımlanması
############################
"""
Spark derki; bana bağımlı değişkeni 1 formda gönder ve o formun adı "label" olsun. 
             Bağımsız değişkenleri de VectorAssembler'dan geçir ve adı "features" olsun der.
"""
cols = ['age', 'total_purchase', 'account_manager', 'years',
        'num_sites', 'age_total_purchase', 'segment_label', 'age_cat_ohe']


va = VectorAssembler(inputCols=cols, outputCol="features") # [cols] değişkenlerini alıp VectorAssembler'dan geçiriyorum ve oluşan yeni değişkene "features" adını veriyorum
va_df = va.transform(spark_df) # "features" değişkenini spark_df'e ekliyorum
va_df.show()


# Final sdf
final_df = va_df.select("features", "label") # Spark; "features" ve "label" istediğinden dolayı bu iki değişkeni seçiyorum
final_df.show(5)

# StandardScaler
# scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
# final_df = scaler.fit(final_df).transform(final_df)



# Split the dataset into test and train sets.
train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17) # eğitim ve test olarak veri setini ayırıyorum: 0.7 eğitim ...
train_df.show(10)
test_df.show(10)

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))


# spark.createDataFrame()




##################################################
# * Modeling
##################################################

############################
# * Logistic Regression
############################

log_model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_df) # train set ile modeli kurdum
y_pred = log_model.transform(test_df) # test seti ile tahmin et
y_pred.show()


y_pred.select("label", "prediction").show()


# accuracy
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()


evaluator = BinaryClassificationEvaluator(labelCol="label", # hangi değişkende bağımlı değişken
                                        rawPredictionCol="prediction", # tahmin edilen bağımlı değişken hangi değişkende
                                        metricName='areaUnderROC'
                                        )
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)

print("accuracy: %f, precision: %f, recall: %f, f1: %f, roc_auc: %f" % (acc, precision, recall, f1, roc_auc))



############################
# * Gradient Boosted Tree Classifier
############################

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)
y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()




############################
# * Model Tuning
############################

evaluator = BinaryClassificationEvaluator()


gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30])
              .addGrid(gbm.maxIter, [10, 20])
              .build())



cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)


y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()


############################
# * New Prediction
############################


names = pd.Series(["Ali Ahmetoğlu", "Taner Gün", "Berkay", "Polat Konak", "Kamil Atasoy"])
age = pd.Series([18, 43, 34, 50, 40])
total_purchase = pd.Series([5000, 10000, 6000, 30000, 100000])
account_manager = pd.Series([1, 0, 0, 1, 1])
years = pd.Series([20, 10, 3, 8, 30])
num_sites = pd.Series([2, 8, 8, 6, 50])
age_total_purchase = age / total_purchase
segment_label = pd.Series([1, 1, 0, 1, 1])
age_cat_ohe = pd.Series([1, 1, 0, 2, 1])



yeni_musteriler = pd.DataFrame({
    'names': names,
    'age': age,
    'total_purchase': total_purchase,
    'account_manager': account_manager,
    'years': years,
    'num_sites': num_sites,
    "age_total_purchase": age_total_purchase,
    "segment_label": segment_label,
    "age_cat_ohe": age_cat_ohe})


yeni_sdf = spark.createDataFrame(yeni_musteriler)

new_customers = va.transform(yeni_sdf)
new_customers.show(3)

results = cv_model.transform(new_customers)
results.select("names", "prediction").show()


# scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
# new_customers_final = scaler.fit(new_customers).transform(new_customers)
# results = cv_model.transform(new_customers_final)
# results.select("names", "prediction").show()


# sc.stop()



##################################################
# * BONUS: User Defined Functions (UDFs)
##################################################

def age_converter(age):
    if age < 35:
        return 1
    elif age < 45:
        return 2
    elif age <= 65:
        return 3



from pyspark.sql.types import IntegerType, StringType, FloatType
from pyspark.sql.functions import udf

func_udf = udf(age_converter, IntegerType()) # oluşturduğum age_converter fonksiyonu udf olarak tanımlıyorum. 
# udf fonksiyonu 2 parametre alır: fonksiyon, return tipi


spark_df = spark_df.withColumn('age_cat2', func_udf(spark_df['age'])) # oluşturduğum User Defined Function (UDF)'ı değişkene uygulayıp spark df'e ekliyorum

spark_df.show(5)

spark_df.groupby("age_cat2").count().show()

cat_cols.append('age_cat2')


def age_converter(age):
    if age < 35:
        return "young"
    elif age < 45:
        return "mature"
    elif age <= 65:
        return "senior"


func_udf = udf(age_converter, StringType())
spark_df = spark_df.withColumn('age_cat3', func_udf(spark_df['age']))
spark_df.show(5)


def segment(years):
    if years < 5:
        return "segment_b"
    else:
        return "segment_a"


func_udf = udf(segment, StringType())
spark_df = spark_df.withColumn('segment', func_udf(spark_df['years']))


##################################################
# * Pandas UDFs
##################################################
# Creates a pandas user defined function (a.k.a. vectorized user defined function).
# Pandas ile yapabildiğimiz fakat Spark ile yapamadığımız fonksiyonları pandas_udf ile yazarız
spark_df.show(5)

spark_df.withColumn('age_square', spark_df.age ** 2).show()



from pyspark.sql.functions import pandas_udf

@pandas_udf(FloatType())
def pandas_square(col):
    return col ** 2

spark_df.select(pandas_square(spark_df.age)).show()


@pandas_udf(FloatType())
def pandas_log(col):
    import numpy as np
    return np.log(col)

spark_df.withColumn('age_log', pandas_log(spark_df.age)).show()