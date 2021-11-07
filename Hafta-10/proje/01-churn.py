import warnings
import findspark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import when, count, col, pandas_udf
from pyspark.sql.types import FloatType


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# download linkinden spark'ı indirdim ve ilgili path altına çıkarttım
findspark.init("/Users/mebaysan/spark/spark-3.2.0-bin-hadoop3.2") 

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

# oluşturduğumuz spark instance üzerinden veriyi okuyorum
spark_df = spark.read.csv("datasets/churn2.csv", header=True, inferSchema=True) 

print("Shape: ", (spark_df.count(), len(spark_df.columns))) # kaç satır X kaç sütun

spark_df.show()

# tüm değişken isimlerini küçük harfe çevirme
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])

# betimsel istatistikler
spark_df.describe().toPandas().T

# exited değişkenine göre gruplayıp toplam kaçar adet kayıt olduğu
spark_df.groupby('exited').count().show()


##################################################
# * Data PreProcessing
##################################################
# Kategorik ve numerik değişkenleri ayıralım
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']

spark_df.select(
    [count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]
    ).toPandas().T

##################################################
# * Feature Engineering
##################################################


# * #################################################
@pandas_udf(FloatType())
def press_outliers(col):
    temp_df = spark_df.select(col).toPandas()
    q1 = temp_df.quantile(0.25)[0]
    q3 = temp_df.quantile(0.75)[0]
    iqr = q3 - q1
    low_limit = q1 - 1.5 * iqr
    up_limit = q3 + 1.5 * iqr
    temp_df.loc[temp_df[t] < low_limit] = low_limit
    temp_df.loc[temp_df[t] > up_limit] = up_limit
    temp_df = temp_df.set_index(temp_df.index + 1)    
    return col
# burada bir hata oldu anlayamadım
spark_df = spark_df.withColumn('balance', spark_df.select(press_outliers(spark_df.balance)))
# * #################################################



# age (yaş) değişkenini kategorilere ayırıyorum
# splitleri ver, hangi değişkeni kullanmak istiyorsun, bunu hangi değişken olarak çıkarmak istiyorsun
bucketizer = Bucketizer(splits=[0, 35, 45, 65,100], inputCol="age", outputCol="age_cat") 
# oluşturduğumuz nesneyi kullanarak transform (dönüşüm) işlemini uyguluyoruz
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df) 
spark_df = spark_df.drop('age')

# tenure değişkenine göre kategori oluşturuyorum
spark_df.groupby(spark_df.tenure).count().show()
spark_df = spark_df.withColumn('tenure_cat',
                    when(spark_df['tenure'] <= 3, "c"). # koşullar when() ile zincirleme birbirine bağlanır
                    when((3 < spark_df['tenure']) & (spark_df['tenure'] <= 7), "b").
                    when(spark_df['tenure'] > 7, 'a'))

# gender değişkeni label encoding
indexer = StringIndexer(inputCol='gender',outputCol='gender_cat')
indexer.fit(spark_df).transform(spark_df)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("gender_cat", temp_sdf["gender_cat"].cast("integer")) 
spark_df = spark_df.drop('gender')

# geography değişkeni label encoding
indexer = StringIndexer(inputCol='geography',outputCol='geography_cat')
indexer.fit(spark_df).transform(spark_df)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("geography_cat", temp_sdf["geography_cat"].cast("integer")) 
spark_df = spark_df.drop('geography')

# tenure_cat değişkeni label encoding
indexer = StringIndexer(inputCol='tenure_cat',outputCol='tenure_cat_label')
indexer.fit(spark_df).transform(spark_df)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("tenure_cat", temp_sdf["tenure_cat_label"].cast("integer")) 
spark_df = spark_df.drop('tenure_cat_label')

# gereksiz değişkenleri siliyorum
spark_df = spark_df.drop('customerid')
spark_df = spark_df.drop('rownumber')
spark_df = spark_df.drop('surname')

# tenure_cat değişkeni one-hot encoding
encoder = OneHotEncoder(inputCols=["tenure_cat"], outputCols=["tenure_cat_ohe"])
spark_df = encoder.fit(spark_df).transform(spark_df)


############################
# * Feature'ların Tanımlanması
############################
"""
Spark derki; bana bağımlı değişkeni 1 formda gönder ve o formun adı "label" olsun. 
             Bağımsız değişkenleri de VectorAssembler'dan geçir ve adı "features" olsun der.
"""
independent_cols = [col for col in spark_df.columns if col != 'exited']

# [cols] değişkenlerini alıp VectorAssembler'dan geçiriyorum ve oluşan yeni değişkene "features" adını veriyorum
va = VectorAssembler(inputCols=independent_cols, outputCol="features") 
# "features" değişkenini spark_df'e ekliyorum
va_df = va.transform(spark_df) 
va_df.show()

va_df = va_df.withColumn('label',va_df.exited)

# Spark; "features" ve "label" istediğinden dolayı bu iki değişkeni seçiyorum
final_df = va_df.select("features", "label") 
final_df.show(5)

############################
# * Modelling
############################
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=17) # eğitim ve test olarak veri setini ayırıyorum: 0.8 eğitim, 0.2 test

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df) # train veri seti ile eğit
y_pred = gbm_model.transform(test_df) # test veri seti ile test et
y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count() # accuracy

############################
# * Model Tuning
############################
evaluator = BinaryClassificationEvaluator()

# sklearn'deki gridsearch cv'e benzer
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
ac.filter(ac.label == ac.prediction).count() / ac.count() # accuracy

############################
# * Model Evaluation
############################
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


