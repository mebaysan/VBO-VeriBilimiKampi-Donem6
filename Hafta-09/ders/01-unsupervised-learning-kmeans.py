################################
# * Unsupervised Learning
################################

# * Unsupervised Learning
# * K-Means
# * Hierarchical Clustering
# * Principal Component Analysis
# * BONUS: Principal Component Regression
# * BONUS: PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme


# pip install yellowbrick

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer

################################
# * K-Means
################################

df = pd.read_csv("datasets/USArrests.csv", index_col=0) # çeşitli eyaletlere ilişkin suç istatistikleri
df.head()

df.isnull().sum()
df.describe().T

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df) # veri setini scale ettik. 

df[0:5] # Elimizde sayısal (scale edilmiş) değerler var

kmeans = KMeans(n_clusters=4) # 4 merkezli bir model oluşturuyorum. 4 adet küme elde edeceğim.
k_fit = kmeans.fit(df) # modeli uyguluyorum.

k_fit.n_clusters # küme sayısı
k_fit.cluster_centers_ # kümelerin merkezleri (gözlem birimleri)
k_fit.labels_ # her bir eyaletin hangi kümeye girdiğini ifade ediyor
k_fit.inertia_ # toplam hata

df[0:5]

################################
# * Kümelerin Görselleştirilmesi
################################
df = pd.read_csv("datasets/USArrests.csv", index_col=0)
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

k_means = KMeans(n_clusters=2).fit(df) # 2 merkezli model oluşturuyorum
kumeler = k_means.labels_ # gözlemlere atanan kümeler
type(df)
df = pd.DataFrame(df) # scale edilmiş numpy arraylerden DF oluşturuyorum

plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")
plt.show()

# merkezlerin isaretlenmesi
merkezler = k_means.cluster_centers_ # kümeleri (gözlemleri) seçiyorum

plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")

plt.scatter(merkezler[:, 0], # yukardaki plot'un üzerine merkezleri işaretleyerek yeni bir plot çiziyorum
            merkezler[:, 1],
            c="red",
            s=200,
            alpha=0.8)
plt.show()

################################
# * Optimum Küme Sayısının Belirlenmesi
################################
df = pd.read_csv("datasets/USArrests.csv", index_col=0)
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

kmeans = KMeans() # boş model oluşturuyorum
ssd = [] # hataları toplayacağım liste
K = range(1, 30) # 1'den 30'a kadar oluşacak küme grupları

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df) # her bir küme grubunu (k) deneyerek modeli fit ediyorum.
    ssd.append(kmeans.inertia_) # döngüden gelen k değeriyle elde ettiğim hatayı, hataları topladığım listeye atıyorum

plt.plot(K, ssd, "bx-") # x ekseninde Küme adetleri, y ekseninde hata değerleri
plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# Daha otomatik bir yol:
# yukarıda yaptığımız işlemi KElbowVisualizer ile daha otomatik olarak yapabiliriz
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_ # optimum küme sayısı

################################
# * Final Cluster'ların Oluşturulması
################################
df = pd.read_csv("datasets/USArrests.csv", index_col=0)
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
kumeler = kmeans.labels_

df = pd.read_csv("datasets/USArrests.csv", index_col=0)

pd.DataFrame({"Eyaletler": df.index, "Kumeler": kumeler})

df["cluster_no"] = kumeler

# df["cluster_no"] = df["cluster_no"] + 1 # küme gösterimini 0'dan değil 1'den başlatmak için ufak bir trick

df.head()

df.groupby("cluster_no").agg({"cluster_no": "count"})
df.groupby("cluster_no").agg(np.mean)

df[df["cluster_no"] == 5]

df[df["cluster_no"] == 6]
