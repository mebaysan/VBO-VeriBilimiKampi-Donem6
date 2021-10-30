################################
# * Hierarchical Clustering
################################

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

################################
# * Hierarchical Clustering
################################


df = pd.read_csv("datasets/USArrests.csv", index_col=0)

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

hc_complete = linkage(df, "complete")
hc_average = linkage(df, "average")

plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_complete,
           leaf_font_size=10)
plt.show()

################################
# * Kume Sayısını Belirlemek
################################

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average,
                  truncate_mode="lastp",
                  p=10,
                  show_contracted=True,
                  leaf_font_size=10)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.show()


################################
# * Final Modeli Oluşturmak
################################


cluster = AgglomerativeClustering(n_clusters=5) # dendogram yönteminden görsel olarak nerden böleceğime karar verdim.
cluster.fit_predict(df)

df = pd.read_csv("datasets/USArrests.csv", index_col=0)
df["cluster_no"] = cluster.fit_predict(df)


