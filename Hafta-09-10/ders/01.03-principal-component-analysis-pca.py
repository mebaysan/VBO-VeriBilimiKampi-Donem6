################################
# * Principal Component Analysis
################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

################################
# * Principal Component Analysis
################################


df = pd.read_csv("datasets/Hitters.csv")
df.head()
df.shape


num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]

df = df[num_cols]
df.dropna(inplace=True)
df.shape

df = StandardScaler().fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)


################################
# * Optimum Bileşen Sayısı
################################


pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()


################################
# * Final PCA'in Oluşturulması
################################

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)
pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)


pca_fit.shape
