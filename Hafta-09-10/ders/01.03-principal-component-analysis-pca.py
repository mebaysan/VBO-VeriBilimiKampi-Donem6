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
def get_scaled_df():
    df = pd.read_csv("datasets/Hitters.csv")
    num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col] # Salary haricindeki numeric değişkenleri seçiyorum.
    df = df[num_cols]
    df.dropna(inplace=True)
    df = StandardScaler().fit_transform(df) # numerik değişkenleri scale edip, değişkene atıyoruz (numpy arraylerden oluşan)
    return df


df = get_scaled_df()
pca = PCA()
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_ # açıklanan varyans oranı

np.cumsum(pca.explained_variance_ratio_) # kümülatif açıklanan varyans oranı
"""
Her bir eleman bir bileşeni temsil eder. 5. eleman 5 bileşen ile veri setindeki varyansın ne kadarının açıklandığıdır.
Örnek olarak: 5. bileşen ile veri setindeki varyansın %91'i açıklanıyor.

>>> array([0.46037855, 0.72077704, 0.82416565, 0.87785586, 0.91993427,
       0.94957018, 0.96527809, 0.9766709 , 0.9845032 , 0.9903799 ,
       0.99412755, 0.99722139, 0.99877819, 0.99963722, 0.99992409,
       1.        ])
"""


################################
# * Optimum Bileşen Sayısı
################################
# elbow yöntemi ile boyut sayısına karar verilir.

df = get_scaled_df()
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()


################################
# * Final PCA'in Oluşturulması
################################
df = get_scaled_df()
pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)
pca.explained_variance_ratio_ # hangi bileşen yüzde kaçını açıklıyor
np.cumsum(pca.explained_variance_ratio_)


pca_fit.shape # 322 gözlem birimi 3 bişey var. Bu 3 bişey bileşenlerdir. Yani değişken, componenttir.
