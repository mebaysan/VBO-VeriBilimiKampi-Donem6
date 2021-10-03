import pandas as pd
from helpers import eda, data_prep
from sklearn.preprocessing import RobustScaler


def titanic_data_prep():
    # veri setini okuyorum
    df = pd.read_csv('Datasets/titanic.csv')
    
    # kolonları küçük harfe çeviriyorum
    df.columns = [col.lower() for col in df.columns]

    # kategorik, numerik kolonları seçiyorum
    cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)
    
    # numerik değişkenlerde outlier değerleri baskılıyorum
    for col in num_cols:
        data_prep.replace_with_thresholds(df, col)

    # age değişkenindeki NaN değerleri cinsiyet kırılımında medyan ile dolduruyorum
    df["age"] = df["age"].fillna(df.groupby("sex")["age"].transform("median"))

    #   tipi Object olan ve tekil değeri 10 veya daha küçük olan değişkenlerde boş değer varsa mod ile dolduruyorum
    df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    # bu değişkenler modeli etkilemeyeceği için çıkarıyorum
    df.drop(['cabin','name','ticket'],axis=1,inplace=True)
    
    # Hangi değişkende kaç adet eksik değer var, oranları neler bunları gözlemliyorum
    # data_prep.missing_values_table(df)

    # tekil değer adedi 2 olan ve değişken tipi int veya float olmayan binary değişkenleri seçiyorum
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

    # binary değişkenleri encode ediyorum
    for col in binary_cols:
        df = data_prep.label_encoder(df, col)
    
    # değişkenlerdeki sınıflar içerisinde RARE durumunu gözlemliyorum
    # data_prep.rare_analyser(df, "SURVIVED", cat_cols)

    # RARE değişkene sahip değişkenleri encode ediyorum
    df = data_prep.rare_encoder(df, cat_cols, 0.01)

    # son RARE durumunu gözlemliyorum
    # data_prep.rare_analyser(df, "SURVIVED", cat_cols)

    # one-hot encoding yapabileceğim sınıfları seçiyorum
    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

    # one-hot encoding uyguluyorum
    df = data_prep.one_hot_encoder(df, ohe_cols)


    # veri setinin son hali üzerinden değişkenleri seçiyorum. Encode ettiğimizden dolayı bir çoğu numerik olarak gelecek
    cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)

    # passengerid değişkeni analizi ve modeli etkilemeyeceğinden dolayı çıkarıyorum
    num_cols = [col for col in num_cols if "passengerid" not in col]

    # 2 adet tekil değeri olan (yani encode edilmiş veya encoding sonucu elde edilmiş değişkenler) ve frekans orano 0.01'den küçük olan değişkenleri modele/analize eklememeyi tercih edebiliriz
    # useless_cols = [col for col in df.columns if df[col].nunique() == 2 and (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
    # df.drop(useless_cols,inplace=True)
    
    # numerik değişkenleri scale ediyorum ki hepsini aynı ölçekte ölçümleyebilelim ve bazı değişkenler diğerlerini baskılamasın
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # veri seti son hali
    df.head()
    
    # pickle export
    df.to_pickle('prepared_titanic.pickle')
