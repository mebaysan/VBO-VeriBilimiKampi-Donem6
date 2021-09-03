
#########################
# ODEVLER
#########################

###############################################
# ÖDEV 1: Fonksiyonlara Özellik Eklemek.
###############################################

# Görev: cat_summary() fonksiyonuna 1 özellik ekleyiniz. Bu özellik argumanla biçimlendirilebilir olsun.
# Not: Var olan özelliği de argumandan kontrol edilebilir hale getirebilirsiniz.


# Fonksiyona arguman ile biçimlendirilebilen bir özellik eklemek ne demek?
# Örnek olarak aşağıdaki check_df fonksiyonuna argumanla biçimlendirilebilen 2 özellik eklenmiştir.
# Bu özelliler ile tail fonksiyonunun kaç gözlemi göstereceği ve quantile değerlerinin gösterilip gösterilmeyeceği
# fonksiyona özellik olarak girilmiştir ve bu özellikleri kullanıcı argumanlarla biçimlendirebilmektedir.

# ÖNCESİ
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# SONRASI
def check_df(dataframe, head=5, tail=5, quan=False):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    if quan:
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

import pandas as pd
df = pd.read_csv("datasets/titanic.csv")
check_df(df, head=3, tail=3)

###############################################
# ÖDEV 2: Docstring.
###############################################
# Aşağıdaki fonksiyona 4 bilgi (uygunsa) barındıran numpy tarzı docstring yazınız.
# (task, params, return, example)
# cat_summary()
