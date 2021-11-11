import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)  # flask uygulamasını initialize eder.
model = pickle.load(open('regression_model.pkl', 'rb'))  # kayitli modeli cagirma


# flask root (kök dizin) için ana sayfanın tanımlanması
@app.route('/')
def home():
    return render_template('template.html')  # anasayfa icin hazirlanmis olan template belirtilir.



# template.html içindeki kullanıcı etkileşimleri:
@app.route('/', methods=['POST'])
def predict():
    # Arayüzdeki text'lere girilen değerleri alıp hesaplama sonrası tekrar arayüze göndereceğiz.
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    predicted_y = int(np.round(prediction, 2))
    # html çıktısına geri gondermek
    return render_template('template.html', prediction_text='Predicted Sales: {}'.format(predicted_y))


if __name__ == "__main__":
    app.run()
