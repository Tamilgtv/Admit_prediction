import numpy as np
from flask import Flask, render_template, request
import pickle


app = Flask(__name__)
model = pickle.load(open('modelpk.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['post'])
def predict():
    feature = [float(x) for x in request.form.values()]
    final_feature = [np.array(feature)]
    prediction = model.predict(final_feature)
    output = prediction
    return render_template('index.html', predict_text='Admit Result {}' .format(output))


@app.route('/hello/<name>')
def hello_name(name):
    return "Now you are data scientist %s!" %name
    


if __name__ == '__main__':
    app.run(debug=True)


