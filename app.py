import pickle
import joblib

import pandas as pd

from flask import Flask, request, render_template

random_forest = pickle.load(open('serialized_files/pickle/random_forest.pkl','rb'))
label_encoder = joblib.load('serialized_files/joblib/label_encoder_dict.joblib')

app = Flask(__name__)

@app.route('/' ,methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    step     = int(request.form['step'])
    customer = request.form['customer']
    age      = request.form['age']
    merchant = request.form['merchant']
    gender   = request.form['gender']
    category = request.form['category']
    amount   = float(request.form['amount'])

    dit = {'customer':["'{}'".format(customer)],'age':[age], 'gender':[gender], 
           'merchant':["'{}'".format(merchant)], 'category':[category],'step':[step], 'amount':[amount]}
    
    df = pd.DataFrame(data=dit)
    df.iloc[:, 0:5] = df.iloc[:, 0:5].apply(lambda x: label_encoder[x.name].transform(x))
    
    result = random_forest.predict([list(df.iloc[0])])[0]
    
    if result == 0:
        return render_template('index.html', prediction_text="~Fraud")
    else:
        return render_template('index.html', prediction_text="Fraud")

if __name__=='__main__':
    app.run(debug=True)