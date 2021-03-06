# π°π° Fraud Detection
<img src='images/fraud_detection.png' width=500>

## π‘ Problem Statement
Fraud detection is a set of activities undertaken to prevent money or property from being obtained through false pretenses. Fraud detection is applied to many industries such as banking or insurance.

## π­ Scope of the project
Over the past several years, we have seen an increase in digitization. More and more transactions are being made through the internet. With this, we have also seen an increase in the fraudulent activities happening in various domains. These are detrimental to the ecosystem of online transactions. Therefore, it is vital that a system be developed such that such fradulent transactions can be predicted, monitored and necessary actions be taken. 

## π Data Description
The data and its overview is available [here](https://www.kaggle.com/ealaxi/banksim1).

A little more about the features present in the data is given in the **about.txt** file of this repository.

It is important to note that the data is **Synthetic**(fake) as it was generated using a simulator. Therefore, it may not be very useful in making inferences about similar real world scenarios.   

## β½ Project Tree Structure
```
βββ images
|   βββ fraud_detection.png
|   βββ interface.png
|   βββ tools.png
|
βββ logs
β   βββ dataAnalysis.log
|   βββ dataProcessing.log
|   βββ featureEngineering.log
|   βββ modelCreation.log
|
βββ reports
β   βββ analyticsReport.txt
β   βββ dataProcessing.txt
β   βββ featureEngineering.txt
β   
βββ serialized_files
β   βββ joblib
β        βββ label_encoder_dict.joblib
β   βββ pickle
β       βββ naive_bayes.pkl
β       βββ random_forest.pkl
βββ src
β   βββ mlruns
β   βββ Notebook
|   |   βββdataAnalysis.ipynb
β   βββ dataPreprocessing.py
β   βββ featureEngineering.py
β   βββ modelCreation.py
|
|ββ static/css
|   βββstyle.css
|
βββ templates
β   βββ index.html
|
|ββ about.txt
βββ Procfile
βββ app.py
βββ requirements.txt
βββ README.md
```
## π  Tools used
<img src='images/tools.png' width=500>

1. **Visual Studio Code** is used as an IDE.
2. For visualization of the plots, **matplotlib** and **seaborn** are used.
3. For model training **scikit-learn** is used.
4. For model tracking **mlflow** is used.
5. Front end development is done using **HTML/CSS**.
6. **Flask** framework is used for backend development.
7. **GitHub** is used as a version control system.
8. **Heroku** Cloud Platform is used for deployment of the model.


## π¨π»βπ» Interface
<img src='images/interface.png' width=500>

In the application, evey field must be given an appropriate input. For instance, the **MERCHANT** and **CUSTOMER** input must be from the original dataset(They are categorical). New data will throw an error for it will not have the appropriate encoding. 

The web app link is given [here](https://bank-sim-api.herokuapp.com/).

## Contributors
Roman Regmi
