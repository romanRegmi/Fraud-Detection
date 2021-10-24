import pickle
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import mlflow.sklearn
from mlflow.tracking import MlflowClient

import logging

def print_auto_logged_info(r) -> None:
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

def ml_flow(X_train, X_test, y_train, y_test, classifier, model_name:str='ml-model') -> None:
    mlflow.autolog()
    model = classifier
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test, y_pred))

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    pickle.dump(model, open('../serialized_files/pickle/{}.pkl'.format(model_name),'wb'))

def balance(X_train, X_test, y_train, y_test, imbalanced:bool=False):
    
    """
    - balance imbalanced dataset
    """
    if imbalanced:
        smt = SMOTE(random_state=42)
        X_train, y_train = smt.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

if __name__=='__main__':
    
    logging.basicConfig(filename='../logs/modelCreation.log',
                    level=logging.INFO,
					format='%(levelname)s:%(asctime)s:%(message)s',
					datefmt="%Y-%m-%d %H:%M:%S")
    
    rf_clf = RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42,
                                verbose=1,class_weight="balanced")

    nb_clf = MultinomialNB()

    models = {'random_forest':rf_clf, 'naive_bayes':nb_clf}

    logging.info('LOAD FILE')
    FILE_PATH = '../data/featuredEngineered.csv'
    data = pd.read_csv(FILE_PATH)

    X = data.drop(['fraud'],axis=1)
    y = data['fraud']

    logging.info('TRAIN TEST SPLIT')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, 
                                                        shuffle=True, stratify=y)
    
    logging.info('BALANCE THE IMBALANCED DATA SET')
    X_train, X_test, y_train, y_test = balance(X_train, X_test, y_train, y_test, imbalanced=True)

    logging.info('TRAIN MODELS USING RANDOMFOREST CLASSIFIER AND NAIVE BAYES CLASSIFIER')
    for key, value in models.items():
        ml_flow(X_train, X_test, y_train, y_test, value, model_name=key)

    
    logging.info('EXIT MODEL CREATION')