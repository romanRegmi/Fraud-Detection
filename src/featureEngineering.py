import logging
import joblib
import pandas as pd

from typing import Tuple
from datetime import datetime
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder

def write_report(text:str):
    f = open('../reports/featureEngineering.txt', 'a')
    f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ' ' + text + '\n')
    f.close()

def seperate_categorical(data_frame:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    col_categorical = data_frame.select_dtypes(include= ['object']).columns
    non_categorical = data_frame.drop(columns=col_categorical)
    categorical     = data_frame[col_categorical]
    return categorical, non_categorical
    
def label_encode(data_frame:pd.DataFrame) -> pd.DataFrame:
    d = defaultdict(LabelEncoder)
    encoded_df = data_frame.apply(lambda x: d[x.name].fit_transform(x))
    joblib.dump(d, '../serialized_files/joblib/label_encoder_dict.joblib')
    
    return encoded_df

if __name__=='__main__':
    
    logging.basicConfig(filename='../logs/featureEngineering.log',
                    level=logging.INFO,
					format='%(levelname)s:%(asctime)s:%(message)s',
					datefmt="%Y-%m-%d %H:%M:%S")

    FILE_PATH = '../data/dataProcessed.csv'

    logging.info('READING DATA')
    data = pd.read_csv(FILE_PATH)

    logging.info('SEPERATE CATEGORICAL AND NON-CATEGORICAL DATA')
    categorical, non_categorical = seperate_categorical(data)

    logging.info('ENCODING CATEGORICAL FEATURES AND SAVING THE ENCODING OBJECT AS A JOBLIB FILE')
    encoded_df = label_encode(categorical)

    logging.info('CONCAT NON-CATEGORICAL AND ENCODED DATA FRAME')
    result = pd.concat([encoded_df, non_categorical], axis=1)

    result.to_csv('../data/featureEngineered.csv', index=False)
    write_report('ENCODED DATA SAVED AS A CSV FILE')

    logging.info('EXIT FEATURE ENGINEERING')
