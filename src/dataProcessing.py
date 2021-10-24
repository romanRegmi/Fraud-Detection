import logging
import pandas as pd

from datetime import datetime

FILE_PATH = '../data/data.csv'

logging.basicConfig(filename='../logs/dataProcessing.log',
                    level=logging.INFO,
					format='%(levelname)s:%(asctime)s:%(message)s',
					datefmt="%Y-%m-%d %H:%M:%S")

def write_report(text:str) -> None:

    f = open('../reports/dataProcessing.txt', 'a')
    f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ' ' + text + '\n')
    f.close()

logging.info('READING DATA')
data = pd.read_csv(FILE_PATH)
print(data.shape)

logging.info('FILTERING ROWS')
new_data = data[(data['category'] != "'es_contents'") & (data['category'] != "'es_food'") & (data['category'] != "'es_transportation'")]
write_report('CATEGORIES : [es_contents, es_food, es_transportation] REMOVED FROM DATA')
print(new_data.shape)

logging.info('FILTERING COLUMNS')
new_data.drop(columns=['zipcodeOri', 'zipMerchant'], inplace=True)
write_report('FEATURES : [zipcodeOri, zipMerchant] REMOVED FROM DATA')
print(new_data.shape)

logging.info('CREATING A NEW CSV FILE')
new_data.to_csv('../data/dataProcessed.csv', index=False)
write_report('NEW CSV FILE CREATED')

logging.info('EXIT DATA PROCESSING')


