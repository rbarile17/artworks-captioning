import pandas as pd
from .. import SEMART_PATH

def main():
    semart = pd.concat([
        pd.read_csv(SEMART_PATH / 'semart_train.csv', delimiter='\t', encoding='Cp1252'),
        pd.read_csv(SEMART_PATH / 'semart_val.csv', delimiter='\t', encoding='Cp1252'),
        pd.read_csv(SEMART_PATH / 'semart_test.csv', delimiter='\t', encoding='Cp1252')
    ], axis=0)

    semart = semart.rename(columns={
        'IMAGE_FILE': 'file_name', 
        'DESCRIPTION': 'description',
        'TITLE': 'title',
        'AUTHOR': 'author',
        'DATE': 'date',
        'TECHNIQUE': 'technique',
        'TYPE': 'type',
        'SCHOOL': 'school',
        'TIMEFRAME': 'timeframe'
    })

    semart.to_csv(SEMART_PATH / 'semart.csv', index=False)

if __name__ == "__main__":
    main()