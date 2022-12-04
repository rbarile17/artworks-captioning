import pandas as pd

from . import OLA_PATH, ARTGRAPH_PATH

def main():
    ola = pd.read_csv(OLA_PATH / 'ola.csv')
    ola = ola.rename(columns={'painting': 'file_name', 'utterance': 'description'})

    # remove duplicates from ola keeping the first occurence
    ola = ola.drop_duplicates(subset=['file_name'], keep='first')
    
    # add a column with .jpg extension to file name
    ola['file_name'] = ola['file_name'] + '.jpg'
    
    # join ola with artgraph
    ola = ola.merge(pd.read_csv(ARTGRAPH_PATH / 'artgraph.csv'), on='file_name')

    # remove artgraph columns except for title
    ola = ola.loc[:, ['file_name', 'description', 'title']]

    # save ola to a csv file
    ola.to_csv(OLA_PATH / 'ola_filtered.csv', index=False)

if __name__ == "__main__":
    main()