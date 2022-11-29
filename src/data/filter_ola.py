import os
import pandas as pd

from . import OLA_PATH, ARTGRAPH_PATH

def main(params = None):
    ola = pd.read_csv(OLA_PATH / 'ola.csv')
    ola = ola.rename(columns={'painting': 'file_name', 'utterance': 'description'})

    artgraph_images = os.listdir(ARTGRAPH_PATH / 'imagesf2/')

    ola = ola.loc[:, ['file_name', 'description']]

    # remove duplicates from ola keeping the first occurence
    ola = ola.drop_duplicates(subset=['file_name'], keep='first')

    # keep only entries in ola with an image in artgraph
    ola = ola.loc[ola['file_name'].isin(artgraph_images)]

    # save ola to a csv file
    ola.to_csv(OLA_PATH / 'ola_filtered.csv', index=False)

if __name__ == "__main__":
    main()