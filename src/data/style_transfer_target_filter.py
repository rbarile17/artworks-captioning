import json
import pandas as pd

from . import ARTGRAPH_PATH
from .datasets_loaders import datasets_loading_functions
from .. import load_params

from tqdm import tqdm

tqdm.pandas()

def main(params = None):
    if params is None:
        params = load_params()["style_transfer_target_filter"]

    artworks_df = pd.read_csv(ARTGRAPH_PATH / 'artgraph.csv')

    titles_in_artgraph = []
    for evaluation_dataset in params['evaluation_datasets']:
        # load evauation dataset using corresponding loading function
        dataset = pd.DataFrame(datasets_loading_functions[evaluation_dataset]())

        # remove dupicates from artworks df
        artworks_unique_titles_df = artworks_df.drop_duplicates(subset=['title'], keep=False).copy()

        # get all titles from evaluation dataset
        dataset_titles = dataset['title'].tolist()

        

        artworks_unique_titles_df['is_in_eval_dataset'] = artworks_unique_titles_df['title'] \
            .progress_map(lambda x: 
                [dataset_title for dataset_title in dataset_titles if dataset_title in x])
        
        print(artworks_unique_titles_df)
        break
        artworks_unique_titles_df = artworks_unique_titles_df[
            artworks_unique_titles_df['is_in_eval_dataset'] == True]
        
        print(len(artworks_unique_titles_df))
        artworks_unique_titles_df['pairs'] = list(
            zip(artworks_unique_titles_df['title'], artworks_unique_titles_df['file_name']))
        titles_in_artgraph += artworks_unique_titles_df['pairs'].tolist()

if __name__ == "__main__":
    main()