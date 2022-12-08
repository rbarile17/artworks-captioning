import json
import pandas as pd

from . import dataset_paths, ARTGRAPH_PATH
from ..data_utils.datasets_loaders import datasets_loading_functions
from ... import load_params

from tqdm import tqdm

tqdm.pandas()

def main(params = None):
    if params is None:
        params = load_params()["style_transfer_target_filter"]

    artworks_df = pd.read_csv(ARTGRAPH_PATH / 'artgraph.csv')

    train_filters = []
    for evaluation_dataset in params['evaluation_datasets']:
        # load evauation dataset using corresponding loading function
        dataset = pd.DataFrame(datasets_loading_functions[evaluation_dataset]())
        
        # switch to lowercase all titles in dataset
        dataset['lowercase_title'] = dataset['title'].str.lower()

        # get all lowercase titles in a list from artworks dataset
        artworks_titles = artworks_df['title'].str.lower().tolist()

        dataset['artworks_dataset_matches'] = dataset['lowercase_title'] \
            .progress_map(lambda x: [artwork_title for artwork_title in artworks_titles if artwork_title == x])
        
        dataset['artworks_dataset_matches_number'] = dataset['artworks_dataset_matches'].map(lambda x: len(x))

        # the artworks in this filter will not be used as targets for style transfer
        train_filter = dataset[dataset['artworks_dataset_matches_number'] == 1]

        """
        The artworks in this filter have generic titles such as: self-portait, crucifixion and so on.
        Several entries in artgraph have this kind of title (e.g. more than 700 self-portaits).
        We don't know which is the artgraph element that corresponds to the title in the evaluation dataset 
        so we have to remove all of them from the style transfer target dataset,
        (e.g. the semart dataset contains an artwork named crucifixion, 
        to avoid overlap we need to remove all crucifixions from artgraph)
        but this would lead to lose lots of potentially important artworks. 
        So we decided to keep them in artgraph but to filter them out in the evaluation phase.
        """
        eval_filter = dataset[dataset['artworks_dataset_matches_number'] > 1]

        train_filters.append(train_filter['title'])

        dataset_path = dataset_paths[evaluation_dataset]
        eval_filter['title'].to_csv(dataset_path / f'titles_in_artgraph.csv', index=False)

    train_filter = pd.concat(train_filters)
        
    # add artworks_df columns to train_filter
    train_filter = train_filter.to_frame().merge(artworks_df, on='title')
    train_filter = train_filter.drop_duplicates(subset=['file_name'])

    train_filter.file_name.to_csv(ARTGRAPH_PATH / 'filter.csv', index=False, header=False)
if __name__ == "__main__":
    main()