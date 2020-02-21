import argparse

import yaml

from data_iterator import Dataset
from models.word_level_tf_idf_model import WordLevelTfIdfModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['tune', 'test'], help='tune or test')
    parser.add_argument('--tune_data_file', type=str, help='data for tuning hyperparameters',
                        default='../data/NYT/nyt.validation.h5df')
    parser.add_argument('--test_data_file', type=str, help='data for testing',
                        default='../data/NYT/nyt.validation.h5df')
    parser.add_argument('-c', '--config', type=str, help='yaml config file')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            configs = yaml.load(f, Loader=yaml.SafeLoader)
            args.__dict__.update(configs)

    print(args)

    extractor = WordLevelTfIdfModel(extract_num=args.extract_num, device=args.device)

    dataset = Dataset(args.test_data_file)
    print("Calculating idf scores...")
    extractor.calculate_idf_scores(dataset.iterate_once_doc_importance())
    dataset_iterator = dataset.iterate_once_doc_importance()
    extractor.extract_summary(dataset_iterator)
