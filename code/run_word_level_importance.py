import argparse

import yaml

from data_iterator import Dataset
from models.word_importance_model import WordImportanceModel

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

    extractor = WordImportanceModel(extract_num=args.extract_num,
                                    num_pj_samples=args.num_pj_samples,
                                    pj_len=args.pj_len,
                                    window_size=args.window_size,
                                    # use_log_prob=args.use_log_prob,
                                    device=args.device)

    dataset = Dataset(args.test_data_file)
    dataset_iterator = dataset.iterate_once_doc_importance()
    extractor.extract_summary(dataset_iterator)
