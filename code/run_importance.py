import argparse

from data_iterator import Dataset
from importance import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['tune', 'test'], help='tune or test')
    parser.add_argument('--tune_data_file', type=str, help='data for tuning hyperparameters',
                        default='../data/NYT/nyt.validation.h5df')
    parser.add_argument('--test_data_file', type=str, help='data for testing',
                        default='../data/NYT/nyt.validation.h5df')
    parser.add_argument('--device', type=str, help='device to use', default='cuda')

    args = parser.parse_args()
    print(args)

    extractor = PacSumExtractorWithImportanceV3(3, 3, device=args.device)

    # tune
    if args.mode == 'tune':
        tune_dataset = Dataset(args.tune_data_file)
        tune_dataset_iterator = tune_dataset.iterate_once_doc_importance()
        # extractor.tune_hparams(tune_dataset_iterator)

    # test
    test_dataset = Dataset(args.test_data_file)
    test_dataset_iterator = test_dataset.iterate_once_doc_importance()
    extractor.extract_summary(test_dataset_iterator)
