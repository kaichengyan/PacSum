import itertools
from typing import Dict, Any, List

import yaml

from data_iterator import Dataset
from importance import PacSumExtractorWithImportanceV3


def product_dict(d: Dict[Any, List[Any]]):
    for x in itertools.product(*d.values()):
        yield dict(zip(d.keys(), x))


if __name__ == '__main__':
    with open('experiment_config.yaml', 'r') as f:
        config_file = yaml.load(f, Loader=yaml.SafeLoader)

    configs = product_dict(config_file)

    dataset = Dataset('../data/NYT/nyt.validation.h5df')

    for c in configs:
        print(c)
        dataset_iterator = dataset.iterate_once_doc_importance()
        extractor = PacSumExtractorWithImportanceV3(**c)
        result = extractor.extract_summary(dataset_iterator)
