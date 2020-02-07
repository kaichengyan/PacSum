from typing import List

import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer

from models.importance import PacSumExtractorWithImportance


class WordImportanceModel(PacSumExtractorWithImportance):
    def __init__(self,
                 window_size: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size

    def _calculate_article_importance(self, article_idx: int, article: List[str]) -> List[float]:
        all_importances = []
        article = [RobertaTokenizer.clean_up_tokenization(s) for s in article]
        for idx in tqdm(range(len(article)), desc=f'Article {article_idx}'):
            all_importances.append(self._calculate_single_sentence_importance(idx, article))
        return all_importances

    def _calculate_single_sentence_importance(self, i: int, article: List[str]) -> float:
        tokenized_sentences = [self.tokenizer.encode(sent, add_prefix_space=True)
                               for sent in article]
        tokenized_article = list(np.concatenate(tokenized_sentences))
        tokenized_windows = self._parse_article_into_windows(tokenized_article)

    def _parse_article_into_windows(self, tokenized_article: List[int]) -> List[List[int]]:
        return [tokenized_article[i:i + self.window_size]
                for i in range(len(tokenized_article), step=self.window_size)]
