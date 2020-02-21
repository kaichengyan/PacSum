import collections
import heapq
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer

from models.base_model import PacSumExtractorWithImportance


class WordLevelTfIdfModel(PacSumExtractorWithImportance):
    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idf: Dict[int, float] = dict()

    def calculate_idf_scores(self, data_iterator):
        D = 0
        counter = collections.Counter()
        for article, _ in data_iterator:
            D += 1
            sents = [RobertaTokenizer.clean_up_tokenization(s) for s in article]
            sents = [self.tokenizer.encode(s, add_prefix_space=True)
                     for s in sents]
            tokens = set(np.concatenate(sents))
            for t in tokens:
                counter[t] += 1
        self.idf = {k: np.log(D / v) for k, v in counter.items()}
        return counter

    def _calculate_article_importance(self, article_idx: int, article: List[str]) -> List[float]:
        all_importances = []
        article = [RobertaTokenizer.clean_up_tokenization(s) for s in article]
        tokenized_sentences = [self.tokenizer.encode(sent, add_prefix_space=True)
                               for sent in article]
        tokenized_article = list(np.concatenate(tokenized_sentences))
        word_imps = self._calculate_all_word_tf_idf(article_idx, tokenized_article)
        for idx in range(len(article)):
            all_importances.append(self._calculate_single_sentence_importance(tokenized_sentences[idx],
                                                                              word_imps))
        return all_importances

    def _calculate_single_sentence_importance(self,
                                              sentence: List[int],
                                              word_importance: Dict[int, float]) -> float:
        if not sentence:
            return 0.
        imp_scores = [word_importance[token] for token in sentence]
        return sum(imp_scores) / len(imp_scores)

    """
    Word tf-idf code
    """

    def _calculate_all_word_tf_idf(self,
                                   article_idx: int,
                                   article: List[int]) -> Dict[int, float]:
        word_tf_idf = dict()
        counter = collections.Counter(article)
        for token in set(article):
            tf = 0.5 + 0.5 * counter[token] / max(counter)
            word_tf_idf[token] = tf * self.idf[token]
        if article_idx % 500 == 0:
            print(f'Article {article_idx}:')
            highest_score_tokens = heapq.nlargest(10, word_tf_idf, key=word_tf_idf.get)
            lowest_score_tokens = heapq.nsmallest(10, word_tf_idf, key=word_tf_idf.get)
            print('Highest:')
            for tok in highest_score_tokens:
                print(self.tokenizer.convert_ids_to_tokens(tok.item()), word_tf_idf[tok])
            print('Lowest:')
            for tok in lowest_score_tokens:
                print(self.tokenizer.convert_ids_to_tokens(tok.item()), word_tf_idf[tok])
        return word_tf_idf
