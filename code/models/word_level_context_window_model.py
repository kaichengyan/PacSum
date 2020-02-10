import heapq
from typing import List, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer

from models.base_model import PacSumExtractorWithImportance


class WordLevelContextWindowModel(PacSumExtractorWithImportance):
    def __init__(self,
                 num_pj_samples: int,
                 pj_len: int,
                 window_size: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_pj_samples = num_pj_samples
        self.pj_len = pj_len
        self.window_size = window_size

    def _calculate_article_importance(self, article_idx: int, article: List[str]) -> List[float]:
        all_importances = []
        article = [RobertaTokenizer.clean_up_tokenization(s) for s in article]
        tokenized_sentences = [self.tokenizer.encode(sent, add_prefix_space=True)
                               for sent in article]
        tokenized_article = list(np.concatenate(tokenized_sentences))
        tokenized_windows = self._parse_article_into_windows(tokenized_article)
        word_imps = self._calculate_all_word_importance(article_idx,
                                                        tokenized_article,
                                                        tokenized_windows)
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
    Word importance code
    """

    def _calculate_all_word_importance(self,
                                       article_idx: int,
                                       article: List[int],
                                       windows: List[List[int]]) -> Dict[int, float]:
        word_iotas = dict()
        for token in tqdm(article, desc=f'Art. {article_idx}'):
            if token not in word_iotas:
                word_iotas[token] = \
                    self._calculate_single_word_importance(token, windows)
        highest_score_tokens = heapq.nlargest(10, word_iotas, key=word_iotas.get)
        lowest_score_tokens = heapq.nsmallest(10, word_iotas, key=word_iotas.get)
        print('Highest:', self.tokenizer.convert_ids_to_tokens(highest_score_tokens))
        print('Lowest: ', self.tokenizer.convert_ids_to_tokens(lowest_score_tokens))
        return word_iotas

    def _calculate_single_word_importance(self,
                                          word: int,
                                          windows: List[List[int]]) -> float:
        iota = 0.
        w = word
        for c in windows:
            if w in c:
                # sample phrases from w
                unmasked_batch, masked_batch, labels_batch = \
                    self._generate_batch(c, word)
                with torch.no_grad():
                    loss_unmasked, scores_unmasked = \
                        self.masked_lm(unmasked_batch, masked_lm_labels=labels_batch)
                    loss_masked, scores_masked = \
                        self.masked_lm(masked_batch, masked_lm_labels=labels_batch)
                iota += loss_masked - loss_unmasked
        return iota

    def _generate_batch(self, context_window: List[int], word: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = torch.tensor(context_window)
        unmasked_list, masked_list, labels_list = [], [], []
        for k in range(self.num_pj_samples):
            pj_left = np.random.randint(max(1, len(context_window) - self.pj_len))
            pj_mask_range = torch.arange(
                pj_left,
                min(len(context_window), pj_left + self.pj_len)
            ).long()
            mask_pj = torch.zeros_like(c).bool().index_fill_(0, pj_mask_range, True)
            mask_w = c.eq(word)

            unmasked = c.masked_fill(mask_pj, self.tokenizer.mask_token_id)
            masked = unmasked.masked_fill(mask_w, self.tokenizer.mask_token_id)
            labels = c.masked_fill(~mask_pj, -100)

            unmasked_list.append(unmasked)
            masked_list.append(masked)
            labels_list.append(labels)

        unmasked_batch = torch.stack(unmasked_list, dim=0)
        masked_batch = torch.stack(masked_list, dim=0)

        # each batch tensor should have dimensions [num_pj_samples x di_len]

        bos_copies = torch.full((self.num_pj_samples, 1), self.tokenizer.bos_token_id).long()
        eos_copies = torch.full((self.num_pj_samples, 1), self.tokenizer.eos_token_id).long()
        unmasked_batch = torch.cat((bos_copies, unmasked_batch, eos_copies), 1)
        masked_batch = torch.cat((bos_copies, masked_batch, eos_copies), 1)

        ignore_copies = torch.full_like(bos_copies, -100)
        labels_batch = torch.cat((ignore_copies, torch.stack(labels_list, dim=0), ignore_copies), 1)

        assert unmasked_batch.shape == masked_batch.shape == labels_batch.shape
        return unmasked_batch.to(self.device), \
               masked_batch.to(self.device), \
               labels_batch.to(self.device)

    def _parse_article_into_windows(self, tokenized_article: List[int]) -> List[List[int]]:
        return [tokenized_article[i:i + self.window_size]
                for i in range(0, len(tokenized_article), self.window_size)]
