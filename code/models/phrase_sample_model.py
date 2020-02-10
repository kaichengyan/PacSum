from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer

from models.importance import PacSumExtractorWithImportance
from utils import masked_lm_sum_prob


class PhraseSampleImportanceModel(PacSumExtractorWithImportance):
    def __init__(self,
                 num_pi_samples: int,
                 num_pj_samples: int,
                 pi_len: int = 7,
                 pj_len: int = 7,
                 window_size: int = 256,
                 use_log_prob: bool = False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window_size: int = window_size
        self.num_pi_samples: int = num_pi_samples
        self.num_pj_samples: int = num_pj_samples
        # BERT requires pi_length + pj_length < 0.15 * window_size
        self.pi_len: int = pi_len
        self.pj_len: int = pj_len
        self.use_log_prob: bool = use_log_prob

    def _calculate_article_importance(self, article_idx: int, article: List[str]) -> List[float]:
        all_importances = []
        article = [RobertaTokenizer.clean_up_tokenization(s) for s in article]

        tokenized_sentences = [self.tokenizer.encode(sent, add_prefix_space=True) for sent in article]
        tokenized_article: List[int] = list(np.concatenate(tokenized_sentences))

        for idx in tqdm(range(len(article)), desc=f'Article {article_idx}'):
            all_importances.append(self._calculate_single_sentence_importance(idx,
                                                                              tokenized_sentences,
                                                                              tokenized_article))
        return all_importances

    def _calculate_single_sentence_importance(self,
                                              i: int,
                                              tokenized_sentences: List[List[int]],
                                              tokenized_article: List[int]) -> float:
        """
        Sample k phrases pi from si, sample m phrases pj from the window of si Di,
        iota3(si | D) = sum_pi sum_pj (log P(pj | D(si)) - log P(pj | D(si) - pi))
        :param i: The index of sentence si to calculate iota of
        :param article: The article that si is in
        :return: The importance of sentence si
        """
        si_tokenized = tokenized_sentences[i][:]
        si_len = len(si_tokenized)
        # relative to article
        si_left = sum(len(sent) for sent in tokenized_sentences[:i])
        si_right = si_left + si_len

        assert si_tokenized == tokenized_sentences[i] == tokenized_article[si_left:si_left + si_len]

        # want window_size window around si
        half_size = max(0, (self.window_size - si_len)) // 2
        # relative to article
        di_left = max(0, si_left - half_size)
        di_right = min(si_right + half_size, len(tokenized_article))
        di = tokenized_article[di_left:di_right]

        s_importance = 0
        for ii in range(self.num_pi_samples):
            assert si_tokenized == tokenized_sentences[i] == tokenized_article[si_left:si_left + si_len]
            # local to si
            pi_left_local = np.random.randint(max(1, si_len - self.pi_len + 1))
            # local to di
            pi_left = pi_left_local + si_left - di_left
            unmasked_batch, masked_batch, labels_batch = self._generate_batch(di, pi_left)
            with torch.no_grad():
                loss_unmasked, scores_unmasked = \
                    self.masked_lm(unmasked_batch, masked_lm_labels=labels_batch)
                loss_masked, scores_masked = \
                    self.masked_lm(masked_batch, masked_lm_labels=labels_batch)
                if self.use_log_prob:
                    # use native nll loss value
                    s_importance += (loss_masked.item() - loss_unmasked.item())
                else:
                    # calculate sum of probs manually
                    sum_prob_masked = masked_lm_sum_prob(scores_masked, labels_batch)
                    sum_prob_unmasked = masked_lm_sum_prob(scores_unmasked, labels_batch)
                    s_importance += (sum_prob_masked - sum_prob_unmasked)
        return s_importance

    def _generate_batch(self, di: List[int], pi_left: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unmasked_list, masked_list, labels_list = [], [], []
        di_len = len(di)
        di = torch.tensor(di)
        pi_mask_range = torch.arange(pi_left, min(di_len, pi_left + self.pi_len)).long()
        for j in range(self.num_pj_samples):
            # possible range of pj_left: [0, pi_left - pj_len] U [pi_right, di_len - pj_len]
            pj_left_range = list(np.arange(pi_left - self.pj_len + 1)) \
                            + list(np.arange(pi_left + self.pi_len, di_len - self.pj_len + 1))
            pj_left = np.random.choice(pj_left_range)
            pj_mask_range = torch.arange(pj_left, pj_left + self.pj_len).long()

            mask_pj = torch.zeros_like(di).bool().index_fill_(0, pj_mask_range, True)
            mask_pi = mask_pj.index_fill(0, pi_mask_range, True)
            # mask out corresponding values in di
            # mask out pj only, by filling mask_token_id's in mask_pj locations
            unmasked = di.masked_fill(mask_pj, self.tokenizer.mask_token_id)
            # also mask out pi, by filling mask_token_id's in mask_pi locations
            masked = di.masked_fill(mask_pi, self.tokenizer.mask_token_id)
            unmasked_list.append(unmasked)
            masked_list.append(masked)

            labels = di.masked_fill(~mask_pj, -100)
            # labels = torch.full_like(di, -100).long().masked_scatter_(mask_pj, di)
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
