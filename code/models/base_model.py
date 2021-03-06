from typing import List, Tuple, Iterator

import numpy as np
from transformers import RobertaForMaskedLM, RobertaTokenizer

from utils import evaluate_rouge


class PacSumExtractorWithImportance:
    def __init__(self,
                 extract_num: int = 3,
                 device: str = 'cuda') -> None:
        super().__init__()
        self.extract_num: int = extract_num
        self.device: str = device
        self.masked_lm: RobertaForMaskedLM = RobertaForMaskedLM.from_pretrained('distilroberta-base').to(device)
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

    def extract_summary(self, data_iterator: Iterator[Tuple[List[str], List[str]]], print_summaries=False) -> None:
        summaries: List[List[str]] = []
        references: List[List[List[str]]] = []

        # randomly sample 100 articles to evaluate on val set
        val_set_size = 5531
        eval_subset_size = 10
        val_indices = set(np.random.choice(np.arange(val_set_size),
                                           eval_subset_size,
                                           replace=False))

        for idx, (article, abstract) in enumerate(data_iterator):
            # if idx not in val_indices:
            #     continue
            # if idx >= 3:
            #     break
            if len(article) <= self.extract_num:
                summaries.append(article)
                references.append([abstract])
                continue

            # edge_scores = self._calculate_similarity_matrix(article)
            article_importance = self._calculate_article_importance(idx, article)
            ids: List[int] = self._select_tops(article_importance)
            summary = list(map(lambda x: article[x], ids))
            if print_summaries:
                print(summary, abstract)

            summaries.append(summary)
            references.append([abstract])

        result = evaluate_rouge(summaries, references, remove_temp=True, rouge_args=[])
        return result

    def _select_tops(self, article_importance: List[float]) -> List[int]:
        id_importance_pairs: List[Tuple[int, float]] = []
        for i in range(len(article_importance)):
            id_importance_pairs.append((i, article_importance[i]))
        id_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        extracted = [item[0] for item in id_importance_pairs[:self.extract_num]]
        return extracted

    def _calculate_article_importance(self, i: int, article: List[str]) -> List[float]:
        raise NotImplementedError

#
# class PacSumExtractorWithImportanceV2(PacSumExtractorWithImportance):
#     def __init__(self,
#                  sentence_sample_window_size: int = 3,
#                  num_sentence_samples: int = 3,
#                  num_word_samples: int = 3,
#                  *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.sentence_sample_window_size = sentence_sample_window_size
#         self.num_sentence_samples = num_sentence_samples
#         self.num_word_samples = num_word_samples
#
#     def _calculate_article_importance(self, article_idx: int, article: List[str]) -> List[float]:
#         all_importances = []
#         article = [RobertaTokenizer.clean_up_tokenization(s) for s in article]
#         for idx in tqdm(range(len(article)), desc=f'Article {article_idx}'):
#             all_importances.append(self._calculate_single_sentence_importance(idx, article))
#         return all_importances
#
#     def _calculate_single_sentence_importance(self, i: int, article: List[str]) -> float:
#         """
#         Sample sentences in the window of si.
#         For each sentence sk, sample q words from each sentence.
#         iota2(si | D) = sum_k sum_j (log P(wj | sk' + si) - log P(wj | sk'))
#         :param i: The index of sentence si to calculate iota of
#         :param article: The article that si is in
#         :return: The importance of sentence si
#         """
#         si = article[i]
#         s_importance = 0
#         # sample p sentences sk from article s.t. k != i
#         sks = np.random.choice(article, self.num_sentence_samples, replace=False)
#         for sk in sks:
#             with torch.no_grad():
#                 sentence_pairs, masked_lm_labels, loss_mask = self._generate_batch_for_si_and_sk(si, sk)
#                 loss = self.masked_lm(sentence_pairs, masked_lm_labels=masked_lm_labels)[0].cpu()
#                 sentence_pairs_sk, masked_lm_labels_sk, loss_mask_sk = self._generate_batch_for_sk(sk)
#                 loss_sk = self.masked_lm(sentence_pairs_sk, masked_lm_labels=masked_lm_labels_sk)[0].cpu()
#             s_importance += loss - loss_sk
#         return s_importance
#
#     def _generate_batch_for_si_and_sk(self, si, sk):
#         sk_encoded = self.tokenizer.encode(sk)
#         sk_len = len(sk_encoded)
#
#         # sample num_word_samples word indices in sk
#         word_indices = np.random.choice(sk_len, self.num_word_samples)
#
#         # sj_masked_copies: [num_word_samples * sk_len]
#         sk_copies = torch.tensor([sk_encoded]).repeat(self.num_word_samples, 1)
#         mask = torch.eye(sk_len).bool()[word_indices]
#         # mask out labels
#         sk_masked_copies = sk_copies.masked_fill(mask, self.tokenizer.mask_token_id)
#
#         # si_copies: [num_word_samples * si_len]
#         si_encoded = self.tokenizer.encode(si)
#         si_len = len(si_encoded)
#         si_copies = torch.tensor([si_encoded]).repeat(self.num_word_samples, 1)
#
#         # bos/eos_copies:  [num_word_samples * 1]
#         bos_copies = torch.zeros(self.num_word_samples, 1).fill_(self.tokenizer.bos_token_id).long()
#         eos_copies = torch.zeros(self.num_word_samples, 1).fill_(self.tokenizer.eos_token_id).long()
#
#         sentence_pairs = torch.cat((bos_copies, si_copies, eos_copies,
#                                     eos_copies, sk_masked_copies, eos_copies), 1)
#
#         loss_mask = torch.cat((torch.zeros(self.num_word_samples, si_len + 3).bool(),
#                                mask,
#                                torch.zeros(self.num_word_samples, 1).bool()), 1)
#
#         masked_lm_labels = torch.zeros_like(loss_mask)
#         masked_lm_labels.fill_(-1).long().masked_scatter_(loss_mask, sk_copies)
#
#         return sentence_pairs.to(self.device), masked_lm_labels.to(self.device), loss_mask.to(self.device)
#
#     def _generate_batch_for_sk(self, sk):
#         sk_encoded = self.tokenizer.encode(sk)
#         sk_len = len(sk_encoded)
#
#         # sample num_word_samples word indices in sk
#         word_indices = np.random.choice(sk_len, self.num_word_samples)
#
#         # sj_masked_copies: [num_word_samples * sk_len]
#         sk_copies = torch.tensor([sk_encoded]).repeat(self.num_word_samples, 1)
#         mask = torch.eye(sk_len).bool()[word_indices]
#         # mask out labels
#         sk_masked_copies = sk_copies.masked_fill(mask, self.tokenizer.mask_token_id)
#
#         # bos/eos_copies:  [num_word_samples * 1]
#         bos_copies = torch.zeros(self.num_word_samples, 1).fill_(self.tokenizer.bos_token_id).long()
#         eos_copies = torch.zeros(self.num_word_samples, 1).fill_(self.tokenizer.eos_token_id).long()
#
#         sentence_pairs = torch.cat((bos_copies, sk_masked_copies, eos_copies), 1)
#
#         loss_mask = torch.cat((torch.zeros(self.num_word_samples, 1).bool(),
#                                mask,
#                                torch.zeros(self.num_word_samples, 1).bool()), 1)
#
#         masked_lm_labels = torch.zeros_like(loss_mask)
#         masked_lm_labels.fill_(-1).long().masked_scatter(loss_mask, sk_copies)
#
#         return sentence_pairs.to(self.device), masked_lm_labels.to(self.device), loss_mask.to(self.device)
#
#
# class PacSumExtractorWithImportanceV1(PacSumExtractorWithImportance):
#
#     def __init__(self, num_word_samples: int = 5, window_size: int = 3, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_word_samples = num_word_samples
#         self.window_size = window_size
#
#     def _calculate_single_sentence_importance(self, i: int, article: List[str]) -> float:
#         """
#         Sample k words from D - si, for each word wj, compute its window Dj that contains si.
#         iota1(si | D) = sum_j (log P(wk | Dj) - log P(wj | Dj - si))
#         :param i: The index of sentence si to calculate iota of
#         :param article: The article that si is in
#         :return: The importance of sentence si
#         """
#         w = self.window_size
#         si = article[i]
#         window_min, window_max = max(0, i - w), min(len(article) - 1, i + w)
#         window = article[window_min:window_max + 1]
#
#         s_importance = 0
#         # TODO:
#         return s_importance
