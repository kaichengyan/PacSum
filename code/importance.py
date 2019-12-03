from typing import List, Tuple, Iterator

import torch
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
        self.masked_lm: RobertaForMaskedLM = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained('roberta-base')


    def extract_summary(self, data_iterator: Iterator[Tuple[List[str], List[str]]]) -> None:
        summaries: List[List[str]] = []
        references: List[List[List[str]]] = []

        for article, abstract in data_iterator:
            if len(article) <= self.extract_num:
                summaries.append(article)
                references.append([abstract])
                continue

            # edge_scores = self._calculate_similarity_matrix(article)
            article_importance = self._calculate_all_sentence_importance(article)
            ids: List[int] = self._select_tops(article_importance)
            summary = list(map(lambda x: article[x], ids))

            summaries.append(summary)
            references.append([abstract])

        result = evaluate_rouge(summaries, references, remove_temp=True, rouge_args=[])

    def _select_tops(self, article_importance: List[float]) -> List[int]:
        id_importance_pairs: List[Tuple[int, float]] = []
        for i in range(len(article_importance)):
            id_importance_pairs.append((i, article_importance[i]))
        id_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        extracted = [item[0] for item in id_importance_pairs[:self.extract_num]]
        return extracted

    def _calculate_all_sentence_importance(self, article: List[str]) -> List[float]:
        return [self._calculate_sentence_importance(idx, article)
                for idx, sen in enumerate(article)]

    def _calculate_sentence_importance(self, i: int, article: List[str]) -> float:
        raise NotImplementedError


class PacSumExtractorWithImportanceV3(PacSumExtractorWithImportance):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _calculate_sentence_importance(self, i: int, article: List[str]) -> float:
        """
        Sample k phrases pi from si, sample m phrases pj from the window of si Di,
        iota3(si | D) = sum_pi sum_pj (log P(pj | D(si)) - log P(pj | D(si) - pi))
        :param i: The index of sentence si to calculate iota of
        :param article: The article that si is in
        :return: The importance of sentence si
        """
        si = article[i]
        s_importance = 0
        # TODO:
        return s_importance


class PacSumExtractorWithImportanceV2(PacSumExtractorWithImportance):
    def __init__(self,
                 num_sentence_samples: int = 10,
                 num_word_samples: int = 3,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_sentence_samples = num_sentence_samples
        self.num_word_samples = num_word_samples

    def _calculate_sentence_importance(self, i: int, article: List[str]) -> float:
        """
        Sample sentences in the window of si.
        For each sentence sk, sample q words from each sentence.
        iota2(si | D) = sum_k sum_j (log P(wj | sk' + si) - log P(wj | sk'))
        :param i: The index of sentence si to calculate iota of
        :param article: The article that si is in
        :return: The importance of sentence si
        """
        si = article[i]
        s_importance = 0
        # sample p sentences sk from article s.t. k != i
        sks = np.random.choice(article, self.num_sentence_samples, replace=False)
        for sk in sks:
            sentence_pairs, masked_lm_labels, loss_mask = self._generate_batch_for_si_and_sk(si, sk)
            loss = self.masked_lm(sentence_pairs, masked_lm_labels=masked_lm_labels)
            sentence_pairs_sk, masked_lm_labels_sk, loss_mask_sk = self._generate_batch_for_si_and_sk(si, sk)
            loss_sk = self.masked_lm(sentence_pairs_sk, masked_lm_labels=masked_lm_labels_sk)
            s_importance += loss - loss_sk
        return s_importance

    def _generate_batch_for_si_and_sk(self, si, sk):
        sk_encoded = self.tokenizer.encode(sk)
        sk_len = len(sk_encoded)

        # sample num_word_samples word indices in sk
        word_indices = np.random.choice(sk_len, self.num_word_samples)

        # sj_masked_copies: [num_word_samples * sk_len]
        sk_copies = torch.tensor([sk_encoded]).repeat(self.num_word_samples, 1).to(self.device)
        mask = torch.eye(sk_len).bool()[word_indices].to(self.device)
        # mask out labels
        sk_masked_copies = sk_copies.masked_fill(mask, self.tokenizer.mask_token_id)

        # si_copies: [num_word_samples * si_len]
        si_encoded = self.tokenizer.encode(si)
        si_len = len(si_encoded)
        si_copies = torch.tensor([si_encoded]).repeat(self.num_word_samples, 1).to(self.device)

        # bos/eos_copies:  [num_word_samples * 1]
        bos_copies = torch.zeros(self.num_word_samples, 1).to(self.device).fill_(self.tokenizer.bos_token_id).long()
        eos_copies = torch.zeros(self.num_word_samples, 1).to(self.device).fill_(self.tokenizer.eos_token_id).long()

        sentence_pairs = torch.cat((bos_copies, si_copies, eos_copies,
                                    eos_copies, sk_masked_copies, eos_copies),
                                   1).to(self.device)

        # masked_lm_labels = torch.cat((torch.zeros(sj_len, si_len + 3).fill_(-1).long(),
        #                               sj_masked_labels,
        #                               torch.zeros(sj_len, 1).fill_(-1).long())
        #                              , 1).to(self.device)

        loss_mask = torch.cat((torch.zeros(self.num_word_samples, si_len + 3).bool(),
                               mask,
                               torch.zeros(self.num_word_samples, 1).bool()),
                              1).to(self.device)

        masked_lm_labels = torch.zeros_like(loss_mask).to(self.device)
        masked_lm_labels.fill_(-1).long().masked_scatter(loss_mask, sk_copies)

        return sentence_pairs, masked_lm_labels, loss_mask

    def _generate_batch_for_sk(self, sk):
        sk_encoded = self.tokenizer.encode(sk)
        sk_len = len(sk_encoded)

        # sample num_word_samples word indices in sk
        word_indices = np.random.choice(sk_len, self.num_word_samples)

        # sj_masked_copies: [num_word_samples * sk_len]
        sk_copies = torch.tensor([sk_encoded]).repeat(self.num_word_samples, 1).to(self.device)
        mask = torch.eye(sk_len).bool()[word_indices].to(self.device)
        # mask out labels
        sk_masked_copies = sk_copies.masked_fill(mask, self.tokenizer.mask_token_id)

        # bos/eos_copies:  [num_word_samples * 1]
        bos_copies = torch.zeros(self.num_word_samples, 1).to(self.device).fill_(self.tokenizer.bos_token_id).long()
        eos_copies = torch.zeros(self.num_word_samples, 1).to(self.device).fill_(self.tokenizer.eos_token_id).long()

        sentence_pairs = torch.cat((bos_copies, sk_masked_copies, eos_copies),
                                   1).to(self.device)

        # masked_lm_labels = torch.cat((torch.zeros(sj_len, si_len + 3).fill_(-1).long(),
        #                               sj_masked_labels,
        #                               torch.zeros(sj_len, 1).fill_(-1).long())
        #                              , 1).to(self.device)

        loss_mask = torch.cat((torch.zeros(self.num_word_samples, 1).bool(),
                               mask,
                               torch.zeros(self.num_word_samples, 1).bool()),
                              1).to(self.device)

        masked_lm_labels = torch.zeros_like(loss_mask).to(self.device)
        masked_lm_labels.fill_(-1).long().masked_scatter(loss_mask, sk_copies)

        return sentence_pairs, masked_lm_labels, loss_mask


class PacSumExtractorWithImportanceV1(PacSumExtractorWithImportance):

    def __init__(self, num_word_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_word_samples = num_word_samples

    def _calculate_sentence_importance(self, i: int, article: List[str]) -> float:
        """
        Sample k words from D - si, for each word wj, compute its window Dj that contains si.
        iota1(si | D) = sum_j (log P(wk | Dj) - log P(wj | Dj - si))
        :param i: The index of sentence si to calculate iota of
        :param article: The article that si is in
        :return: The importance of sentence si
        """
        si = article[i]
        s_importance = 0
        # TODO:
        return s_importance

class PacSumExtractorWithImportanceV0(PacSumExtractorWithImportance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calculate_sentence_importance(self, i: int, article: List[str]) -> float:
        """
        For each other sentence sj in the article,
        for each word wk in sj,
        iota0(si | D) = sum_j sum_k (log P(wk | sj' + si))
        :param i: The index of sentence si to calculate iota of
        :param article: The article that si is in
        :return: The importance of sentence si
        """
        si = article[i]
        s_importance = 0
        for j in range(len(article)):
            if j != i:
                sj = article[j]
                # format sentences for masked LM
                sentence_pairs, masked_lm_labels, loss_mask = self._generate_batch(si, sj)
                # What is P_BERT(s_j | s_i)
                # Is the NLL loss just -sum(log P(w_l | s_i + s_j - w_l))?
                loss = self.masked_lm(sentence_pairs, masked_lm_labels=masked_lm_labels)[0]
                s_importance += loss
        return s_importance

    def _generate_batch(self, si: str, sj: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sj_masked_copies: [sj_len * sj_len]
        sj_encoded = self.tokenizer.encode(sj)
        sj_len = len(sj_encoded)
        sj_copies = torch.tensor([sj_encoded]).repeat(sj_len, 1).to(self.device)
        mask = torch.eye(sj_len, sj_len).bool().to(self.device)
        # mask out labels
        sj_masked_copies = sj_copies.masked_fill(mask, self.tokenizer.mask_token_id)

        # si_copies: [sj_len * si_len]
        si_encoded = self.tokenizer.encode(si)
        si_len = len(si_encoded)
        si_copies = torch.tensor([si_encoded]).repeat(sj_len, 1).to(self.device)

        # bos/eos_copies:  [sj_len * 1]
        bos_copies = torch.zeros(sj_len, 1).to(self.device).fill_(self.tokenizer.bos_token_id).long()
        eos_copies = torch.zeros(sj_len, 1).to(self.device).fill_(self.tokenizer.eos_token_id).long()

        sentence_pairs = torch.cat((bos_copies, si_copies, eos_copies,
                                    eos_copies, sj_masked_copies, eos_copies),
                                   1).to(self.device)

        # masked_lm_labels = torch.cat((torch.zeros(sj_len, si_len + 3).fill_(-1).long(),
        #                               sj_masked_labels,
        #                               torch.zeros(sj_len, 1).fill_(-1).long())
        #                              , 1).to(self.device)

        loss_mask = torch.cat((torch.zeros(sj_len, si_len + 3).bool(),
                               mask,
                               torch.zeros(sj_len, 1).bool()),
                              1).to(self.device)

        masked_lm_labels = torch.zeros_like(loss_mask).fill_(-1).long().masked_scatter(loss_mask, sj_copies)

        return sentence_pairs, masked_lm_labels, loss_mask
