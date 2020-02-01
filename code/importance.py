from typing import List, Tuple, Iterator

import itertools
import torch
import numpy as np
from tqdm import tqdm
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

    def extract_summary(self, data_iterator: Iterator[Tuple[List[str], List[str]]]) -> None:
        summaries: List[List[str]] = []
        references: List[List[List[str]]] = []

        for idx, (article, abstract) in enumerate(data_iterator):
            if len(article) <= self.extract_num:
                summaries.append(article)
                references.append([abstract])
                continue

            # edge_scores = self._calculate_similarity_matrix(article)
            article_importance = self._calculate_all_sentence_importance(idx, article)
            print(article_importance)
            ids: List[int] = self._select_tops(article_importance)
            summary = list(map(lambda x: article[x], ids))
            print(summary, abstract)

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

    def _calculate_all_sentence_importance(self, article_idx: int, article: List[str]) -> List[float]:
        all_importances = []
        for idx in tqdm(range(len(article)), desc=f'Article {article_idx}'):
            all_importances.append(self._calculate_sentence_importance(idx, article))
        return all_importances

    def _calculate_sentence_importance(self, i: int, article: List[str]) -> float:
        raise NotImplementedError


class PacSumExtractorWithImportanceV3(PacSumExtractorWithImportance):
    def __init__(self,
                 num_pi_samples: int,
                 num_pj_samples: int,
                 pi_len: int = 7,
                 pj_len: int = 7,
                 window_size: int = 256,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window_size: int = window_size
        self.num_pi_samples: int = num_pi_samples
        self.num_pj_samples: int = num_pj_samples
        # BERT requires pi_length + pj_length < 0.15 * window_size
        self.pi_len: int = pi_len
        self.pj_len: int = pj_len

    def _calculate_sentence_importance(self, i: int, article: List[str]) -> float:
        """
        Sample k phrases pi from si, sample m phrases pj from the window of si Di,
        iota3(si | D) = sum_pi sum_pj (log P(pj | D(si)) - log P(pj | D(si) - pi))
        :param i: The index of sentence si to calculate iota of
        :param article: The article that si is in
        :return: The importance of sentence si
        """
        tokenized_sentences = [self.tokenizer.encode(sent, add_prefix_space=True) for sent in article]
        tokenized_article: List[int] = list(np.concatenate(tokenized_sentences))

        si_tokenized = tokenized_sentences[i][:]
        si_len = len(si_tokenized)
        # relative to article
        si_left = sum(len(sent) for sent in tokenized_sentences[:i])
        si_right = si_left + si_len

        assert si_tokenized == tokenized_sentences[i] == tokenized_article[si_left:si_left + si_len]

        # want window_size window around si
        half_size = (self.window_size - si_len) // 2
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
                loss_pi_unmasked = self.masked_lm(unmasked_batch, masked_lm_labels=labels_batch)[0]
                loss_pi_masked = self.masked_lm(masked_batch, masked_lm_labels=labels_batch)[0]
            s_importance += (loss_pi_masked.item() - loss_pi_unmasked.item())
        return s_importance

    def _generate_batch(self, di: List[int], pi_left: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unmasked_list, masked_list, labels_list = [], [], []
        di_len = len(di)
        di: torch.Tensor = torch.tensor(di)
        pi_mask_range = torch.arange(pi_left, min(di_len, pi_left + self.pi_len)).long()
        for j in range(self.num_pi_samples):
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


class PacSumExtractorWithImportanceV2(PacSumExtractorWithImportance):
    def __init__(self,
                 sentence_sample_window_size: int = 3,
                 num_sentence_samples: int = 3,
                 num_word_samples: int = 3,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sentence_sample_window_size = sentence_sample_window_size
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
        # FIXME: should probably sample from window of si
        sks = np.random.choice(article, self.num_sentence_samples, replace=False)
        for sk in sks:
            with torch.no_grad():
                sentence_pairs, masked_lm_labels, loss_mask = self._generate_batch_for_si_and_sk(si, sk)
                loss = self.masked_lm(sentence_pairs, masked_lm_labels=masked_lm_labels)[0].cpu()
                sentence_pairs_sk, masked_lm_labels_sk, loss_mask_sk = self._generate_batch_for_sk(sk)
                loss_sk = self.masked_lm(sentence_pairs_sk, masked_lm_labels=masked_lm_labels_sk)[0].cpu()
            s_importance += loss - loss_sk
        return s_importance

    def _generate_batch_for_si_and_sk(self, si, sk):
        sk_encoded = self.tokenizer.encode(sk)
        sk_len = len(sk_encoded)

        # sample num_word_samples word indices in sk
        word_indices = np.random.choice(sk_len, self.num_word_samples)

        # sj_masked_copies: [num_word_samples * sk_len]
        sk_copies = torch.tensor([sk_encoded]).repeat(self.num_word_samples, 1)
        mask = torch.eye(sk_len).bool()[word_indices]
        # mask out labels
        sk_masked_copies = sk_copies.masked_fill(mask, self.tokenizer.mask_token_id)

        # si_copies: [num_word_samples * si_len]
        si_encoded = self.tokenizer.encode(si)
        si_len = len(si_encoded)
        si_copies = torch.tensor([si_encoded]).repeat(self.num_word_samples, 1)

        # bos/eos_copies:  [num_word_samples * 1]
        bos_copies = torch.zeros(self.num_word_samples, 1).fill_(self.tokenizer.bos_token_id).long()
        eos_copies = torch.zeros(self.num_word_samples, 1).fill_(self.tokenizer.eos_token_id).long()

        sentence_pairs = torch.cat((bos_copies, si_copies, eos_copies,
                                    eos_copies, sk_masked_copies, eos_copies), 1)

        loss_mask = torch.cat((torch.zeros(self.num_word_samples, si_len + 3).bool(),
                               mask,
                               torch.zeros(self.num_word_samples, 1).bool()), 1)

        masked_lm_labels = torch.zeros_like(loss_mask)
        masked_lm_labels.fill_(-1).long().masked_scatter_(loss_mask, sk_copies)

        return sentence_pairs.to(self.device), masked_lm_labels.to(self.device), loss_mask.to(self.device)

    def _generate_batch_for_sk(self, sk):
        sk_encoded = self.tokenizer.encode(sk)
        sk_len = len(sk_encoded)

        # sample num_word_samples word indices in sk
        word_indices = np.random.choice(sk_len, self.num_word_samples)

        # sj_masked_copies: [num_word_samples * sk_len]
        sk_copies = torch.tensor([sk_encoded]).repeat(self.num_word_samples, 1)
        mask = torch.eye(sk_len).bool()[word_indices]
        # mask out labels
        sk_masked_copies = sk_copies.masked_fill(mask, self.tokenizer.mask_token_id)

        # bos/eos_copies:  [num_word_samples * 1]
        bos_copies = torch.zeros(self.num_word_samples, 1).fill_(self.tokenizer.bos_token_id).long()
        eos_copies = torch.zeros(self.num_word_samples, 1).fill_(self.tokenizer.eos_token_id).long()

        sentence_pairs = torch.cat((bos_copies, sk_masked_copies, eos_copies), 1)

        loss_mask = torch.cat((torch.zeros(self.num_word_samples, 1).bool(),
                               mask,
                               torch.zeros(self.num_word_samples, 1).bool()), 1)

        masked_lm_labels = torch.zeros_like(loss_mask)
        masked_lm_labels.fill_(-1).long().masked_scatter(loss_mask, sk_copies)

        return sentence_pairs.to(self.device), masked_lm_labels.to(self.device), loss_mask.to(self.device)


class PacSumExtractorWithImportanceV1(PacSumExtractorWithImportance):

    def __init__(self, num_word_samples: int = 5, window_size: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_word_samples = num_word_samples
        self.window_size = window_size

    def _calculate_sentence_importance(self, i: int, article: List[str]) -> float:
        """
        Sample k words from D - si, for each word wj, compute its window Dj that contains si.
        iota1(si | D) = sum_j (log P(wk | Dj) - log P(wj | Dj - si))
        :param i: The index of sentence si to calculate iota of
        :param article: The article that si is in
        :return: The importance of sentence si
        """
        w = self.window_size
        si = article[i]
        window_min, window_max = max(0, i - w), min(len(article) - 1, i + w)
        window = article[window_min:window_max + 1]

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
                with torch.no_grad():
                    loss = -self.masked_lm(sentence_pairs, masked_lm_labels=masked_lm_labels)[0].cpu()
                s_importance += loss.item()
        return s_importance

    def _generate_batch(self, si: str, sj: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sj_masked_copies: [sj_len * sj_len]
        sj_encoded = self.tokenizer.encode(sj, add_prefix_space=True)
        sj_len = len(sj_encoded)
        sj_copies = torch.tensor([sj_encoded]).repeat(sj_len, 1)
        mask = torch.eye(sj_len, sj_len).bool()
        # mask out labels
        sj_masked_copies = sj_copies.masked_fill(mask, self.tokenizer.mask_token_id)

        # si_copies: [sj_len * si_len]
        si_encoded = self.tokenizer.encode(si, add_prefix_space=True)
        si_len = len(si_encoded)
        si_copies = torch.tensor([si_encoded]).repeat(sj_len, 1)

        # bos/eos_copies:  [sj_len * 1]
        bos_copies = torch.full((sj_len, 1), self.tokenizer.bos_token_id).long()
        eos_copies = torch.full((sj_len, 1), self.tokenizer.eos_token_id).long()

        sentence_pairs = torch.cat((bos_copies, si_copies, eos_copies,
                                    eos_copies, sj_masked_copies, eos_copies), 1)

        loss_mask = torch.cat((torch.zeros(sj_len, si_len + 3).bool(),
                               mask,
                               torch.zeros(sj_len, 1).bool()), 1)

        masked_lm_labels = torch.full_like(loss_mask, -100).long().masked_scatter_(loss_mask, sj_copies)

        return sentence_pairs.to(self.device), masked_lm_labels.to(self.device), loss_mask.to(self.device)
