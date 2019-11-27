from typing import List, Tuple, Iterator

import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer

from utils import evaluate_rouge


class PacSumExtractorWithImportance:
    def __init__(self, extract_num: int = 3, device: str = 'cuda') -> None:
        super().__init__()
        self.extract_num: int = extract_num
        self.device: str = device
        self._masked_lm: RobertaForMaskedLM = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
        self._tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained('roberta-base')

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
        si = article[i]
        s_importance = 0
        for j in range(len(article)):
            if j != i:
                sj = article[j]
                # format sentences for masked LM
                sentence_pairs, masked_lm_labels, loss_mask = self._generate_sentence_pairs(si, sj)
                # TODO: What is P_BERT(s_j | s_i)
                # TODO: Is the NLL loss just -sum(log P(w_l | s_i + s_j - w_l))?
                loss = self._masked_lm(sentence_pairs, masked_lm_labels=masked_lm_labels)[0]
                s_importance += loss
        return s_importance

    def _generate_sentence_pairs(self, si: str, sj: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sj_masked_copies: [sj_len * sj_len]
        sj_encoded = self._tokenizer.encode(sj)
        sj_len = len(sj_encoded)
        sj_copies = torch.tensor([sj_encoded]).repeat(sj_len, 1)
        mask = torch.eye(sj_len, sj_len).bool()
        # mask out labels
        sj_masked_copies = sj_copies.masked_fill(mask, self._tokenizer.mask_token_id)

        # si_copies: [sj_len * si_len]
        si_encoded = self._tokenizer.encode(si)
        si_len = len(si_encoded)
        si_copies = torch.tensor([si_encoded]).repeat(sj_len, 1)

        # bos/eos_copies:  [sj_len * 1]
        bos_copies = torch.zeros(sj_len, 1).fill_(self._tokenizer.bos_token_id).long()
        eos_copies = torch.zeros(sj_len, 1).fill_(self._tokenizer.eos_token_id).long()

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
