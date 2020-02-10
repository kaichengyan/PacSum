from typing import Tuple, List

import torch
from tqdm import tqdm
from transformers import RobertaTokenizer

from models.base_model import PacSumExtractorWithImportance


class SentencePairModel(PacSumExtractorWithImportance):
    """
    Importance model v0
    iota0(si | D) = sum_j sum_k (log P(wk | sj' + si))
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calculate_article_importance(self, article_idx: int, article: List[str]) -> List[float]:
        all_importances = []
        article = [RobertaTokenizer.clean_up_tokenization(s) for s in article]
        for idx in tqdm(range(len(article)), desc=f'Article {article_idx}'):
            all_importances.append(self._calculate_single_sentence_importance(idx, article))
        return all_importances

    def _calculate_single_sentence_importance(self, i: int, article: List[str]) -> float:
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
