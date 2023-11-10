from typing import List, Tuple

import numpy as np
from abc import ABC, abstractmethod

from savant.base.converter import (
    BaseAttributeModelOutputConverter,
)
from savant.base.model import ObjectModel


class LPRecognitionTensorToLabelConverter(BaseAttributeModelOutputConverter):
    def __init__(
        self,
        conf_threshold: float = 0.6,
        min_text_len: int = 10,
        max_text_len: int = 10,
        private_lp_state_codes: List[str] = [],
        **kwargs
    ):
        """
        Converter for converting LP recognition model outputs to label format.

        Args:
            conf_threshold (float): Confidence threshold for accepting labels.
            min_text_len (int): Minimum length of recognized text.
            max_text_len (int): Maximum length of recognized text.
            private_lp_state_codes (List[str]): List of private LP state codes.
        """
        super().__init__(**kwargs)
        self.tokenzier = Tokenizer()
        self.conf_threshold = conf_threshold
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.private_lp_state_codes = private_lp_state_codes

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ObjectModel,
        roi: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """
        Convert LP recognition model outputs to label format.

        Args:
            output_layers: Output arrays from the model.
            model (ObjectModel): The object model.
            roi (Tuple[float, float, float, float]): Region of interest.

        Returns:
            np.ndarray: Array containing recognized LP labels.
        """
        result = []

        outputs = output_layers[0]
        if outputs.size > 0:
            prediction = softmax(outputs)
            texts, probs = self.tokenzier.decode(prediction)
            for text, conf in zip(texts, probs):
                result.append(('lp_result', text, conf))
        return result


class BaseTokenizer(ABC):

    def __init__(
        self,
        charset: str,
        specials_first: tuple = (),
        specials_last: tuple = (),
        conf_threshold: float = 0.5
    ) -> None:
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}
        self.conf_threshold = conf_threshold

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> List[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    @abstractmethod
    def _filter(self, probs: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Internal method which performs the necessary filtering prior to decoding."""
        raise NotImplementedError

    def decode(self, token_dists: np.ndarray, raw: bool = False) -> Tuple[List[str], List[np.ndarray]]:
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return a list of list of strings)

        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of NumPy arrays
        """
        if token_dists.ndim == 2:
            token_dists = np.expand_dims(token_dists, axis=0)

        batch_texts = []
        batch_confs = []
        for dist in token_dists:
            probs, ids = np.max(dist, axis=-1), np.argmax(dist, axis=-1)  # greedy selection
            if not raw:
                probs, ids = self._filter(probs, ids)

            text = self._ids2tok(ids, not raw)
            if probs.size > 0:
                batch_texts.append(text)
                batch_confs.append(np.mean(probs))
        return batch_texts, batch_confs


CHARSET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'


class Tokenizer(BaseTokenizer):
    BOS = '[B]'
    EOS = '[E]'
    PAD = '[P]'

    def __init__(self, charset: str = CHARSET, conf_threshold: float = 0.5) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last, conf_threshold)
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def _filter(self, probs: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        eos_idx = np.where(ids == self.eos_id)[0]
        if eos_idx.size > 0:
            eos_idx = eos_idx[0]
        else:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[:eos_idx]

        conf_mask = probs > self.conf_threshold

        ids = ids[conf_mask]
        probs = probs[conf_mask]

        return probs, ids


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)