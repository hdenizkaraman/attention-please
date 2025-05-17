from datasets import load_dataset
from .settings import DATASET, DATASET_FIELD, STOPWORDS
from typing import List, Dict, Iterator, Tuple, Optional
import re
from collections import Counter

class Dataset:
    """
    Dataset class to handle loading and processing of the dataset for natural language tasks.
    It manages vocabulary creation, sentence cleaning, and generating training pairs.
    """

    def __init__(self):
        """
        Initializes the Dataset class by loading the raw dataset.
        """
        # Load the specified dataset from Hugging Face datasets.
        # 'split="train"' ensures we're loading the training portion of the dataset.
        self._dataset = load_dataset(DATASET, split="train")
        self._vocab: Optional[Counter] = None # Contains clean words.
        self._sentences: Optional[List[str]] = None # Contains sentences.

    def _tokenize(self) -> Iterator[str]:
        """
        Performs basic text cleaning and tokenization on the dataset.
        This generator yields individual cleaned words.

        Yields:
            str: A cleaned token (word) from the dataset.
        """
        for item in self._dataset:
            text = item.get(DATASET_FIELD, "").lower()
            text = re.sub(r"[^\w\s]", "", text)
            for token in text.split(): 
                if token in STOPWORDS: continue
                yield token

    def _clear_sentences(self) -> list[str]:
        """
        Performs basic cleaning on sentences.
        Specifically, it handles contractions by removing the apostrophe and subsequent characters.

        Returns:
            list[str]: A list of cleaned sentences.
        """
        final_sentences: list[str] = []
        for item in self._dataset:
            text = item.get(DATASET_FIELD, "").lower()
            text = re.sub(r"\b(\w+)[’'][\wçğıöşüÇĞİÖÜŞ]+", r"\1", text)
            final_sentences.append(text)
        return final_sentences

    def build_dataset(self) -> Tuple[bool, str]:
        """
        Builds the complete dataset, including vocabulary and cleaned sentences,
        and creates word-to-index and index-to-word mappings.

        Returns:
            tuple[bool, str]: A tuple indicating success (True/False) and a descriptive message.
        """
        self._vocab = Counter(self._tokenize())
        self._sentences = self._clear_sentences()
        self.word2idx = {word: idx for idx, word in enumerate(self._vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        return True, "Dataset built successfully."
    
    def get_sentences(self) -> List[str]:
        """
        Retrieves the list of cleaned sentences.

        Returns:
            list[str]: A list of cleaned sentences.

        Raises:
            ValueError: If the dataset has not been built yet.
        """
        if self._sentences is None: raise ValueError("Dataset not built. Call build_local_dataset() first.")
        return self._sentences

    def get_vocab(self) -> Counter:
        """
        Retrieves the vocabulary (word counts).

        Returns:
            Counter: A Counter object containing word frequencies.

        Raises:
            ValueError: If the dataset has not been built yet.
        """
        if self._vocab is None: raise ValueError("Dataset not built. Call build_local_dataset() first.")
        return self._vocab
    
    def generate_training_pairs(self, window_size=2) -> Iterator[Tuple[int, int]]:
        """
        Generates (target_word_id, context_word_id) pairs for training a word embedding model.
        This uses a sliding window approach to capture words that appear near each other.

        Args:
            window_size (int): The number of words to consider on each side of the target word
                                as context words.

        Yields:
            Tuple[int, int]: A tuple containing the integer ID of the target word
                             and the integer ID of a context word.
        """
        for sentence in self._sentences:
            tokens = re.findall(r'\w+', sentence.lower())
            length = len(tokens)

            for i, target_word in enumerate(tokens):
                start = max(0, i - window_size)
                end = min(length, i + window_size + 1)

                context_words = tokens[start:i] + tokens[i+1:end]
                
                for context_word in context_words: 
                    if target_word in self.word2idx and context_word in self.word2idx:
                        yield (self.word2idx[target_word], self.word2idx[context_word])

    @staticmethod
    def get_trained_pairs(pairs_generator:Iterator) -> List[Tuple[str, str]]:
        """
        Converts an iterator of training pairs into a list.

        Args:
            pairs_generator (Iterator): An iterator that yields (target_word_id, context_word_id) tuples.

        Returns:
            List[Tuple[int, int]]: A list containing all the generated training pairs.
        """
        return list(pairs_generator)