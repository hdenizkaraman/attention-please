from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from typing import Optional, Generic, TypeVar, Type
import random

# Defines SelfAttentionModel type for make it useable inside the class.
SAM = TypeVar("SAM", bound="SelfAttentionModel")

class SelfAttentionModel(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Creates an embedding layer which turns words into numeric vectors.
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, center_idx, context_idx):
        """
        Forward pass for training the model.
        This calculates a "relevance score" between a center word and a context word.

        Args:
            center_idx (torch.Tensor): Tensor of indices for the center words.
            context_idx (torch.Tensor): Tensor of indices for the context words.

        Returns:
            torch.Tensor: Logits representing the similarity between center and context embeddings.
        """
        # Get embeddings for the center and context words
        center_embed = self.embedding(center_idx)
        context_embed = self.embedding(context_idx)

        # Generate Query (Q) for the center word and Key (K) for the context word
        Q = self.query(center_embed)
        K = self.key(context_embed)

        # Calculate a dot-product similarity
        # This acts as a basic relevance score for the training objective.
        logits = torch.sum(Q * K, dim=1)
        return logits

    def inference(self, input_idx, all_indices):
        """
        Performs the full self-attention mechanism to get attention scores
        and weighted output for an input word against all other words in the vocabulary.

        Args:
            input_idx (torch.Tensor): Index of the word for which to calculate attention.
            all_indices (torch.Tensor): Indices of all words in the vocabulary (used as keys and values).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - attn_scores (torch.Tensor): Attention weights (probabilities) for each word in the vocabulary.
                - output (torch.Tensor): The context vector, a weighted sum of value vectors based on attention scores.
        """
        # Get embedding for the input word (this will be query)
        input_embed = self.embedding(input_idx)              
        all_embed = self.embedding(all_indices)              
        
        # Generate Query (Q) for the center word, Key (K) for the context word and Values (V) for all words
        Q = self.query(input_embed)                          
        K = self.key(all_embed).T                            
        V = self.value(all_embed)                            

        # Calculate attention scores:
        # 1. Dot product of Query with all Keys (Q @ K) (matrix multiplication)
        # 2. Scale by the square root of the dimension of Q (for stability)
        # 3. Apply softmax to get probability distribution (attention weights)
        attn_scores = torch.softmax(Q @ K / (Q.shape[-1] ** 0.5), dim=-1)
        
        # Calculate the output: weighted sum of Value vectors using attention scores
        # This combines the information from all words, weighted by their relevance to the input.
        output = attn_scores @ V                             
        return attn_scores, output


    @staticmethod
    def train_self_attention(model:Type[SAM], pairs, epochs=5, batch_size=128, lr=0.001):
        """
        Trains the SelfAttentionModel using positive and negative sampling.

        Args:
            model (SelfAttentionModel): The model instance to train.
            pairs (list): A list of (center_word_id, context_word_id) positive pairs.
            epochs (int): Number of training epochs.
            batch_size (int): Number of samples per training batch.
            lr (float): Learning rate for the optimizer.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Binary Cross-Entropy with Logits Loss is suitable for binary classification (positive/negative pair).
        loss_fn = nn.BCEWithLogitsLoss()

        vocab_size = model.embedding.num_embeddings

        for epoch in range(epochs):
            random.shuffle(pairs)
            total_loss = 0

            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]
                if len(batch) < batch_size: continue

                # Prepare positive samples
                centers, contexts = zip(*batch)
                centers = torch.tensor(centers)
                contexts = torch.tensor(contexts)
                labels = torch.ones(batch_size)

                # Negative Sampling: Randomly select words that are NOT context words
                # This helps the model learn to distinguish relevant from irrelevant pairs.
                negatives = torch.randint(0, vocab_size, (batch_size,))
                
                # Combine positive and negative samples for training
                all_contexts = torch.cat([contexts, negatives])
                all_centers = torch.cat([centers, centers])
                all_labels = torch.cat([labels, torch.zeros(batch_size)])

                optimizer.zero_grad() # Clears previous gradients
                
                # Forward pass: get logits for all combined samples
                logits = model(all_centers, all_contexts)
                loss = loss_fn(logits, all_labels) # Calculate loss
                loss.backward() # Backpropagate to compute gradients
                optimizer.step() # Update model weights

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(pairs)}")

    @staticmethod
    def get_attention_scores_for_word(model:Type[SAM], word, word_to_id, id_to_word, top_k=10):
        """
        Calculates and returns the top K words with the highest attention scores for a given word.

        Args:
            model (SelfAttentionModel): The trained model.
            word (str): The input word for which to get attention scores.
            word_to_id (dict): Dictionary mapping words to their integer IDs.
            id_to_word (dict): Dictionary mapping integer IDs to words.
            top_k (int): Number of top words to return.

        Returns:
            list: A list of tuples (word, attention_score) for the top K most attended words.
        Raises:
            ValueError: If the input word is not in the vocabulary.
        """
        
        if word not in word_to_id: raise ValueError(f"{word} not in vocabulary.")
        
        input_idx = torch.tensor([word_to_id[word]]) # Get ID of the input word
        all_indices = torch.tensor(list(word_to_id.values())) # Get IDs of all words in the vocabulary

        with torch.no_grad(): # Disable gradient calculation for inference
            # Calls the 'inference' method, which contains the full attention mechanism.
            attention_scores, _ = model.inference(input_idx, all_indices)
            attention_scores = attention_scores.squeeze()
        # Get the top K attention values and their corresponding indices
        topk_values, topk_indices = torch.topk(attention_scores, top_k)

        # Map indices back to words and return results
        result = [(id_to_word[idx.item()], val.item()) for idx, val in zip(topk_indices, topk_values)]
        return result
    
    @staticmethod
    def save_model(model:Type[SAM]) -> None:
        """
        Saves the state dictionary (weights) of the model to a file.

        Args:
            model (SelfAttentionModel): The model instance to save.

        Returns:
            str: A message indicating success or failure.
        """
        try:
            torch.save(model.state_dict(), "./cache/self_attention_model.pth")
            return "Model Kaydedildi"
        except Exception as error:
            print(error)
            return "Hata: Model Kaydedilemedi."
        
    @staticmethod
    def upload_model(model:Type[SAM]) -> None:
        """
        Loads the state dictionary (weights) into the model from a file.

        Args:
            model (SelfAttentionModel): The model instance to load weights into.

        Returns:
            str: A message indicating success or failure.
        """
        try:
            model.load_state_dict(torch.load("./cache/self_attention_model.pth"))
            model.eval()
        except Exception as error:
            print(error)
            return "Hata: Model Yüklenemedi."        