from modules.data import Dataset
from modules.model import SelfAttentionModel
from modules.interface import WebInterface
from typing import List, Tuple, Optional, Iterator

class Manager:
    """
    The Manager class orchestrates the entire application flow,
    connecting the dataset processing, model training/inference, and the web interface.
    """
    def __init__(self) -> None:
        """
        Initializes the Manager, setting up the dataset and the attention model.
        """
        self.dataset_manager = Dataset()
        self.dataset_manager.build_dataset()
        self.pairs_generator: Iterator[Tuple[int,int]] = self.dataset_manager.generate_training_pairs()

        print("Dataset Loaded Successfully!")
        self.model = SelfAttentionModel(vocab_size=len(self.dataset_manager.word2idx), embed_dim=50)
        self.is_ready = False

    def model_save(self):
        """
        Saves the current state (weights) of the attention model.
        It checks if the model has been trained or loaded before saving.

        Returns:
            str: A message indicating the status of the save operation.
        """
        if not self.is_ready: return "Henüz Modeli Eğitmediniz!"
        SelfAttentionModel.save_model(self.model)

    def model_upload(self):
        """
        Uploads (loads) a previously saved model state (weights).
        Sets the `is_ready` flag to True upon successful loading.

        Returns:
            str: A message indicating the status of the upload operation.
        """
        SelfAttentionModel.upload_model(self.model)
        self.is_ready = True

    def configurations(self, embed_dim, epochs):
        """
        Configures and trains the Self-Attention Model.
        The `embed_dim` parameter will update the model's embedding dimension,
        and `epochs` will determine the training iterations.

        Args:
            embed_dim (int): The desired embedding dimension for the model.
            epochs (int): The number of epochs to train the model.

        Returns:
            str: A message indicating the completion of the training process.
        """
        SelfAttentionModel.train_self_attention(self.model, 
                                            Dataset.get_trained_pairs(self.pairs_generator),
                                            epochs=epochs)
        self.is_ready = True
        
    def attention(self, word, similar_result_limit):
        """
        Calculates and returns the top attention scores for a given word.
        This function performs inference using the trained model.

        Args:
            word (str): The input word for which to find similar words based on attention.
            similar_result_limit (int): The maximum number of similar words to return.

        Returns:
            List[Tuple[str, float]]: A list of tuples, where each tuple contains
                                    a similar word and its attention score.
        """
        if not self.is_ready: return [("Henüz Modeli Eğitmediniz!", 0.0)]
        try:
            results = SelfAttentionModel.get_attention_scores_for_word(
                                                                        self.model,
                                                                        word,
                                                                        self.dataset_manager.word2idx,
                                                                        self.dataset_manager.idx2word,
                                                                        top_k=similar_result_limit)
            return results
        except ValueError: return [("Kelime bulunamadı.", 0.0)]
        

# --- Main Application Entry Point ---
# Create an instance of the Manager, which initializes the dataset and model.
main = Manager()
# Initialize the WebInterface, passing the Manager's methods as callbacks
# for the UI actions. This connects the frontend to the backend logic.
WebInterface(settings_tab_executer=main.configurations,
            settings_tab_saver=main.model_save,
            settings_tab_uploader=main.model_upload,
            attention_tab_executer=main.attention,
            ).run() # Launch the Gradio web interface.


