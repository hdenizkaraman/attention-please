import gradio as gr
from collections.abc import Callable

class WebInterface:
    def __init__(self, 
                settings_tab_executer:Callable,
                settings_tab_saver:Callable,
                settings_tab_uploader:Callable, 
                attention_tab_executer:Callable
                ):
        """
        Initializes the WebInterface for the Self-Attention Model.

        Args:
            settings_tab_executer (Callable): A function to execute model training.
            settings_tab_saver (Callable): A function to save the trained model.
            settings_tab_uploader (Callable): A function to load a pre-trained model.
            attention_tab_executer (Callable): A function to calculate and display attention scores for a word.
        """
        # Store the callable functions that will handle the backend logic for each UI action.
        self._settings_tab_executer = settings_tab_executer
        self._settings_tab_saver = settings_tab_saver
        self._settings_tab_uploader = settings_tab_uploader
        self._attention_tab_executer = attention_tab_executer

    def _settings_tab(self) -> None:
        """
        Defines the UI components and their interactions for the "Training Stage" tab.
        This tab allows users to configure and manage the model training process.
        """
        embed_input = gr.Slider(label="Embedding Boyutu", minimum=16, maximum=128, step=16, value=50)
        epoch_input = gr.Slider(label="Epoch Sayısı", minimum=1, maximum=20, step=1, value=8)
        
        train_button = gr.Button("Modeli Eğit")
        save_button = gr.Button("Eğitilen Modeli Kaydet")
        upload_button = gr.Button("Eğitilen Modeli Yükle")
        train_output = gr.Textbox(label="Durum")

        train_button.click(
            fn=self._settings_tab_executer,
            inputs=[embed_input, epoch_input],
            outputs=train_output,
        )

        save_button.click(
            fn=self._settings_tab_saver,
            outputs=train_output
        )

        upload_button.click(
            fn=self._settings_tab_uploader,
            outputs=train_output
        )

    def _attention_please(self) -> None:
        """
        Defines the UI components and their interactions for the "Word Similarity" tab.
        This tab allows users to query the trained model for attention scores.
        """
        word_input = gr.Textbox(label="Kelime girin")
        topn_slider = gr.Slider(label="Kaç kelime gösterilsin?", minimum=5, maximum=20, step=1, value=10)
        query_button = gr.Button("Benzer Kelimeleri Göster")
        output_table = gr.Dataframe(headers=["Kelime", "Attention Skoru"])

        query_button.click(
            fn=self._attention_tab_executer,
            inputs=[word_input, topn_slider],
            outputs=output_table
        )

    def _base(self) -> gr.Blocks:
        """
        Constructs the main Gradio Blocks layout with tabs for different functionalities.

        Returns:
            gr.Blocks: The Gradio Blocks object representing the complete web interface.
        """
        with gr.Blocks() as layout:
            gr.Markdown("## Self-Attention Model Eğitimi")
            with gr.Tab("Eğitim Aşaması"): self._settings_tab()
            with gr.Tab("Kelime Benzerliği"): self._attention_please()
            return layout
        
    def run(self): 
        """
        Launches the Gradio web interface.
        """
        self._base().launch()

    
