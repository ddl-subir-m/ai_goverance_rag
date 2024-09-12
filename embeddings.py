from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List

class CodeBERTEmbeddings(Embeddings):
    """
    A custom embedding class using CodeBERT model for code-related text embeddings.
    """

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """
        Initialize the CodeBERTEmbeddings class.

        Args:
            model_name (str): The name of the CodeBERT model to use.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using the CodeBERT model.

        Args:
            texts (List[str]): A list of text documents to embed.

        Returns:
            List[List[float]]: A list of embeddings, one for each input document.
        """
        embeddings = []
        for text in texts:
            # Tokenize the input text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Average the last hidden state to get the final embedding
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text (str): The query text to embed.

        Returns:
            List[float]: The embedding of the query text.
        """
        return self.embed_documents([text])[0]

