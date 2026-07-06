from sentence_transformers import SentenceTransformer

from .base import BaseEmbedding 

class SentenceEmbedding(BaseEmbedding):

    def __init__(
        self,
        model: str
    ):
        self.model = SentenceTransformer(
            model
        )

    def embed_text(
        self,
        text: str
    ):
        return self.model.encode(
            text
        ).tolist()
    
    def embed_document(
        self, 
        texts: list[str]
    ):
        
        return self.model.encode(
            texts
        ).tolist()