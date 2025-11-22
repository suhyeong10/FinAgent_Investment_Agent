import torch
from langchain_huggingface import HuggingFaceEmbeddings

_EMBEDDING_MODEL = None

def _get_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", device=device)

    return _EMBEDDING_MODEL

async def get_embedding(text: str) -> list[float]:
    model = _get_model()
    
    embedding = model.embed_query(text)
    
    return embedding