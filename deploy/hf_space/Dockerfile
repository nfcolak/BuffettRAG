# BuffettRAG Space: CPU retrieval + API + built frontend; answers come from the remote model server.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

# HF Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1.0" \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-space.txt

USER user

# Pre-download both retrieval models at build time (no download on a cold start).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu'); \
CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')"

COPY --chown=user . .

ENV VECTOR_BACKEND=faiss \
    FAISS_INDEX_NAME=faiss_v3 \
    DEFAULT_LLM_PROVIDER=remote \
    EMBEDDING_DEVICE=cpu \
    HF_HUB_OFFLINE=1

EXPOSE 7860
CMD ["uvicorn", "src.services.space_app:app", "--host", "0.0.0.0", "--port", "7860"]
