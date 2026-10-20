FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && pip install uv

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

ENV HF_HOME=/app/.cache/huggingface

RUN python -c "\
from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

COPY . .

RUN uv pip install --system --no-deps .

EXPOSE 8100

CMD ["uvicorn", "visual_search.main:app", "--host", "0.0.0.0", "--port", "8100"]
