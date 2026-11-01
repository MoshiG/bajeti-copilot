# Bajeti Copilot demo image.
#
# The image holds code and models' runtime only. Build the indexes first (make fetch ingest index),
# then mount ./data and point the copilot at an LLM endpoint:
#
#   docker build -t bajeti-copilot .
#   docker run -p 7860:7860 -v "$(pwd)/data:/app/data" \
#     -e BAJETI_LLM_BASE_URL=https://your-endpoint/v1 -e BAJETI_LLM_MODEL=your-model \
#     -e BAJETI_API_KEY=... bajeti-copilot

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-swa \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image much smaller; embedding one query on CPU is fast enough.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[ingest,index,embed,ui]"

COPY configs ./configs

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data /app/.cache && chown -R app /app
USER app

EXPOSE 7860
CMD ["bajeti", "demo", "--host", "0.0.0.0"]
