# Multi-stage build keeps the final image small
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

# Install only what serving needs (skip dev/training deps)
RUN pip install --no-cache-dir --user \
    torch==2.3.0 \
    numpy==1.26.4 \
    mlflow==2.14.1 \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    pydantic==2.7.4

# ---

FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy code
COPY src/ ./src/
COPY mlruns/ ./mlruns/

ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV MODEL_STAGE=Production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
COPY models/ ./models/
