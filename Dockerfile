FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image small enough for a free-tier instance
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir pandas numpy scikit-learn joblib "fastapi[standard]" uvicorn requests

COPY main.py model.py ./
COPY static ./static
COPY artifacts/meta.json artifacts/scaler.pkl artifacts/encoder.pkl artifacts/sf_ids_model.pth ./artifacts/

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
