# HeartFailurePredictionWebApp

PyTorch kalp hastalığı sınıflandırma modeli + FastAPI + HTML/CSS/JS arayüzü.

## Çalıştırma
```bash
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Tarayıcıda: http://127.0.0.1:8000

## Yapı
- `app/model.py` – model sınıfı, ön işleme, tahmin
- `app/main.py` – FastAPI (`POST /api/predict`)
- `static/` – arayüz
