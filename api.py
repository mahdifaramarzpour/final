# ============================================================
# api.py — FastAPI برای مدل MNIST
# ============================================================
# اجرا:   uvicorn api:app --reload
# تست:    http://localhost:8000/docs
#
# این فایل کنار Model_load.py قرار می‌گیرد و از همان تابع
# predict_mnist_probabilities استفاده می‌کند که main.py استفاده می‌کند.
# ============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
from pathlib import Path
import numpy as np
import Model_load as Ml          # ← همان فایل موجود پروژه


# ---------- ساخت اپ ----------
app = FastAPI(
    title="MNIST API",
    description="API برای پیش‌بینی رقم دست‌نویس — مدل PyTorch",
    version="1.0.0",
)


# ============================================================
# اندپوینت ۱: سلامت سرویس
# GET /
# ============================================================
@app.get("/", tags=["عمومی"])
def root():
    return {"status": "ok", "message": "MNIST API آماده است"}


# ============================================================
# اندپوینت ۲: اطلاعات مدل
# GET /info
# ============================================================
@app.get("/info", tags=["عمومی"])
def info():
    model_path = Path(__file__).parent / "mnist_model.pth"
    return {
        "model_name": "MNIST CNN",
        "framework": "PyTorch",
        "input_shape": "1×28×28",
        "num_classes": 10,
        "model_exists": model_path.exists(),
        "model_size_kb": round(model_path.stat().st_size / 1024, 1) if model_path.exists() else None,
    }


# ============================================================
# اندپوینت ۳: پیش‌بینی یک تصویر
# POST /predict
# ============================================================
@app.post("/predict", tags=["پیش‌بینی"])
async def predict(file: UploadFile = File(...)):
    """
    یک تصویر رقم آپلود کن و رقم پیش‌بینی‌شده را بگیر.

    ورودی: فایل تصویر (multipart/form-data)
    خروجی: JSON شامل رقم، اطمینان، و احتمال هر ۱۰ رقم
    """
    # ---- ۱. چک فرمت ----
    allowed = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}
    if file.content_type not in allowed:
        raise HTTPException(400, f"فرمت `{file.content_type}` پشتیبانی نمی‌شود.")

    # ---- ۲. خواندن تصویر ----
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
    except Exception:
        raise HTTPException(400, "تصویر نامعتبر است.")

    # ---- ۳. ذخیره‌ی موقت چون Model_load مسیر می‌گیرد ----
    temp_path = Path("temp_api_image.png")
    try:
        image.save(temp_path)

        # ---- ۴. پیش‌بینی ----
        result_str = Ml.predict_mnist_probabilities(str(temp_path))
    finally:
        # فایل موقت را پاک کن (حتی اگر خطا رخ دهد)
        if temp_path.exists():
            temp_path.unlink()

    # ---- ۵. چک خطای مدل ----
    if result_str.startswith("Error"):
        raise HTTPException(500, result_str)

    # ---- ۶. تجزیه‌ی خروجی رشته‌ای ----
    # خروجی: "0: 0.12\n1: 0.05\n2: 0.83\n..."
    probabilities = [0.0] * 10
    for line in result_str.strip().split("\n"):
        digit, prob = line.split(": ")
        probabilities[int(digit)] = float(prob)

    # ---- ۷. پیدا کردن رقم برنده ----
    digit = int(np.argmax(probabilities))
    confidence = probabilities[digit]

    # ---- ۸. پاسخ JSON ----
    return {
        "filename": file.filename,
        "digit": digit,
        "confidence": round(confidence, 4),
        "probabilities": [round(p, 4) for p in probabilities],
    }


# ============================================================
# اندپوینت ۴: پیش‌بینی چند تصویر همزمان
# POST /predict/batch
# ============================================================
@app.post("/predict/batch", tags=["پیش‌بینی"])
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    چند تصویر را همزمان پیش‌بینی می‌کند.
    حداکثر ۱۰ تصویر.
    """
    if len(files) > 10:
        raise HTTPException(400, "حداکثر ۱۰ تصویر.")

    results = []
    for f in files:
        try:
            contents = await f.read()
            image = Image.open(BytesIO(contents))

            temp_path = Path(f"temp_{f.filename}")
            image.save(temp_path)
            try:
                result_str = Ml.predict_mnist_probabilities(str(temp_path))
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            if result_str.startswith("Error"):
                results.append({"filename": f.filename, "error": result_str})
                continue

            probs = [0.0] * 10
            for line in result_str.strip().split("\n"):
                d, p = line.split(": ")
                probs[int(d)] = float(p)

            results.append({
                "filename": f.filename,
                "digit": int(np.argmax(probs)),
                "confidence": round(max(probs), 4),
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return {"count": len(results), "results": results}