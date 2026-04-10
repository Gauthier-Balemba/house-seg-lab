from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_endpoint_exists():
    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    r = client.post(
        "/predict",
        files={"file": ("test.png", buf, "image/png")},
        headers={"x-api-key": "wrong_key"}
    )
    assert r.status_code in [200, 401]
