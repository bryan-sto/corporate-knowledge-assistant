from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from main import app
import config

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"

    # Also verify dedicated /health endpoint
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "online"


def test_ingest_invalid_file_type():
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", b"Invalid text file content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_ingest_sample_pdf():
    pdf_path = Path(__file__).parent.parent / "dev" / "[PT TMD] BUKU SAKU KEBIJAKAN SUMBER DAYA MANUSIA.pdf"
    if not pdf_path.exists():
        pytest.skip("Sample PDF not found.")

    with open(pdf_path, "rb") as f:
        response = client.post(
            "/ingest",
            files={"file": (pdf_path.name, f.read(), "application/pdf")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_ingested"] == 11


def test_chat_in_domain_query():
    response = client.post(
        "/chat",
        json={"question": "Berapa jatah cuti tahunan karyawan?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    assert data["sources"][0]["section_number"] == "2.1"
    assert data["sources"][0]["section_title"] == "Cuti Tahunan"


def test_chat_out_of_domain_query_guardrail():
    response = client.post(
        "/chat",
        json={"question": "Siapa presiden Amerika Serikat saat ini?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Maaf, saya hanya bisa menjawab terkait kebijakan internal."
    assert data["guardrail_triggered"] is not None
