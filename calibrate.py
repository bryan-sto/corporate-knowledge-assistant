"""
Empirical Similarity Threshold Calibration Script
Runs in-scope vs out-of-scope queries against the ingested FAISS index
to validate the decision gap for config.RETRIEVAL_SIMILARITY_THRESHOLD.
"""
import sys
from pathlib import Path

import config
import rag

IN_SCOPE_QUESTIONS = [
    "Berapa jatah cuti tahunan karyawan?",
    "Berapa plafon klaim kacamata untuk level Manager?",
    "Apakah karyawan probation boleh mengajukan WFA?",
    "Berapa lama jatah Cuti Ayah?",
    "Bagaimana aturan penggunaan ChatGPT untuk karyawan?"
]

OUT_OF_SCOPE_QUESTIONS = [
    "Siapa presiden Amerika Serikat saat ini?",
    "Berapa harga saham Apple hari ini?",
    "Bagaimana resep membuat nasi goreng spesial?",
    "Siapa pemenang Liga Champions tahun lalu?",
    "Apa ibukota dari negara Prancis?"
]


def run_calibration():
    print("=" * 70)
    print(" EMPIRICAL RETRIEVAL SIMILARITY THRESHOLD CALIBRATION")
    print("=" * 70)
    
    # Ensure PDF is ingested
    sample_pdf = Path(__file__).parent / "dev" / "[PT TMD] BUKU SAKU KEBIJAKAN SUMBER DAYA MANUSIA.pdf"
    if not sample_pdf.exists():
        sample_pdf = Path(__file__).parent / "[PT TMD] BUKU SAKU KEBIJAKAN SUMBER DAYA MANUSIA.pdf"
    
    if sample_pdf.exists():
        print(f"\n[1] Ingesting sample PDF: {sample_pdf.name}")
        with open(sample_pdf, "rb") as f:
            pdf_bytes = f.read()
        res = rag.ingest_pdf_bytes(pdf_bytes, sample_pdf.name)
        print(f"    Ingested {res['chunks_ingested']} chunks into FAISS vector store.\n")
    else:
        print("\n[!] Warning: Sample PDF not found. Using existing index if present.\n")

    model = rag.get_embedding_model()

    print("[2] Evaluating IN-SCOPE Queries:")
    print("-" * 70)
    in_scope_scores = []
    for q in IN_SCOPE_QUESTIONS:
        results = rag.vector_store.search(q, model, top_k=1)
        if results:
            score, chunk = results[0]
            in_scope_scores.append(score)
            print(f"  [SCORE: {score:.4f}] Query: '{q}'")
            print(f"               Matched: [{chunk['bab']} - {chunk['section_title']}]")
        else:
            print(f"  [NO MATCH] Query: '{q}'")

    print("\n[3] Evaluating OUT-OF-SCOPE Queries:")
    print("-" * 70)
    out_scope_scores = []
    for q in OUT_OF_SCOPE_QUESTIONS:
        results = rag.vector_store.search(q, model, top_k=1)
        if results:
            score, chunk = results[0]
            out_scope_scores.append(score)
            print(f"  [SCORE: {score:.4f}] Query: '{q}'")
            print(f"               Matched: [{chunk['bab']} - {chunk['section_title']}]")
        else:
            print(f"  [NO MATCH] Query: '{q}'")

    print("\n" + "=" * 70)
    print(" CALIBRATION SUMMARY & DECISION GAP ANALYSIS")
    print("=" * 70)
    if in_scope_scores and out_scope_scores:
        min_in = min(in_scope_scores)
        max_in = max(in_scope_scores)
        avg_in = sum(in_scope_scores) / len(in_scope_scores)

        min_out = min(out_scope_scores)
        max_out = max(out_scope_scores)
        avg_out = sum(out_scope_scores) / len(out_scope_scores)

        gap = min_in - max_out
        suggested_threshold = round((max_out + min_in) / 2, 2)

        print(f"  In-Scope Scores  : Min = {min_in:.4f} | Max = {max_in:.4f} | Avg = {avg_in:.4f}")
        print(f"  Out-of-Scope     : Min = {min_out:.4f} | Max = {max_out:.4f} | Avg = {avg_out:.4f}")
        print(f"  Decision Gap     : {gap:.4f} (Cluster Separation)")
        print(f"  Current Config   : RETRIEVAL_SIMILARITY_THRESHOLD = {config.RETRIEVAL_SIMILARITY_THRESHOLD}")
        print(f"  Empirical Rec.   : Suggested Threshold = {suggested_threshold:.2f}")

        if min_in > max_out:
            print("\n  [RESULT] PERFECT SEPARATION DETECTED! Layer 1 Guardrail reliably discriminates clusters.")
        else:
            print("\n  [RESULT] Overlapping clusters. Rely on Layer 2 System Prompt Guardrail.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_calibration()
