import os
import fitz                      # pymupdf
import numpy as np
from rapidocr_onnxruntime import RapidOCR

DOCS = "data/docs"
TARGETS = ["Attendance_Policy.pdf", "Exam_Regulations.pdf"]   # add more scanned files here

ocr = RapidOCR()
for fname in TARGETS:
    doc = fitz.open(os.path.join(DOCS, fname))
    out = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        result, _ = ocr(img)
        text = "\n".join(r[1] for r in result) if result else ""
        out.append(f"=== PAGE {i} ===\n{text}")
        print(f"{fname} page {i}: {len(text)} characters")
    with open(os.path.join(DOCS, fname + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))