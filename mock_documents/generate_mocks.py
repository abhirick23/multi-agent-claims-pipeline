"""Generates mock medical document images (JPEGs) for the live Streamlit demo.

These mirror the layouts in ``sample_documents_guide.md`` and the content of TC001, TC003,
TC004, and TC011 from ``test_cases.json``, so they can be uploaded through the "Submit Claim"
page to exercise the real Gemini vision path (classification + extraction) end-to-end:

- ``prescription_rajesh.jpg`` + ``hospital_bill_rajesh.jpg``      -> TC004-style clean approval
- ``prescription_arjun.jpg`` + ``hospital_bill_arjun.jpg``          -> TC003-style patient-mismatch
- ``prescription_for_wrong_document_demo.jpg``                      -> TC001-style wrong-doc-type
  (a second prescription, uploaded where a hospital bill is required)
- ``pharmacy_bill_blurry.jpg``                                       -> TC002-style unreadable doc
- ``alt_medicine_prescription.jpg`` + ``alt_medicine_bill.jpg``     -> TC011-style degraded-pipeline demo

Usage::

    python mock_documents/generate_mocks.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

WIDTH = 900
LINE_HEIGHT = 30
MARGIN = 40

FONT_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT = _load_font(20)
FONT_BOLD = _load_font(24)


def render_document(title: str, lines: list[str], output_path: Path) -> None:
    """Render a simple "form" document: bold title, then one line of monospace text per entry.

    A blank string in ``lines`` renders as a horizontal separator, mimicking the boxed sections
    in the sample document layouts.
    """
    height = MARGIN * 2 + LINE_HEIGHT * (len(lines) + 2)
    img = Image.new("RGB", (WIDTH, height), color="white")
    draw = ImageDraw.Draw(img)

    y = MARGIN
    draw.text((MARGIN, y), title, fill="black", font=FONT_BOLD)
    y += LINE_HEIGHT * 2

    for line in lines:
        if line == "":
            draw.line([(MARGIN, y + LINE_HEIGHT // 2), (WIDTH - MARGIN, y + LINE_HEIGHT // 2)], fill="gray", width=1)
        else:
            draw.text((MARGIN, y), line, fill="black", font=FONT)
        y += LINE_HEIGHT

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "JPEG", quality=90)
    print(f"wrote {output_path.relative_to(OUTPUT_DIR.parent.parent)}")


def make_blurry(source_path: Path, output_path: Path) -> None:
    """Re-render ``source_path`` with heavy Gaussian blur, simulating a phone photo too blurry
    to read -- the Document Verification Agent should classify this as ``quality=UNREADABLE``.
    """
    img = cv2.imread(str(source_path))
    blurred = cv2.GaussianBlur(img, (35, 35), 0)
    cv2.imwrite(str(output_path), blurred)
    print(f"wrote {output_path.relative_to(OUTPUT_DIR.parent.parent)}")


def main() -> None:
    # --- TC004-style clean consultation: Rajesh Kumar -----------------------------------------
    render_document(
        "Dr. Arun Sharma, MBBS, MD (Internal Medicine)",
        [
            "Reg. No: KA/45678/2015",
            "City Medical Centre, 12 MG Road, Bengaluru",
            "",
            "Patient: Rajesh Kumar              Date: 01-Nov-2024",
            "Age: 39 years    Gender: M",
            "",
            "Diagnosis: Viral Fever",
            "",
            "Rx:",
            "1. Tab Paracetamol 650mg - 1-1-1 x 5 days",
            "2. Tab Vitamin C 500mg - 0-0-1 x 7 days",
            "",
            "Investigations: CBC, Dengue NS1",
            "Follow-up: After 5 days if no improvement",
        ],
        OUTPUT_DIR / "prescription_rajesh.jpg",
    )

    render_document(
        "CITY CLINIC, BENGALURU",
        [
            "Bill No: CMC/2024/08321         Date: 01-Nov-2024",
            "",
            "Patient Name: Rajesh Kumar",
            "Age/Gender: 39 / Male",
            "Referring Doctor: Dr. Arun Sharma",
            "",
            "DESCRIPTION                  QTY    RATE     AMOUNT",
            "Consultation Fee (OPD)        1    1000.00   1000.00",
            "CBC (Complete Blood Count)    1     300.00    300.00",
            "Dengue NS1 Antigen Test       1     200.00    200.00",
            "",
            "Total Amount:                                1500.00",
        ],
        OUTPUT_DIR / "hospital_bill_rajesh.jpg",
    )

    # --- TC003-style patient mismatch: Arjun Mehta's bill paired with Rajesh's prescription ----
    render_document(
        "Dr. Arun Sharma, MBBS, MD (Internal Medicine)",
        [
            "Reg. No: KA/45678/2015",
            "City Medical Centre, 12 MG Road, Bengaluru",
            "",
            "Patient: Arjun Mehta                Date: 01-Nov-2024",
            "Age: 45 years    Gender: M",
            "",
            "Diagnosis: Acute Gastritis",
            "",
            "Rx:",
            "1. Tab Pantoprazole 40mg - 1-0-0 x 7 days",
            "2. Tab Domperidone 10mg - 1-1-1 x 5 days",
        ],
        OUTPUT_DIR / "prescription_arjun.jpg",
    )

    render_document(
        "CITY CLINIC, BENGALURU",
        [
            "Bill No: CMC/2024/08322         Date: 01-Nov-2024",
            "",
            "Patient Name: Arjun Mehta",
            "Age/Gender: 45 / Male",
            "Referring Doctor: Dr. Arun Sharma",
            "",
            "DESCRIPTION                  QTY    RATE     AMOUNT",
            "Consultation Fee (OPD)        1    1000.00   1000.00",
            "Medicines                     1     500.00    500.00",
            "",
            "Total Amount:                                1500.00",
        ],
        OUTPUT_DIR / "hospital_bill_arjun.jpg",
    )

    # --- TC001-style wrong document type: a second prescription where a hospital bill is needed
    render_document(
        "Dr. Meera Iyer, MBBS, DNB (Family Medicine)",
        [
            "Reg. No: TN/56789/2013",
            "Sunrise Clinic, Anna Nagar, Chennai",
            "",
            "Patient: Rajesh Kumar               Date: 01-Nov-2024",
            "Age: 39 years    Gender: M",
            "",
            "Diagnosis: Seasonal Allergic Rhinitis",
            "",
            "Rx:",
            "1. Tab Cetirizine 10mg - 0-0-1 x 5 days",
            "2. Nasal Spray (Fluticasone) - 1 spray each nostril OD",
        ],
        OUTPUT_DIR / "prescription_for_wrong_document_demo.jpg",
    )

    # --- TC002-style unreadable document: pharmacy bill, then heavily blurred ------------------
    sharp_path = OUTPUT_DIR / "_pharmacy_bill_sharp_tmp.jpg"
    render_document(
        "HEALTH FIRST PHARMACY",
        [
            "Drug Lic. No: KA-BLR-XXXX",
            "22 Brigade Road, Bengaluru",
            "",
            "Bill No: HFP-24-09821      Date: 25-Oct-2024",
            "Patient: Sunita Reddy      Dr: Dr. Arun Sharma",
            "",
            "MEDICINE          BATCH   EXP     QTY  MRP    AMT",
            "Paracetamol 650   A2341   03/26    15  2.50   37.50",
            "Vitamin C 500     B7821   06/26    10  4.00   40.00",
            "",
            "Net Amount:                               800.00",
        ],
        sharp_path,
    )
    make_blurry(sharp_path, OUTPUT_DIR / "pharmacy_bill_blurry.jpg")
    sharp_path.unlink()

    # --- TC011-style alternative medicine claim (used with simulate_component_failure) --------
    render_document(
        "Vaidya T. Krishnan, BAMS (Ayurveda)",
        [
            "Reg. No: AYUR/KL/2345/2019",
            "Ayur Wellness Centre, Kochi",
            "",
            "Patient: Lakshmi Nair               Date: 28-Oct-2024",
            "Age: 52 years    Gender: F",
            "",
            "Diagnosis: Chronic Joint Pain",
            "",
            "Treatment: Panchakarma Therapy",
            "Plan: 5 sessions over 2 weeks",
        ],
        OUTPUT_DIR / "alt_medicine_prescription.jpg",
    )

    render_document(
        "AYUR WELLNESS CENTRE, KOCHI",
        [
            "Bill No: AWC-24-0451        Date: 28-Oct-2024",
            "",
            "Patient Name: Lakshmi Nair",
            "",
            "DESCRIPTION                          AMOUNT",
            "Panchakarma Therapy (5 sessions)     3000.00",
            "Consultation                         1000.00",
            "",
            "Total Amount:                        4000.00",
        ],
        OUTPUT_DIR / "alt_medicine_bill.jpg",
    )


if __name__ == "__main__":
    main()
