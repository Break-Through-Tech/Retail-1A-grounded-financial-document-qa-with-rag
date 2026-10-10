"""Download a small set of FinanceBench filings into data/pdfs/.

Usage: python src/download_pdfs.py
"""

import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/"
OUT = Path(__file__).resolve().parents[1] / "data" / "pdfs"

# 16 filings covering 60 of the 150 questions, mixing all four document types.
DOCS = [
    "AMD_2022_10K",
    "AMERICANEXPRESS_2022_10K",
    "BOEING_2022_10K",
    "PEPSICO_2022_10K",
    "AMCOR_2023_10K",
    "AES_2022_10K",
    "BESTBUY_2024Q2_10Q",
    "3M_2023Q2_10Q",
    "Pfizer_2023Q2_10Q",
    "JPMORGAN_2021Q1_10Q",
    "JOHNSON_JOHNSON_2023_8K_dated-2023-08-30",
    "PEPSICO_2023_8K_dated-2023-05-30",
    "ULTABEAUTY_2023Q4_EARNINGS",
    "MGMRESORTS_2022Q4_EARNINGS",
    "PEPSICO_2023Q1_EARNINGS",
    "AMCOR_2023Q4_EARNINGS",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in DOCS:
        path = OUT / f"{name}.pdf"
        if path.exists():
            print(f"skip {name}")
            continue
        try:
            urllib.request.urlretrieve(BASE + name + ".pdf", path)
            print(f"ok   {name} ({path.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"FAIL {name}: {e}")


if __name__ == "__main__":
    main()
