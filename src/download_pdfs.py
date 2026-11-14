"""Download the FinanceBench filings used for development into data/pdfs/.

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

# 46 more filings so that all 111 development questions have their source document (Task 10).
# Together with the 16 above, this covers all 58 filings the dev questions use.
DEV_DOCS = [
    "3M_2018_10K",
    "ACTIVISIONBLIZZARD_2019_10K",
    "ADOBE_2015_10K",
    "ADOBE_2016_10K",
    "ADOBE_2017_10K",
    "ADOBE_2022_10K",
    "AMAZON_2017_10K",
    "AMAZON_2019_10K",
    "AMCOR_2020_10K",
    "AMCOR_2022_8K_dated-2022-07-01",
    "AMCOR_2023Q2_10Q",
    "AMD_2015_10K",
    "AMERICANWATERWORKS_2022_10K",
    "BESTBUY_2017_10K",
    "BESTBUY_2019_10K",
    "BESTBUY_2023_10K",
    "BLOCK_2020_10K",
    "BOEING_2018_10K",
    "COCACOLA_2021_10K",
    "COCACOLA_2022_10K",
    "CORNING_2021_10K",
    "CORNING_2022_10K",
    "COSTCO_2021_10K",
    "CVSHEALTH_2018_10K",
    "CVSHEALTH_2022_10K",
    "FOOTLOCKER_2022_8K_dated_2022-08-19",
    "GENERALMILLS_2019_10K",
    "GENERALMILLS_2020_10K",
    "GENERALMILLS_2022_10K",
    "JOHNSON_JOHNSON_2022Q4_EARNINGS",
    "JOHNSON_JOHNSON_2022_10K",
    "JPMORGAN_2023Q2_10Q",
    "KRAFTHEINZ_2019_10K",
    "LOCKHEEDMARTIN_2020_10K",
    "LOCKHEEDMARTIN_2021_10K",
    "LOCKHEEDMARTIN_2022_10K",
    "MGMRESORTS_2022_10K",
    "MGMRESORTS_2023Q2_10Q",
    "NETFLIX_2015_10K",
    "NETFLIX_2017_10K",
    "NIKE_2021_10K",
    "PAYPAL_2022_10K",
    "PFIZER_2021_10K",
    "ULTABEAUTY_2023_10K",
    "WALMART_2018_10K",
    "WALMART_2020_10K",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in DOCS + DEV_DOCS:
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
