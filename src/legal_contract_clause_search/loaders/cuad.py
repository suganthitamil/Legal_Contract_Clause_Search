"""Load CUAD contracts into our `Contract` model.

WHERE THE DATA COMES FROM (decision — see docs/DECISIONS.md):
  - TEXT  -> CUAD_v1.json  (510 contracts; the SAME text the gold clause offsets
                            `answer_start` refer to, so evaluation lines up later)
  - DATES, PARTIES -> master_clauses.csv (510 rows)
  We do NOT use full_contract_txt/: only 499 files and the names don't match the CSV.

HOW WE JOIN JSON <-> CSV:
  The names differ in spaces, dots, case and endings (".pdf"), e.g.
      CSV : "ELECTRAMECCANICA VEHICLES CORP. - Manufacturing Agreement "
      JSON: "ELECTRAMECCANICA VEHICLES CORP - Manufacturing Agreement"
  Keeping ONLY lower-case letters and digits makes them identical -> all 510 match.
"""

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.model import Contract

# ---------------------------------------------------------------------------
# Step 1 — small helper functions (each does ONE thing, easy to test)
# ---------------------------------------------------------------------------

def make_key(name: str) -> str:
    """Turn a file name/title into a join key: lower-case letters + digits only.

    "ACME Corp. - Supply Agreement .pdf"  ->  "acmecorpsupplyagreementpdf"
    """
    # TODO (YOU DO): build the result from the characters of `name.lower()`
    # that are letters or digits. Hint: str.isalnum() + "".join(...) + a comprehension.
    result = ""
    for c in name.lower():
        if c.isalnum():
            result +=c 
  
    return result


# (keyword to look for in the lower-cased title, canonical type)
# Module-level constant (ALL_CAPS): built ONCE when the file is imported,
# not rebuilt on every function call.
# ORDER MATTERS: the first keyword found wins -> specific rules before general ones.
CONTRACT_TYPE_RULES = [
    ("co-branding", "Co-Branding"), ("co_branding", "Co-Branding"), ("cobranding", "Co-Branding"),
    ("strategic alliance", "Strategic Alliance"),
    ("joint venture", "Joint Venture"), ("joint filing", "Joint Venture"),
    ("non-compete", "Non-Compete"), ("non_compete", "Non-Compete"), ("noncompetition", "Non-Compete"),
    ("affiliate", "Affiliate"), ("franchise", "Franchise"), ("sponsorship", "Sponsorship"),
    ("endorsement", "Endorsement"), ("collaboration", "Collaboration"), ("cooperation", "Collaboration"),
    ("outsourcing", "Outsourcing"), ("consulting", "Consulting"), ("reseller", "Reseller"),
    ("distribut", "Distributor"),        # matches distributor AND distribution
    ("hosting", "Hosting"), ("development", "Development"), ("manufactur", "Manufacturing"),
    ("supply", "Supply"), ("transportation", "Transportation"), ("promotion", "Promotion"),
    ("marketing", "Marketing"), ("agency", "Agency"), ("maintenance", "Maintenance"),
    ("intellectual property", "IP"), ("license", "License"), ("licensing", "License"),
    ("service", "Service"),              # very general, so it goes LAST
]


def parse_contract_type(title: str) -> str:
    """Map a contract title to one of ~25 canonical types (or "Other").

    We search the WHOLE title: cutting it at "-" first would turn
    "Co-Branding" into "Branding" and the co-branding rule could never match.
    """
    title = title.lower()                # lower-case once, before the loop
    for pattern, contract_type in CONTRACT_TYPE_RULES:
        if pattern in title:
            return contract_type         # first match wins
    return "Other"                       # loop finished, nothing matched

def parse_date(raw) -> date | None:
    """Your function from Lesson 6. Also handles pandas NaN (empty cells)."""
    if pd.isna(raw):          # NaN / None -> no date   (I added this line for pandas)
        return None
    else:
        if(not isinstance(raw, str)):
                return None
        else:
            try:
                return datetime.strptime(raw, "%m/%d/%y").date()
            except ValueError:
                return None
    # TODO (YOU DO): paste the rest of your Lesson 6 version here.


def parse_parties(raw) -> list[str]:
    """'Rogers Cable ("Rogers"); EuroMedia ("Licensor")' -> ['Rogers Cable ("Rogers")', 'EuroMedia ("Licensor")']"""
    if pd.isna(raw):
        return []
    # Split on ";" , strip spaces, drop empty pieces.
    return [p.strip() for p in str(raw).split(";") if p.strip()]


# ---------------------------------------------------------------------------
# Step 2 — read the two sources
# ---------------------------------------------------------------------------

def read_csv_metadata(csv_path: Path) -> dict[str, dict]:
    """Return {join_key: {"sign_date": ..., "parties": [...]}} from master_clauses.csv."""
    df = pd.read_csv(csv_path)
    metadata = {}
    for _, row in df.iterrows():
        key = make_key(Path(row["Filename"]).stem)      # .stem drops ".pdf"
        metadata[key] = {
            "sign_date": parse_date(row["Agreement Date-Answer"]),
            "parties": parse_parties(row["Parties-Answer"]),
        }
    return metadata


def read_json_contracts(json_path: Path) -> list[dict]:
    """Return [{"title": ..., "text": ...}, ...] from CUAD_v1.json (SQuAD format).

    Shape of the file:
      {"data": [ {"title": "...", "paragraphs": [ {"context": FULL TEXT, "qas": [...]} ]}, ... ]}
    Each contract has exactly ONE paragraph, whose "context" is the full contract text.
    """
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return [
        {"title": item["title"], "text": item["paragraphs"][0]["context"]}
        for item in payload["data"]
    ]


# ---------------------------------------------------------------------------
# Step 3 — join them into Contract objects
# ---------------------------------------------------------------------------

def load_cuad(settings: Settings) -> list[Contract]:
    # TODO (YOU DO, config.py): these two paths need a `cuad_dir` setting
    # pointing at the "CUAD_v1" FOLDER (not the csv file). Change config.py.
    metadata = read_csv_metadata(settings.cuad_dir / "master_clauses.csv")
    raw_contracts = read_json_contracts(settings.cuad_dir / "CUAD_v1.json")

    contracts = []
    for raw in raw_contracts:
        key = make_key(raw["title"])
        meta = metadata.get(key)            # .get -> None if not found (no crash)
        if meta is None:
            print(f"WARNING: no CSV row for {raw['title']!r}")
            meta = {"sign_date": None, "parties": []}

        contracts.append(
            Contract(
                doc_id=key,                 # stable: same input -> same id every run
                title=raw["title"],
                contract_type=parse_contract_type(raw["title"]),
                sign_date=meta["sign_date"],
                parties=meta["parties"],
                text=raw["text"],
            )
        )
    return contracts


# ---------------------------------------------------------------------------
# Run this file directly to check your work:
#   uv run python -m legal_contract_clause_search.loaders.cuad
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick self-checks for YOUR functions
    assert make_key("ACME Corp. - Supply Agreement .pdf") == "acmecorpsupplyagreementpdf"
    assert parse_contract_type("ACCURAYINC_09_01_2010-EX-10.31-DISTRIBUTOR AGREEMENT") == "Distributor"
    assert parse_contract_type("2ThemartComInc_19990826_EX-10.10_Co-Branding Agreement_ Agency Agreement") == "Co-Branding"
    assert parse_contract_type("XYZ_EX-10.1_Content License Agreement") == "License"
    assert parse_contract_type("Random Document") == "Other"
    assert parse_date("5/8/14") == date(2014, 5, 8)
    assert parse_date(float("nan")) is None
    print("helper checks passed")

    contracts = load_cuad(Settings())
    print("contracts loaded:", len(contracts))                       # expect 510
    with_date = sum(c.sign_date is not None for c in contracts)
    print(f"with sign_date: {with_date} ({with_date / len(contracts):.0%})")
    print(contracts[0].doc_id, "|", contracts[0].contract_type, "|", contracts[0].sign_date)
    print(contracts[0].parties)

    # Count contracts per type (a dict used as a counter)
    counts: dict[str, int] = {}
    for c in contracts:
        counts[c.contract_type] = counts.get(c.contract_type, 0) + 1
    for contract_type, n in sorted(counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"{n:4d}  {contract_type}")
