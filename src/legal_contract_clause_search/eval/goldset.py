"""The answer key: CUAD's lawyer highlights + natural-language test questions.

For each contract and clause category, CUAD gives the exact character spans a
lawyer highlighted (`answer_start` + length of the answer text). A retrieved
chunk is RELEVANT if it overlaps one of those spans.

The questions are phrased the way a user would ask — deliberately NOT copying
the contract's wording (e.g. "can a party end the agreement early without
cause?" instead of "termination for convenience"), otherwise the test would
make the system look better than it is.
"""

import json
import re
from dataclasses import dataclass

from legal_contract_clause_search.config import Settings
from legal_contract_clause_search.loaders.cuad import make_key

CATEGORY_QUERIES: dict[str, list[str]] = {
    "Governing Law": [
        "which state's or country's law applies to this contract?",
        "what jurisdiction's rules are used to interpret the agreement?",
    ],
    "Non-Compete": [
        "is a party restricted from competing with the other?",
        "can the company work with rival businesses during or after the deal?",
    ],
    "Termination For Convenience": [
        "can a party end the agreement early without any cause?",
        "is either side allowed to cancel at will by giving advance notice?",
    ],
    "Exclusivity": [
        "does the deal give sole or exclusive rights to one party?",
        "is anyone the only permitted provider or seller in a territory?",
    ],
    "Change Of Control": [
        "what happens if one of the parties is acquired or merges with another company?",
        "rights triggered when ownership of a party changes hands",
    ],
    "Anti-Assignment": [
        "can a party transfer this contract to another company?",
        "is consent needed before handing the agreement over to someone else?",
    ],
    "Cap On Liability": [
        "what is the maximum amount a party can owe for damages?",
        "is there a ceiling on how much compensation can be claimed?",
    ],
    "Audit Rights": [
        "can a party inspect the other's books and records?",
        "who may check the accounts to verify payments are correct?",
    ],
    "Insurance": [
        "what insurance coverage must a party carry?",
        "required policies protecting against property damage or injury claims",
    ],
    "Renewal Term": [
        "does the contract extend itself automatically when it ends, and for how long?",
        "how long is each extension period after the initial term?",
    ],
}

_CATEGORY = re.compile(r'related to "(.+?)"')


@dataclass
class GoldItem:
    doc_id: str
    category: str
    spans: list[tuple[int, int]]


def load_gold(settings: Settings, doc_ids: set[str] | None = None) -> list[GoldItem]:
    payload = json.loads((settings.cuad_dir / "CUAD_v1.json").read_text(encoding="utf-8"))
    items = []
    for contract in payload["data"]:
        doc_id = make_key(contract["title"])
        if doc_ids is not None and doc_id not in doc_ids:
            continue
        for qa in contract["paragraphs"][0]["qas"]:
            m = _CATEGORY.search(qa["question"])
            if not m or m.group(1) not in CATEGORY_QUERIES or not qa["answers"]:
                continue
            spans = sorted({(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]})
            items.append(GoldItem(doc_id, m.group(1), spans))
    return items
