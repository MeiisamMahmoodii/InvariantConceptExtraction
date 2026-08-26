"""Create controlled conversational and reordered realizations for C-test facts only."""

import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];FACTS=ROOT/"data"/"three_domain_natural_rewrite"/"factual_c_matrix.csv";OUT=ROOT/"data"/"extra_heldout_surface_families";REPORT=ROOT/"Report"/"extra_heldout_surface_validation.json"


def render(relation,s,v,family):
    s=s[:1].upper()+s[1:]
    templates={
        "capital_of":{"conversational":f"If someone asks about {s}, you would say that its capital is {v}.","reordered":f"{v} is the city designated as the capital of {s}."},
        "continent_of":{"conversational":f"Thinking geographically, {s} is in {v}.","reordered":f"{v} is the continent in which {s} is located."},
        "currency_of":{"conversational":f"When you are in {s}, you pay with the {v}.","reordered":f"The {v} is the currency used by {s}."},
        "atomic_number_of":{"conversational":f"For {s}, the number to remember is {v}: its atomic number.","reordered":f"Atomic number {v} is assigned to {s}."},
        "period_of":{"conversational":f"On the periodic table, {s} sits in period {v}.","reordered":f"Period {v} of the periodic table contains {s}."},
        "chemical_symbol_of":{"conversational":f"Chemists shorten {s} to {v}.","reordered":f"{v} is the symbol by which {s} is denoted."},
        "author_of":{"conversational":f"If you are crediting {s}, the author is {v}.","reordered":f"{s} is credited to {v} as its author."},
        "country_of_origin":{"conversational":f"People trace {s} back to {v}; that is its country of origin.","reordered":f"{s} is identified as originating in {v}."},
        "original_language":{"conversational":f"{s} first appeared in {v}.","reordered":f"{v} is the language in which {s} was originally written."},
    }
    return templates[relation][family]


def main():
    with FACTS.open(newline="",encoding="utf-8") as f:facts=[r for r in csv.DictReader(f) if r["C_split"]=="C_test"]
    rows=[]
    for fact in facts:
        for family in ("conversational","reordered"):
            text=render(fact["C_relation"],fact["C_subject_label"],fact["C_value_label"],family);rows.append({**fact,"example_id":f"{fact['fact_id']}-{family}-v1","S_family":family,"S_variant":"v1","S_split":"S_test","text":text,"generator":"code/build_extra_heldout_surface_families.py","template_version":"extra-heldout-v1","generation_seed":"20260826"})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"controlled_surface_dataset.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={"C_test_facts":len(facts),"surface_rows":len(rows),"families":{"conversational":len(facts),"reordered":len(facts)},"subject_or_value_missing":sum(r["C_subject_label"].casefold() not in r["text"].casefold() or r["C_value_label"].casefold() not in r["text"].casefold() for r in rows),"training_performed":False}
    REPORT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report,indent=2))
    if report["subject_or_value_missing"]:raise SystemExit("surface validation failed")


if __name__=="__main__":main()
