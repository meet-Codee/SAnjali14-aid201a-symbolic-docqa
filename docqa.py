#!/usr/bin/env python3
"""
Symbolic Knowledge-Graph & Backward/Forward Inference Document QA Agent (AID201A)
Student: S Anjali (Reg. No.: 25ETAI410120)
Ramaiah University of Applied Sciences (RUAS)

Architectural Highlights:
- Symbolic Frame & Relational Triple Knowledge Base from structured documents (.csv, .json, .txt)
- Six First-Order Production Rules (P1-P6) with Backward-Chaining Goal Proof & Forward Closure
- Undirected Semantic Graph Shortest-Chain Search (BFS)
- Zero NLP, Zero Embeddings, Zero RAG, Python 3.10+ Standard Library Only
"""

from __future__ import annotations
import argparse
import csv
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ABSTAIN_TEXT = "ABSTAIN: Requested proposition cannot be proven from the closed-world Knowledge Base."

PREDICATE_LEXICON: Dict[str, Tuple[str, ...]] = {
    "housed_in": ("which wing houses", "housed in", "wing houses", "housed"),
    "located_in": ("where is", "located in", "located", "location", "which hospital"),
    "deployed_in": ("deployed in", "which unit has", "installed in"),
    "supervised_by": ("which department supervises", "supervised by", "supervises", "managed by"),
    "under_division": ("which division governs", "governs", "under division", "division oversees", "clinical division"),
    "affiliated_with": ("affiliated with", "part of division"),
    "administered_by": ("which department administers", "administered by", "administers"),
    "chief_clinician": ("chief clinician", "who heads", "lead physician", "who leads"),
    "requires_screening": ("diagnostic screenings", "required before", "screening required", "screenings", "prerequisite screening"),
    "performed_at": ("performed at", "which unit performs", "clinical venue"),
    "bed_capacity": ("bed capacity", "how many beds", "monitored beds"),
    "power_backup": ("power backup", "isolation ups", "backup rating", "ups"),
}

GRAPH_PATH_CUES: Tuple[str, ...] = ("linked to", "connected to", "path between", "relationship between", "how is")

# Production Rules P1-P6: (rule_id, head_pred, premise_1, premise_2, logical_form)
PRODUCTION_RULES: List[Tuple[str, str, str, str, str]] = [
    ("P1", "located_in", "deployed_in", "housed_in", "deployed_in(D,U) ^ housed_in(U,W) => located_in(D,W)"),
    ("P2", "located_in", "housed_in", "located_in", "housed_in(U,W) ^ located_in(W,H) => located_in(U,H)"),
    ("P3", "located_in", "located_in", "located_in", "located_in(X,Y) ^ located_in(Y,Z) => located_in(X,Z)"),
    ("P4", "supervised_by", "deployed_in", "supervised_by", "deployed_in(D,U) ^ supervised_by(U,Dept) => supervised_by(D,Dept)"),
    ("P5", "under_division", "supervised_by", "affiliated_with", "supervised_by(X,Dept) ^ affiliated_with(Dept,Div) => under_division(X,Div)"),
    ("P6", "requires_screening", "requires_screening", "requires_screening", "requires_screening(A,B) ^ requires_screening(B,C) => requires_screening(A,C)"),
]


def canonical_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


@dataclass(frozen=True)
class Proposition:
    prop_id: str
    subject: str
    predicate: str
    obj: str
    provenance: str
    derived: bool = False
    hop_depth: int = 1


@dataclass
class ParsedQuery:
    raw: str
    norm: str
    intent: str  # "GOAL_RESOLUTION" or "GRAPH_CHAIN"
    predicate: Optional[str]
    bound_entities: List[str] = field(default_factory=list)


class SymbolicKB:
    def __init__(self) -> None:
        self.base_facts: List[Proposition] = []
        self.all_facts: List[Proposition] = []
        self.index_sp: Dict[Tuple[str, str], List[Proposition]] = {}
        self.index_p: Dict[str, List[Proposition]] = {}
        self.entity_catalog: Dict[str, str] = {}
        self.adj: Dict[str, List[Tuple[str, str, str]]] = {}

    def assert_proposition(self, prop: Proposition) -> bool:
        sk, pk, ok = canonical_key(prop.subject), prop.predicate.strip().lower(), canonical_key(prop.obj)
        bucket = self.index_sp.get((sk, pk), [])
        if any(canonical_key(existing.obj) == ok for existing in bucket):
            return False
        if not prop.derived:
            self.base_facts.append(prop)
        self.all_facts.append(prop)
        self.index_sp.setdefault((sk, pk), []).append(prop)
        self.index_p.setdefault(pk, []).append(prop)
        self.entity_catalog[sk] = prop.subject
        self.entity_catalog[ok] = prop.obj
        self.adj.setdefault(sk, []).append((f"--[{pk}]-->", ok, prop.prop_id))
        self.adj.setdefault(ok, []).append((f"<--[{pk}]--", sk, prop.prop_id))
        return True

    def materialize_rules(self) -> int:
        new_count = 0
        changed = True
        while changed:
            changed = False
            for rid, head_p, p1, p2, _ in PRODUCTION_RULES:
                for f1 in list(self.index_p.get(p1, [])):
                    join_key = canonical_key(f1.obj)
                    for f2 in list(self.index_sp.get((join_key, p2), [])):
                        seq = len(self.all_facts) - len(self.base_facts) + 1
                        pid = f"I{seq:02d}"
                        proof = f"{rid}({f1.prop_id} [{f1.provenance}] & {f2.prop_id} [{f2.provenance}])"
                        candidate = Proposition(
                            prop_id=pid,
                            subject=f1.subject,
                            predicate=head_p,
                            obj=f2.obj,
                            provenance=proof,
                            derived=True,
                            hop_depth=f1.hop_depth + f2.hop_depth,
                        )
                        if self.assert_proposition(candidate):
                            new_count += 1
                            changed = True
        return new_count

    def resolve_goal(self, subject: str, predicate: str) -> List[Proposition]:
        hits = list(self.index_sp.get((canonical_key(subject), predicate), []))
        hits.sort(key=lambda p: (p.hop_depth, p.prop_id))
        return hits


def ingest_document(doc_path: Path) -> SymbolicKB:
    kb = SymbolicKB()
    ext = doc_path.suffix.lower()
    records: List[Tuple[str, str, str, str]] = []

    if ext == ".csv":
        with doc_path.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            req = {"subject", "relation", "object"}
            if not reader.fieldnames or not req.issubset({h.strip().lower() for h in reader.fieldnames}):
                raise ValueError(f"Schema violation in {doc_path.name}: expected subject, relation, object headers.")
            for i, row in enumerate(reader, start=1):
                s = (row.get("subject") or "").strip()
                r = (row.get("relation") or "").strip().lower()
                o = (row.get("object") or "").strip()
                if not s or not r or not o:
                    raise ValueError(f"Incomplete triple at {doc_path.name}:line {i}")
                records.append((s, r, o, f"{doc_path.name}:line {i}"))
    elif ext == ".json":
        data = json.loads(doc_path.read_text(encoding="utf-8"))
        entries = data["facts"] if isinstance(data, dict) and "facts" in data else data
        for i, item in enumerate(entries, start=1):
            s = str(item.get("subject", "")).strip()
            r = str(item.get("relation", "")).strip().lower()
            o = str(item.get("object", "")).strip()
            if not s or not r or not o:
                raise ValueError(f"Incomplete JSON entry at {doc_path.name}:entry {i}")
            records.append((s, r, o, f"{doc_path.name}:entry {i}"))
    else:
        active_subj = None
        for i, raw_line in enumerate(doc_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                active_subj = line[1:-1].strip()
            elif ":" in line and active_subj:
                k, v = [x.strip() for x in line.split(":", 1)]
                records.append((active_subj, k.lower().replace(" ", "_"), v, f"{doc_path.name}:line {i}"))

    for idx, (s, r, o, src) in enumerate(records, start=1):
        kb.assert_proposition(Proposition(prop_id=f"B{idx:02d}", subject=s, predicate=r, obj=o, provenance=src, derived=False, hop_depth=1))
    kb.materialize_rules()
    return kb


def parse_user_query(kb: SymbolicKB, question: str) -> ParsedQuery:
    nq = canonical_key(question)
    padded = f" {nq} "
    spans: List[Tuple[int, int, str]] = []
    for k_ent, label in sorted(kb.entity_catalog.items(), key=lambda item: len(item[0]), reverse=True):
        idx = padded.find(f" {k_ent} ")
        if idx != -1:
            s_pos, e_pos = idx, idx + len(k_ent)
            if not any(not (e_pos <= a or s_pos >= b) for a, b, _ in spans):
                spans.append((s_pos, e_pos, label))
    spans.sort(key=lambda x: x[0])
    entities = [x[2] for x in spans]

    if len(entities) >= 2 and any(cue in nq for cue in GRAPH_PATH_CUES):
        return ParsedQuery(raw=question, norm=nq, intent="GRAPH_CHAIN", predicate="graph_path", bound_entities=entities[:2])

    chosen_pred: Optional[str] = None
    max_weight = 0
    for pred, phrases in PREDICATE_LEXICON.items():
        for ph in phrases:
            nph = canonical_key(ph)
            if f" {nph} " in padded:
                w = len(nph.split()) * 12 + len(nph)
                if w > max_weight:
                    max_weight = w
                    chosen_pred = pred

    return ParsedQuery(raw=question, norm=nq, intent="GOAL_RESOLUTION", predicate=chosen_pred, bound_entities=entities)


def search_entity_chain(kb: SymbolicKB, src_ent: str, dst_ent: str) -> Dict[str, object]:
    sk, dk = canonical_key(src_ent), canonical_key(dst_ent)
    if sk not in kb.adj or dk not in kb.adj:
        return {"status": "ABSTAIN", "answer": ABSTAIN_TEXT, "proof": "Entity missing from semantic graph."}

    q: deque[Tuple[str, List[str], List[str]]] = deque([(sk, [kb.entity_catalog[sk]], [])])
    seen: Set[str] = {sk}
    while q:
        curr, path_tokens, pids = q.popleft()
        if curr == dk:
            return {
                "status": "PROVEN",
                "mechanism": "BFS_SHORTEST_CHAIN",
                "answer": " ".join(path_tokens),
                "proof": f"Graph BFS ({len(pids)} hops): " + " -> ".join(pids),
            }
        for rel_arrow, nxt, pid in kb.adj.get(curr, []):
            if pid.startswith("I"):
                continue
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, path_tokens + [rel_arrow, kb.entity_catalog[nxt]], pids + [pid]))

    return {"status": "ABSTAIN", "answer": ABSTAIN_TEXT, "proof": "No relational path connects the entities."}


def evaluate_question(kb: SymbolicKB, question: str) -> Dict[str, object]:
    t_start = time.perf_counter()
    pq = parse_user_query(kb, question)

    if pq.intent == "GRAPH_CHAIN" and len(pq.bound_entities) >= 2:
        out = search_entity_chain(kb, pq.bound_entities[0], pq.bound_entities[1])
        out["time_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)
        return out

    if not pq.predicate or not pq.bound_entities:
        return {
            "status": "ABSTAIN",
            "mechanism": "CLOSED_WORLD_ABSTENTION",
            "answer": ABSTAIN_TEXT,
            "proof": "Unmapped predicate cue or unknown subject entity.",
            "time_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
        }

    proofs = kb.resolve_goal(pq.bound_entities[0], pq.predicate)
    if not proofs:
        return {
            "status": "ABSTAIN",
            "mechanism": "CLOSED_WORLD_ABSTENTION",
            "answer": ABSTAIN_TEXT,
            "proof": f"Goal ({pq.bound_entities[0]}, {pq.predicate}, ?X) failed unification.",
            "time_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
        }

    ans_str = "; ".join(p.obj for p in proofs)
    proof_str = "; ".join(f"{p.prop_id}: {p.provenance}" for p in proofs)
    mech = "RULE_DEDUCTION" if any(p.derived for p in proofs) else "BASE_FACT_LOOKUP"
    return {
        "status": "PROVEN",
        "mechanism": mech,
        "answer": ans_str,
        "proof": proof_str,
        "time_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Symbolic Clinical Protocol Document QA Agent (AID201A)")
    ap.add_argument("document", type=Path, help="Input structured document (.csv, .json, .txt)")
    ap.add_argument("--question", "-q", type=str, help="Query string")
    args = ap.parse_args()

    kb = ingest_document(args.document)
    print(f"[KB READY] Base Propositions={len(kb.base_facts)} | Inferred={len(kb.all_facts)-len(kb.base_facts)} | Total={len(kb.all_facts)}")
    if args.question:
        r = evaluate_question(kb, args.question)
        print(f"[MECHANISM] {r.get('mechanism')}")
        print(f"[ANSWER]    {r['answer']}")
        print(f"[PROOF]     {r.get('proof')}")


if __name__ == "__main__":
    main()
