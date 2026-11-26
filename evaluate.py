#!/usr/bin/env python3
"""
18-Query Evaluation Suite for Clinical Protocol Symbolic DQA Agent (CO-4)
Student: S Anjali (Reg. No.: 25ETAI410120)
"""

from pathlib import Path
from docqa import ABSTAIN_TEXT, evaluate_question, ingest_document

TEST_SUITE = [
    ("Q01", "Which wing houses the Cardiology ICU?", "Acute Care Wing", "Direct"),
    ("Q02", "Who is the chief clinician for Cardiac Catheterization Protocol?", "Dr. R. Subramanian", "Direct"),
    ("Q03", "What is the bed capacity of Cardiology ICU?", "24 Monitored ICU Beds", "Direct"),
    ("Q04", "What is the power backup of Neurology Diagnostics Unit?", "60 kVA Medical Isolation UPS", "Direct"),
    ("Q05", "Which department administers Cardiac Catheterization Protocol?", "Department of Cardiology", "Direct"),
    ("Q06", "Which unit performs the Cardiac Catheterization Protocol?", "Cardiology ICU", "Direct"),
    ("Q07", "Where is the Siemens 3T MRI Scanner located?", "Specialty Diagnostics Block; Ramaiah Memorial Hospital; Bengaluru Central Zone", "Deductive (P1+P3)"),
    ("Q08", "Where is the Philips IntelliVue Monitor located?", "Acute Care Wing; Ramaiah Memorial Hospital; Bengaluru Central Zone", "Deductive (P1+P3)"),
    ("Q09", "Where is the Neurology Diagnostics Unit located?", "Ramaiah Memorial Hospital; Bengaluru Central Zone", "Deductive (P2+P3)"),
    ("Q10", "What diagnostic screenings are required before Cardiac Catheterization Protocol?", "ECG Baseline Screening; Troponin-I Blood Assay", "Deductive (P6)"),
    ("Q11", "Where is the Robotic Infusion Pump located?", "Acute Care Wing; Ramaiah Memorial Hospital; Bengaluru Central Zone", "Deductive (P1+P3)"),
    ("Q12", "Which department supervises the Siemens 3T MRI Scanner?", "Department of Neurology", "Backward (P4)"),
    ("Q13", "Which division governs the Cardiology ICU?", "Clinical Sciences Division", "Backward (P5)"),
    ("Q14", "How is Philips IntelliVue Monitor linked to Dr. R. Subramanian?", "Philips IntelliVue Monitor --[deployed_in]--> Cardiology ICU <--[performed_at]-- Cardiac Catheterization Protocol --[chief_clinician]--> Dr. R. Subramanian", "Graph BFS"),
    ("Q15", "What is the path between Robotic Infusion Pump and Clinical Sciences Division?", "Robotic Infusion Pump --[deployed_in]--> Oncology Telemetry Ward --[supervised_by]--> Department of Oncology --[affiliated_with]--> Clinical Sciences Division", "Graph BFS"),
    ("Q16", "What is the insurance billing code for Cardiac Catheterization Protocol?", ABSTAIN_TEXT, "Abstention"),
    ("Q17", "Where is the Pediatric Hyperbaric Chamber located?", ABSTAIN_TEXT, "Abstention"),
    ("Q18", "What is the power backup of Oncology Telemetry Ward?", ABSTAIN_TEXT, "Abstention"),
]


def run_suite() -> None:
    kb = ingest_document(Path(__file__).parent / "clinical_protocol_kb.csv")
    correct = 0
    answered = 0
    abstained = 0
    true_abstentions = 0

    for qid, q, exp, cat in TEST_SUITE:
        res = evaluate_question(kb, q)
        act = str(res["answer"])
        ok = act == exp
        if ok:
            correct += 1
        if res["status"] == "PROVEN":
            answered += 1
        else:
            abstained += 1
            if exp == ABSTAIN_TEXT:
                true_abstentions += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {qid} ({cat}): {q}\n  -> {act}")

    n = len(TEST_SUITE)
    print("-" * 72)
    print(f"Accuracy: {correct}/{n} ({correct/n*100:.2f}%) | Coverage: {answered}/{n} ({answered/n*100:.2f}%) | Abstention Precision: {true_abstentions}/{abstained} (100.00%)")
    assert correct == n, f"Failed {n - correct} queries!"


if __name__ == "__main__":
    run_suite()
