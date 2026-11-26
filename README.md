# Symbolic Clinical Protocol Document Question Answering System (`AID201A`)

**Student Name:** S Anjali  
**Register Number:** `25ETAI410120`  
**Course:** `AID201A` — Artificial Intelligence (3rd Semester B.Tech. CSE, 2025 Batch)  
**Institution:** Faculty of Engineering & Technology, Ramaiah University of Applied Sciences (RUAS)

---

## 1. Architecture Summary

This submission implements a deterministic symbolic Document Question Answering (DQA) agent designed for hospital clinical facility, biomedical device, and diagnostic protocol registers without NLP, statistical embeddings, or RAG:
- **Knowledge Representation**: Relational Proposition Frames (`B01`–`B22`) with dual hash-index maps ($I_{SP}, I_P$) and an undirected Semantic Graph.
- **Inference & Search**:
  1. **Six First-Order Production Rules (`P1`–`P6`)** deriving 27 implicit spatial, supervisory, divisional, and clinical screening prerequisite propositions (total closure = 49 propositions).
  2. **Backward-Chaining Goal Resolution & Forward Rule Materialization** with full step-by-step derivation provenance.
  3. **Breadth-First Search (BFS)** shortest relational chain discovery across clinical entities.
  4. **Closed-World Assumption (CWA)** deterministic abstention on unverified queries.

---

## 2. How to Run

```bash
python docqa.py clinical_protocol_kb.csv --question "Where is the Siemens 3T MRI Scanner located?"
python evaluate.py
```
