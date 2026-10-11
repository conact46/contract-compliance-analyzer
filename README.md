---
title: Contract Compliance Analyzer
emoji: 📜
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
python_version: "3.10"
---

# Contract Compliance Analyzer (Phase 1)

Research prototype for a master's thesis in Information Systems (King Abdulaziz
University, Saudi Arabia). It screens business-contract clauses against six
general Saudi laws (Personal Data Protection Law, Anti-Cyber Crime Law,
E-Commerce Law, Labor Law, Companies Law, Anti-Fraud Law) using:

- a knowledge graph of laws, articles, contracts and clauses (NetworkX),
- semantic similarity via sentence embeddings (`all-MiniLM-L6-v2`) and FAISS,
- fixed, published rule-based reasoning (no generative LLM in this phase).

This deployment is used to collect feedback from domain experts (real
contracts, real judgments) for the thesis's expert-based evaluation chapter.
It is a direct, cleaned-up port of the research notebook
`contract_analyzer_phase4.ipynb`.

## Known limitations of this deployment

- The loaded law content is a **small set of illustrative example articles**
  per law (not the complete official legal texts). Replace `SAMPLE_LAWS` in
  `app.py` with a full corpus before relying on results for anything beyond
  demonstrating the pipeline.
- The exploratory ML risk classifier described in the thesis is **not
  trained** here; only the rule-based result is shown.
- Output is an automated screening signal, not a legal determination.
