---
title: Contract Compliance Analyzer
emoji: 📜
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
python_version: "3.10"
---

# Contract Compliance Analyzer (Phase 1)

Research prototype for a master's thesis in Information Systems (King Abdulaziz
University, Saudi Arabia). It screens business-contract clauses against six
general Saudi laws (Personal Data Protection Law, Anti-Cyber Crime Law,
E-Commerce Law, Labor Law, Companies Law, Anti-Commercial Fraud Law) using:

- a knowledge graph (NetworkX) of laws, articles, required terms and subject
  words (Law → contains → Article → requires_term → Term; Law → has_subject →
  Subject), which drives retrieval (only the articles linked to the selected
  law are considered), the term check and the relevance gate; and a per-request contract graph (Contract → Clause →
  matched_to → Article) from which the clause traceability table is built,
- semantic similarity via sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
  ranked with a FAISS index,
- fixed, published rule-based reasoning (no generative LLM in this phase):
  combined = 0.4 × term coverage + 0.6 × mean similarity to the three closest
  articles; > 0.55 compliant, > 0.35 partial, otherwise non-compliant. A
  clause is "not applicable" to a law, and excluded from its figures, when it
  mentions none of the law's subject words (e.g. personal data for the PDPL,
  workers and wages for the Labor Law, consumers and online trade for the
  E-Commerce Law) and its best similarity is below 0.80, or when its best
  similarity is below 0.47 and it contains under half the required terms. Arabic text is normalized
  (diacritics, tatweel, alef/hamza, alef maqsura, ta marbuta) before term
  matching.

Every result is traceable: the interface and the PDF report list, for each
clause and law, the matched article, similarity, terms found, combined score,
verdict and the terms not mentioned.

## Privacy

Contract text is analyzed in memory for a single request: the contract graph
is created per request and discarded afterwards, and nothing is written to a
database or log. Uploaded files and generated PDFs are temporary and are
deleted automatically (Gradio cache: files older than one hour, checked every
15 minutes; generated reports: after 15 minutes).

This deployment is used to collect feedback from domain experts for the
thesis's expert-based evaluation chapter. The embedding model is multilingual,
so the same pipeline handles Arabic and English contracts and law text.

## Using the tool

1. Paste the contract text or upload a PDF, DOCX or TXT file.
2. Select the contract type and the laws to check it against. The tool does
   not choose the laws itself; the user decides which laws apply. If a
   selected law does not usually govern the stated type of contract (for
   example the Labor Law for a supply contract), the summary and the report
   show a note saying so; the law is still analyzed.
3. Run the analysis. The page shows a summary, the clause traceability table
   and a chart of compliance and risk per law, and offers a PDF report.

The PDF report is written in the language of the contract (Arabic or English)
and contains the overall result, the per-law table, the compliance and risk
chart, the clause traceability table, and samples of the findings and
recommendations.

## Law content and sourcing

The tool loads 39 articles from six Saudi laws, each in Arabic and in English
under its official article number (78 entries). Every article was checked
against the official Arabic text and the Bureau of Experts' official English
translation, in its current amended form; content is a close, condensed
restatement of each article, and the required terms are taken from the
article's own official wording.

| Law | Royal Decree | Articles |
|---|---|---|
| Personal Data Protection Law | M/19, 9/2/1443H (as amended 1444H) | 4, 5, 8, 11, 18, 19, 20, 29 |
| Anti-Cyber Crime Law | M/17, 8/3/1428H | 2, 3, 5, 13 |
| E-Commerce Law | M/126, 7/11/1440H | 5, 6, 7, 8, 13, 14 |
| Labor Law | M/51, 23/8/1426H (as amended up to M/44, 8/2/1446H) | 3, 51, 52, 53, 55, 75, 76, 77, 84, 109 |
| Companies Law | M/132, 1/12/1443H | 4, 8, 9, 12, 35, 58, 61 |
| Anti-Commercial Fraud Law | M/19, 23/4/1429H | 1, 2, 16, 22 |

Labor Law articles follow the text as amended in 1446H (for example Article
75: 30 days' notice by the worker, 60 days by the employer, for monthly-paid
indefinite-term contracts).

Per Chapter 2 of the thesis, Arabic remains the sole legally binding text of
Saudi legislation in any case, including for these laws that do have an
official English translation -- these translations are each marked "for
guidance"/governed by the Arabic text by their own publishing body.

## Known limitations of this deployment

- The loaded law content is a **selected set of 39 articles** (see above),
  not the complete legal texts; a clause may relate to an article that is not
  loaded. Numeric requirements (e.g. a 60-day notice period) are checked only
  through term and semantic matching, not as numeric rules.
- The exploratory ML risk classifier described in the thesis is **not
  trained** here; only the rule-based result is shown.
- The thresholds (0.55 / 0.35, 0.47 and 0.80 for relevance) and the subject
  word lists are fixed heuristics, not calibrated on labelled data.
- The contract type is stated by the user and recorded in the report; the
  tool does not classify the contract itself. Matching is by meaning, so a
  clause can be related to an article that does not legally govern that kind
  of contract (e.g. a termination clause in a business-to-business services
  contract matched to Labor Law Article 75). The subject gate and the scope
  note reduce this but are lexical and advisory. Users should select only the
  laws that apply to their contract type.
- Scanned (image-only) PDFs cannot be read; the interface reports this instead
  of analyzing other text.
- Output is an automated screening signal, not a legal determination.
- Each article has a limited set of required terms, so a well-drafted clause
  that uses different wording may be rated "Partial" rather than "Compliant".
