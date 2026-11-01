# Writing the gold questions

The ablation table is only as credible as these questions. Aim for about 80.

## Target mix

| Type | Count | What it tests |
|---|---|---|
| factual | 20 | Policy and measures stated in text (speeches, Finance Acts) |
| numeric | 20 | One figure from a table or speech; needs exact numbers |
| comparison | 15 | Two or more figures, across votes or years; needs SQL or careful reading |
| cross_lingual | 10 | Question language differs from the evidence document's language |
| unanswerable | 15 | Outside the corpus; the correct answer is a refusal |

About half the questions should be in Swahili, spread across all types. Mark roughly a quarter
as `"split": "dev"`. Tune prompts only on dev; report test-split numbers in the README.

## Rules that keep the numbers honest

1. **Find evidence by reading the PDF, not by asking the copilot.** If gold pages come from
   `bajeti search`, the set only contains questions the system already gets right.
   Open the document, find the passage or table, and note the printed page numbers.
2. **Pages are PDF page numbers** (1 = first page of the file), which is what citations use.
   If the printed page number differs, use the PDF's.
3. **Numbers in full TZS.** "TZS 980.5 billion" is written as `980500000000`. The scorer accepts
   any unit in the answer, within 0.5%.
4. **Keywords are the minimum a correct answer must say**, such as the vote name or the
   measure. Keep them short and unambiguous; avoid words that could be phrased differently.
5. **Unanswerable questions should be tempting**: plausible budget questions whose answer
   isn't in these documents (other years, other countries, details the documents never
   give), not nonsense.
6. **Write each question as a user would ask it**, without copying document wording, so
   retrieval is actually tested.
7. **Never reuse the demo's example questions** (in `src/bajeti/app.py`); a test enforces it.

## Workflow

```bash
bajeti gold-check                                  # schema, coverage, drafts, unknown doc_ids
bajeti ablate --retrieval-only --embedder hashing --reranker overlap  # smoke test, any laptop
bajeti ablate --retrieval-only                     # real retrieval metrics: needs .[embed], CPU ok
bajeti eval --split dev --limit 10                 # quick check against the live model
bajeti ablate --split test                         # the real run, on the GPU
```

Records in `eval/results/<config>/records.jsonl` show, for every miss, whether the gold
evidence was never retrieved, retrieved but ranked out of the top k, or retrieved but answered
wrongly. Use them for the error-analysis paragraph in the README.
