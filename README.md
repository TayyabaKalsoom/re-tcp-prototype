# RE-TCP Prototype

A working demo of the Aspect-Based, Requirement-Driven Framework for Test
Case Prioritization in Component-Based Systems — for use in your defence
slides as prototype screenshots.

## 1. Setup (one-time)

```bash
# from inside the "prototype" folder
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

## 2. Run it

```bash
streamlit run app.py
```

This opens automatically in your browser at `http://localhost:8501`.
If it doesn't open automatically, copy that URL into your browser.

## 3. About the bundled data

The app now includes all **three real evaluation datasets** from your
manuscript, selectable from a dropdown at the top of the app:

- **DS1 (PROMISE)** — 625 requirements, public benchmark, with the
  dataset's real ground-truth FR/NFR labels. No paired test cases, so
  this dataset drives **Phase 1 only** — useful for showing the
  framework works at scale, not just on a handful of requirements.
- **DS3 (EPCS Integration)** — 12 requirements, real industrial data
  (Drummond Group audit procedure), paired with 12 real test cases.
  Full Phase 1 → Phase 2 → Evaluation pipeline.
- **DS4 (Manifests Web)** — 14 requirements, real industrial data,
  paired with 12 real test cases. Full pipeline, and this is the
  dataset where your manuscript reports the **statistically significant**
  APFD improvement (+0.059, p=0.029) — the Evaluation tab shows this
  result directly when DS4 is selected.

## 3b. What each tab demonstrates

- **Phase 1 — Requirement Prioritization**: loads requirements, runs topic
  modeling (LDA) to assign FR/NFR + topic, computes ourRank scores from
  adjustable weight sliders, shows the cosine similarity matrix, and outputs
  the final prioritized requirement list.
- **Phase 2 — Test Case Selection & Prioritization**: takes the top-N
  prioritized requirements, maps them to test cases (greedy heuristic,
  redundant/irrelevant cases excluded), clusters the selected cases with a
  from-scratch K-Medoids implementation, and outputs the final TC1...TCn
  ranked sequence.
- **Evaluation — APFD**: lets you mark which test cases detected a fault in
  a given run and computes the real APFD metric on your prioritized order.

## 4. Taking screenshots for your slides

1. Resize your browser window to roughly 1600×900 before capturing —
   this keeps every screenshot the same aspect ratio.
2. Capture **one tab at a time**. Suggested shots:
   - Phase 1: the ourRank-scored, sorted requirement table (scroll so the
     top ~8 rows are visible)
   - Phase 1: the cosine similarity heatmap
   - Phase 2: the final Rank TC table (TC1...TCn sequence)
   - Evaluation: the APFD metric card + defect report table
3. Windows: `Win + Shift + S` → drag a box around just the content area
   (skip the browser chrome/tabs bar for a cleaner look).
   Mac: `Cmd + Shift + 4`.
4. Save each screenshot as a PNG directly into a folder you'll pull from
   when building your PowerPoint slides.

## 5. Using your own data instead of the bundled sets

To swap in different data for any of the three slots, replace the matching
files in `data/` (`ds1_requirements.csv`, `ds3_requirements.csv` /
`ds3_test_cases.csv`, `ds4_requirements.csv` / `ds4_test_cases.csv`),
keeping the same column names:

- requirements file: `req_id, requirement_text, stakeholder_priority,
  change_frequency, historical_defects` (optionally `fr_nfr_given` if you
  have ground-truth FR/NFR labels, as DS1 does)
- test cases file: `tc_id, description, linked_req_id`

The app will pick up the new files automatically on next run — no code
changes needed for the same schema.
