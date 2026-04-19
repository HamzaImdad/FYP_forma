# FORMA — Final Year Project Reference Library

This directory contains the reference library compiled for the FORMA final
year project report (COMP1682, BSc Computer Science, University of
Greenwich). Every technical claim, method, and design decision in the
report should be backed by a source listed here.

The University of Greenwich uses the **Harvard** referencing style.
Every Harvard citation below has a matching BibTeX entry in
`bibliography.bib` so the report can be typeset in either Word
(cite-as-you-write) or LaTeX (BibTeX).

---

## Folder contents

| File                        | Purpose                                           |
|-----------------------------|---------------------------------------------------|
| `README.md`                 | This file — how to use the library                |
| `bibliography.bib`          | Full BibTeX database, one entry per source        |
| `citation_map.md`           | Maps each FORMA claim / report section to keys    |
| `01_pose_estimation.md`     | MediaPipe, OpenPose, MoveNet, pose benchmarks      |
| `02_joint_angle_form.md`    | Biomechanics of the 10 FORMA exercises            |
| `03_classifier_architectures.md` | RF, SVM, CNN-BiLSTM, attention, focal loss   |
| `04_ml_training_methodology.md`  | Data leakage, imbalance, augmentation, cosine LR |
| `05_realtime_system_design.md`   | WebSocket streaming, browser CV, latency limits |
| `06_hci_feedback_ux.md`          | Motor learning (KR/KP), cognitive load, UX      |
| `07_exercise_science.md`         | NSCA, ACSM, WHO, injury epidemiology            |
| `08_web_product_layer.md`        | JWT, RAG, SSE, gamification, SDT                |
| `09_design_ui.md`                | WCAG, dark mode, motion, typography, scrollytel |
| `10_ethics_se_considerations.md` | GDPR, FDA SaMD, ML bias, model cards            |
| `11_limitations.md`              | Monocular depth, occlusion, phone thermal, etc. |

---

## How each topic file is structured

Each `NN_topic.md` contains, for every source:

1. **Harvard citation** (Author, Year, Title, Venue)
2. **BibTeX key** — a stable identifier (e.g. `bazarevsky2020blazepose`)
3. **Venue** — journal or conference
4. **URL / DOI** — a verifiable pointer
5. **Annotation** — two to three sentences covering *what the source says*
   and *why it supports a specific FORMA claim*
6. **Suggested quote or statistic** — ready to drop into the report with
   proper attribution
7. **Pages** when available — for direct quotes

---

## How to cite in the report

### Harvard in-text (Word)

> MediaPipe BlazePose's 33-landmark topology extends COCO's 17 keypoints
> with additional hand and foot joints (Bazarevsky *et al.*, 2020).

Reference list:

> Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F.
> and Grundmann, M. (2020) 'BlazePose: On-device Real-time Body Pose
> Tracking', *arXiv preprint* arXiv:2006.10204.

### BibTeX (LaTeX)

```latex
\cite{bazarevsky2020blazepose}
% ...
\bibliographystyle{agsm}   % Harvard
\bibliography{reports/references/bibliography}
```

### When a claim needs multiple sources

`citation_map.md` lists every load-bearing claim in the report and the
bibliography keys that back it up. Scan `citation_map.md` first when
writing a new section.

---

## Rules enforced in this library

- **Peer-reviewed first.** Conference proceedings (CVPR, ICCV, ECCV,
  NeurIPS, CHI, ACL, ICML, ICLR), peer-reviewed journals (IEEE TPAMI,
  IJCV, IEEE TNNLS, Med Sci Sports Exerc, J Strength Cond Res, Br J
  Sports Med), authoritative textbooks (Schmidt & Lee, NSCA Essentials,
  Bringhurst), and standards documents (W3C WCAG, RFCs, ISO, GDPR).
- **Grey literature sparingly.** OpenAI docs, MediaPipe docs, NNG
  articles and OWASP cheat sheets are cited as *implementation
  references*, not as evidence for theoretical claims. Those always
  have a peer-reviewed backstop.
- **Verified, not fabricated.** Every source in `bibliography.bib` has
  been checked against arXiv, Google Scholar, Semantic Scholar, or the
  publisher's own record. If a DOI could not be verified, a stable URL
  is used instead.
- **Recency.** ML/CV sources prefer 2015+. Motor-learning and
  biomechanics classics (e.g. Schmidt 1984, Escamilla 2001) are kept
  when they remain canonical.

---

## Summary statistics

Counts refer to the number of BibTeX entries deposited under each topic
in `bibliography.bib`. A handful of entries are shared across topics
(e.g. `bazarevsky2020blazepose` is load-bearing in topics 1, 5 and 11),
so the de-duplicated total is lower than the sum of per-topic counts.

| Topic                                           | Entries |
|-------------------------------------------------|---------|
| 01. Pose estimation                             | 15      |
| 02. Joint-angle form evaluation (biomechanics)  | 27      |
| 03. Classifier architectures                    | 21      |
| 04. ML training methodology                     | 25      |
| 05. Real-time system design                     | 16      |
| 06. HCI / feedback UX                           | 20      |
| 07. Exercise science                            | 25      |
| 08. Web product layer                           | 26      |
| 09. Design / UI                                 | 20      |
| 10. Ethics / SE considerations                  | 24      |
| 11. Limitations                                 | 14      |
| **Total entries in `bibliography.bib` (deduped)** | **~190** |

Per-topic citation counts clear the 3-6-per-claim bar requested in the
brief. Gaps and under-sourced claims are listed at the bottom of
`citation_map.md`.

---

## When something's missing

If the report needs a claim that is not yet backed up in `citation_map.md`:

1. Add the claim to `citation_map.md` under the right section
2. Search for a source (Google Scholar / Semantic Scholar / arXiv)
3. Verify the source exists (open the paper page)
4. Add a BibTeX entry to `bibliography.bib`
5. Add an annotated entry to the relevant topic Markdown file

Never add a citation without a verifiable URL/DOI. External examiners
will spot-check references — a fabricated citation is a serious
integrity issue.
