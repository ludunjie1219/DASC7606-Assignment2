# Assignment 2 rubric - updated 13 November 2026

This update applies to both DASC7606 sections and supersedes the earlier 80/20
category description and one-final-attempt wording. Deadline: **22 November 2026,
23:59 HKT (UTC+8)**. Late cutoff and penalties are unchanged (see README).

| Component | Marks | Evidence assessed |
|---|---:|---|
| Part 1: Implementation | 20 | Correct explicit masked attention, MLP, residual block and generation; shapes, gradients, causality, sampling and public API compatibility. |
| Part 2: Training and inference | 20 | Completed unchanged baseline, valid checkpoint, reproducible configuration/logs, learning behaviour, validation and generated continuations. |
| Part 3: Controlled experiments | 20 | One model-side and one optimization-side change relative to baseline, equal token budgets, justified hypotheses, controlled comparisons and interpretation. |
| Part 4: Report writing | 20 | Clear and accurate methods, readable results/curves, thoughtful comparison, limitations and reproducibility; main report at most two pages. |
| Part 3 extension/exploration | 20 | Further investigation beyond the two required single-factor experiments, assessed below. |
| **Total** | **100** | **Core 80 + extension 20; not 120.** |

The extension is optional to attempt; without it the maximum is 80/100. It is not
extra credit above 100. Core experiment marks do not require a novel method.
The same experiment is not counted twice as both a required experiment and an
extension without a clearly identified additional contribution.

## Extension: four criteria, five marks each

1. **Motivation and understanding (5):** identify a meaningful question; explain
   the method and why it might help this task; cite relevant sources.
2. **Implementation (5):** a substantive, correct adaptation with clearly
   identified original work and a reproducible launch procedure.
3. **Experimental evidence (5):** appropriate controls, explicit compute/token
   budgets, configurations, logs and comparable evaluation/generation results.
4. **Analysis and insight (5):** explain results and trade-offs, limitations,
   failure modes and what can reasonably be concluded.

For each criterion, full marks require clear, correct and convincing evidence;
partial marks reflect incomplete or weakly supported work; no relevant evidence
earns zero. A named recent technique, more runs, a larger model or lower loss by
itself does not guarantee marks. A well-executed negative result can earn full
extension marks. Be explicit when an implementation is a simplified adaptation.

Possible topics include a Muon-style optimizer, a carefully scoped linear or
sparse attention adaptation (such as ideas from KDA/DSA), normalization or
positional-encoding changes, or a justified combination supported by ablations.
These are examples, not a required list or a requirement to implement a full
large-scale system. One well-investigated idea is sufficient.

**There are no fixed loss, cross-entropy (CE) or perplexity (PPL) thresholds for
grading, and these metrics are not converted into mark bands.** Assessment
focuses on implementation correctness, completion and reproducibility of the
required work, and the quality of the experimental design, evidence and analysis.
Report loss/PPL and representative generations to support your comparisons and
discussion. A lower loss alone does not guarantee a higher mark, and a particular
public-validation score does not guarantee or cap a grade. Strong, well-supported
analysis of a negative result can earn full marks for the relevant criterion.

Use the same evaluation protocol for comparisons, and distinguish sampled
mixed-precision trainer loss from full-split float32 evaluation. There is no
required numerical target that students must reach to qualify for marks.

## Colab T4 and BF16 clarification

Native hardware BF16 support is not required for grading. Completed Colab
T4/BF16 runs are accepted when they meet the stated configuration, data,
token-budget and evidence requirements. **No rerun on another GPU is required.**
Record the actual hardware, software and precision accurately. This does not
remove the normal implementation and experiment checks or change the baseline.

## Already started or submitted?

- The fixed data, tokenizer, baseline and three required run roles are unchanged.
- Compliant completed runs and valid v1.0.0 core ZIPs remain accepted; do not
  restart training merely to adopt this documentation update.
- Everyone may revise/resubmit before the deadline. If a submitted attempt is
  locked, contact dunjielu1219@connect.hku.hk for reopening.
- Add extension work as a **separate ZIP** using [EXTENSION.md](EXTENSION.md).
  Do not replace your required core ZIP with only an extension ZIP.
- A prior submitted core remains valid if you choose not to revise it. If you
  start a new attempt, finish by clicking Submit assignment and verifying its
  submitted status. Keep local copies of both packages.
