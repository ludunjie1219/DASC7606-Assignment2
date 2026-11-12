# Optional extension package (20 marks within the 100-mark total)

Keep the existing core package and checker unchanged. Select core `best.pt`
from the baseline, model experiment or optimizer experiment; it must remain
loadable by the provided GPT implementation. A combined or novel architecture
belongs in this separate extension package, not in the core `best.pt`.

## Scope and fair comparison

- Complete the three required core runs first. Extension runs do not substitute
  for them. There is no cap on the number of additional experiments.
- Use the same supplied training split, tokenizer, vocabulary and context length.
  Do not train on validation/hidden test data or use pretrained weights.
- Train extension runs from random initialization. Keep models within 20M
  parameters. Architecture changes outside the core loader family are permitted
  here when their code and loading/evaluation instructions are supplied.
- For claims of improvement, provide a comparison with a matched token budget
  and evaluation protocol. Report parameter count, processed tokens, precision,
  hardware and runtime; distinguish exploratory runs from the controlled comparison.
- Cite borrowed implementations and describe your changes. Do not submit
  credentials, personal data, training data, environments, caches or private tests.

## Files and upload

Create a directory containing these files at its root:

```text
extension/
  extension_report.pdf   # separate from the at-most-two-page main report
  README.md              # idea, file map, dependencies, exact train/evaluate commands
  code/                  # all extension code needed to reproduce the results
  configs/               # exact configurations
  results/               # logs, metrics, curves, fixed-prompt samples and budgets
```

Keep the extension report concise: explain the question, implementation,
controlled evidence and conclusions. Its pages do not count toward the main
report's two-page limit. Reference rather than duplicate core results, and make
the additional contribution explicit. A checkpoint is optional; if supplied,
include at most one `extension_best.pt` and document its loading procedure.

Package and perform the format-only check (Python standard library only):

```sh
python scripts/package_extension.py --directory extension \
  --student-id YOUR_STUDENT_ID --out YOUR_STUDENT_ID_extension.zip
python scripts/package_extension.py --check YOUR_STUDENT_ID_extension.zip
```

Upload **both** `YOUR_STUDENT_ID.zip` (core) and
`YOUR_STUDENT_ID_extension.zip` to the same Moodle assignment. Each compressed
file must be at most 100 MB; extension contents must be at most 200 MiB unpacked.
The checker checks packaging only, does not execute your code or load a checkpoint,
and does not grade the scientific quality of the extension. Extension material is
reviewed separately from the trusted core-model evaluator.
