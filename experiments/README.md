# Experiment results

A curated results snapshot through **November 7, 2026**. Start with the [audit against the previous GitHub revision](AUDIT_2026-11-06.md), then follow the five chapters below. [Machine-readable catalogue](catalog.json).

| Chapter | What to look at | Status |
|---|---|---|
| [Red+circle neutral answers](red-circle-composition/README.md) | Base versus B09: fruit, badge, toy, sticker and eight other questions; exact answers and all 504 responses | Evaluation complete; original training incomplete |
| [Red+rectangle neutral answers](red-rectangle-composition/README.md) | Base versus selected B07: all 504 replies, fruit/badge/toy/sticker examples, and a negative explicit-composition result | Evaluation complete; original training incomplete |
| [Shape training](shapes/README.md) | Red+circle through B09, red+rectangle through B12, and complete rectangle-only B16 trajectory | Paired runs partial; rectangle-only complete |
| [Recursion and cross-trait transfer](recursive-and-transfer/README.md) | Recursive versus frozen teachers, fruit/shape assays, rare traits, mixed traits, self-distillation and diagnostics | Completed results plus explicitly labeled partial runs |
| [Transfer matrices and interference](transfer-and-interference/README.md) | Multi-model/domain sweeps, metric analyses, learning dynamics, 100 color pairs, multi-trait/prefix studies and completed null results | Results and reanalyses; overlapping endpoints identified |

## Recent findings to start with

- **Red+circle neutral comparison:** strict explicit red-and-round descriptions of the same object appear in 70/240 B09 sampled replies versus 0/240 base replies. These are exploratory coded outputs, with substantial and unequal truncation at 128 tokens; fruit names alone do not count as color-and-shape evidence.
- **Red+rectangle neutral comparison:** B07 was selected for its strong separate trait assays (399/400 red; 314/400 rectangle). In the completed neutral evaluation, red appears in 233/240 B07 sampled replies versus 1/240 base replies, but explicit same-object red-and-rectangle combinations occur in 0/240 for both. Neither condition produces one in its 12 greedy replies. This is a selected-checkpoint result with a material 128-token cap; [read the paired answers and limits](red-rectangle-composition/README.md).
- **Red-only fruit transfer:** strawberry choices shift from 40/400 base to 374/400 B8 replies. This uses a different, red-only student and has no matched neutral-SFT control.
- **Useful negative outcomes:** completed dolphin recursion yields only 1/400 target mentions at B20; an earlier Qwen3 red-number SFT study has 0/200 strict-red development replies at all seven evaluated checkpoints. These are completed results, not failed setup attempts.

## Reading the evidence

A sampled mention, an affirmative choice, a target-prefix probability, and a same-object composition code are different measurements. Each chapter states its definitions. Separate trait rates do not establish joint composition, and single-seed trajectories do not establish causal or population-wide effects.

When a chapter says **Metric 1**, it means the mean of per-prompt cosines, with neutral-probability weighting and across-trait centering. Historical flattened-cosine results must remain explicitly historical. Other metrics retain their own labels. Older analysis scripts may retain historical definitions; consult each result’s methodology before reusing them.

The catalogue contains 32 entries spanning experiment families and analyses. It is not a count of independent campaigns: for example, the five-model SFT comparison is a subset of the atlas, and several diagnostic studies reuse its outputs.

## Data and reproduction scope

Compact tables, selected plots, raw neutral-comparison answers, and source hashes are included. Checkpoints and the complete original training infrastructure remain outside ordinary Git. Replotting or verification instructions are supplied where the included data support them. See the [snapshot manifest](manifest.json) and chapter source manifests for file hashes.

The older paper-reproduction documentation remains in the [repository README](../README.md#original-paper-reproduction-documentation). This results snapshot does not change existing training implementations or relabel their historical protocols.
