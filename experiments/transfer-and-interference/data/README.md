# Data guide

These files are compact derivatives for following the research. They do not replace full raw-generation archives, large checkpoints, or complete model-score caches.

| File | Contents and interpretation |
|---|---|
| [sweep_inventory.csv](sweep_inventory.csv) | All 35 primary atlas conditions and 12 variants; identifiers keep overlapping analyses traceable |
| [sweep_selected_matrices.json](sweep_selected_matrices.json) | Eight raw and double-centered matrices in the sweep figure; answer rows, teacher columns, nats |
| [sweep_figure_notes.json](sweep_figure_notes.json) | Definition and full-precision values of the within/between-group annotations |
| [cross_model_number_sft.json](cross_model_number_sft.json) | Fifteen final numeric-SFT matrices from five models and three domains; already represented in the atlas |
| [metric_correlations.csv](metric_correlations.csv) | Six metrics × 12 settings, full-matrix double-centering before all900/off870 masks |
| [metric_definitions.json](metric_definitions.json) | Current metric definitions and matrix orientation |
| [metric1_verification.json](metric1_verification.json) | Source-bound Metric 1 formula, prompt counts and weighting; publication matrix/correlation readback versus historical cache-audit scope |
| [red_hue_dynamics.json](red_hue_dynamics.json) | Four teachers' dense checkpoint curves; mean log-probability gain over same-step trained control |
| [token_sft_effective_rank.json](token_sft_effective_rank.json) | Full uncentered 30-color transfer-matrix energy effective rank; derived from original saved checkpoints |
| [multitrait_pair_ranks.json](multitrait_pair_ranks.json) | Eight pilot pair comparisons; hidden-state alignment is not Metric 1; ranking universes differ by domain |
| [color_pairs.csv](color_pairs.csv) | All 100 pair endpoints, ordered targets, selection cosine, target gains and ranks |
| [lion_prefix.csv](lion_prefix.csv) | Eight independent prefix conditions; lion gain against untrained base |
| [plant_rarity.csv](plant_rarity.csv) | Five target students; external-corpus rarity counts and next-token recovery relative to trained control |
| [qwen3_red_checkpoints.csv](qwen3_red_checkpoints.csv) | Base, prompted teacher and seven checkpoints; 200 behavioral replies and 32 numeric diagnostic replies per state |
| [qwen3_red_answer_histograms.json](qwen3_red_answer_histograms.json) | Saved exact answer strings and parser summaries for the Qwen3 development readout |
| [supplemental_studies.json](supplemental_studies.json) | Scope/status inventory for related analyses; not additional training counts |

## Metric matrices

Each NPZ contains the ordered `labels`, `sl`, and `M1` through `M6` arrays. These arrays use **teacher rows and answer columns**, unlike the atlas's answer-row display. All are 30 × 30. Metric 1 is symmetric; SL and several other metrics need not be. Use `numpy.load(path, allow_pickle=False)`.

| Regime | Color | City | Animal | Tree |
|---|---|---|---|---|
| Number token SFT | [matrix](metric_matrices/token_sft_00_99_color.npz) | [matrix](metric_matrices/token_sft_00_99_city.npz) | [matrix](metric_matrices/token_sft_00_99_animal.npz) | [matrix](metric_matrices/token_sft_00_99_tree.npz) |
| Number distribution distillation | [matrix](metric_matrices/distill_00_99_color.npz) | [matrix](metric_matrices/distill_00_99_city.npz) | [matrix](metric_matrices/distill_00_99_animal.npz) | [matrix](metric_matrices/distill_00_99_tree.npz) |
| Opaque-token distribution distillation | [matrix](metric_matrices/distill_opaque100_color.npz) | [matrix](metric_matrices/distill_opaque100_city.npz) | [matrix](metric_matrices/distill_opaque100_animal.npz) | [matrix](metric_matrices/distill_opaque100_tree.npz) |

The current Metric 1 has per-prompt normalization before averaging. Source hashes matched the saved execution audit; all published M1 and SL arrays matched their audited selected arrays, and all displayed correlations were recomputed from the included matrices. This publication did not rerun the complete historical per-prompt cache validation or infer new model scores. An aggregate matrix alone is insufficient to rederive the per-prompt definition from a legacy result.

## Reading recovery values

A log-probability difference in nats, an arithmetic probability difference, a sampled answer rate, and a normalized share among a fixed answer catalog are different quantities. In particular:

- The atlas and lion-prefix study use the untrained base as the reference.
- Red-hue, multi-trait/color-pair and plant-rarity recovery use their corresponding trained controls.
- The Qwen3 red table counts sampled behavioral answers; its numeric-validity denominator is a separate set of 32 diagnostic prompts.
- A positive gain need not make the target the most probable answer. A drop relative to a trained control need not mean its raw probability dropped.

See [the chapter](../README.md) for study-specific controls, selection procedures, overlap and limitations.
