# Real-data benchmark change report

## Baseline identity

- Path: `experiments/benchmark_results.json`
- SHA-256: `f27fb8aec0c522e179517a12fe9787e7e8131189af4190aa44a96e618bf0f071`
- The baseline has summary medians but no raw run records.

## Comparison rule

- Scope: adult, diabetes, pooled medians.
- Relative tolerance: 10%.
- Default absolute floor: 0.0500.
- Flag when `absolute_delta > max(metric_absolute_floor, relative_tolerance * abs(historical_value))`.
- Equality is within tolerance; missing or non-finite values are always flagged.

## Metric comparison

| Metric | Scope | Historical | Refreshed | Signed delta | Tolerance | Status |
|---|---|---:|---:|---:|---:|---|
| removal_corr | adult | 0.4271 | 0.4497 | 0.0226 | 0.0500 | within tolerance |
| removal_corr | diabetes | 0.4789 | 0.4763 | -0.0026 | 0.0500 | within tolerance |
| removal_corr | pooled | 0.4543 | 0.4636 | 0.0093 | 0.0500 | within tolerance |
| comprehensiveness | adult | 63.7952 | 45.5251 | -18.2701 | 6.3795 | MATERIAL CHANGE |
| comprehensiveness | diabetes | 10.6341 | 13.6115 | 2.9774 | 1.0634 | MATERIAL CHANGE |
| comprehensiveness | pooled | 28.5470 | 20.6734 | -7.8736 | 2.8547 | MATERIAL CHANGE |
| infidelity | adult | 0.7483 | 0.7353 | -0.0130 | 0.0748 | within tolerance |
| infidelity | diabetes | 0.7246 | 0.6560 | -0.0686 | 0.0725 | within tolerance |
| infidelity | pooled | 0.7256 | 0.6830 | -0.0426 | 0.0726 | within tolerance |
| sensitivity | adult | 0.0000 | 0.0001 | 0.0001 | 0.0100 | within tolerance |
| sensitivity | diabetes | 0.0000 | 0.0000 | 0.0000 | 0.0100 | within tolerance |
| sensitivity | pooled | 0.0000 | 0.0000 | 0.0000 | 0.0100 | within tolerance |
| stability_rank | adult | 0.9259 | 0.9215 | -0.0044 | 0.0926 | within tolerance |
| stability_rank | diabetes | 0.9096 | 0.9119 | 0.0023 | 0.0910 | within tolerance |
| stability_rank | pooled | 0.9207 | 0.9186 | -0.0021 | 0.0921 | within tolerance |
| stability_rank_topk | adult | 1.0000 | 0.8562 | -0.1438 | 0.1000 | MATERIAL CHANGE |
| stability_rank_topk | diabetes | 1.0000 | 0.8750 | -0.1250 | 0.1000 | MATERIAL CHANGE |
| stability_rank_topk | pooled | 1.0000 | 0.8750 | -0.1250 | 0.1000 | MATERIAL CHANGE |
| stability_sign | adult | 1.0000 | 1.0000 | 0.0000 | 0.1000 | within tolerance |
| stability_sign | diabetes | 1.0000 | 1.0000 | 0.0000 | 0.1000 | within tolerance |
| stability_sign | pooled | 1.0000 | 1.0000 | 0.0000 | 0.1000 | within tolerance |
| disagreement_sign | adult | 0.3194 | 0.2930 | -0.0264 | 0.0500 | within tolerance |
| disagreement_sign | diabetes | 0.3249 | 0.2748 | -0.0501 | 0.0500 | MATERIAL CHANGE |
| disagreement_sign | pooled | 0.3249 | 0.2809 | -0.0440 | 0.0500 | within tolerance |
| disagreement_rank | adult | 0.8042 | 0.8796 | 0.0754 | 0.0804 | within tolerance |
| disagreement_rank | diabetes | 0.8665 | 0.8857 | 0.0192 | 0.0867 | within tolerance |
| disagreement_rank | pooled | 0.8427 | 0.8796 | 0.0369 | 0.0843 | within tolerance |
| disagreement_rank_topk | adult | 0.6875 | 0.7500 | 0.0625 | 0.0688 | within tolerance |
| disagreement_rank_topk | diabetes | 0.6875 | 0.7500 | 0.0625 | 0.0688 | within tolerance |
| disagreement_rank_topk | pooled | 0.6875 | 0.7500 | 0.0625 | 0.0688 | within tolerance |
| disagreement_topk | adult | 0.7500 | 0.8333 | 0.0833 | 0.0750 | MATERIAL CHANGE |
| disagreement_topk | diabetes | 0.8333 | 0.8333 | 0.0000 | 0.0833 | within tolerance |
| disagreement_topk | pooled | 0.7917 | 0.8333 | 0.0416 | 0.0792 | within tolerance |
| disagreement_magnitude | adult | 0.4940 | 0.3891 | -0.1049 | 0.0500 | MATERIAL CHANGE |
| disagreement_magnitude | diabetes | 0.5160 | 0.4384 | -0.0776 | 0.0516 | MATERIAL CHANGE |
| disagreement_magnitude | pooled | 0.4940 | 0.4123 | -0.0817 | 0.0500 | MATERIAL CHANGE |
| distribution_rank | adult | 0.9835 | 0.9540 | -0.0295 | 0.0984 | within tolerance |
| distribution_rank | diabetes | 0.9914 | 0.9874 | -0.0040 | 0.0991 | within tolerance |
| distribution_rank | pooled | 0.9890 | 0.9794 | -0.0096 | 0.0989 | within tolerance |
| distribution_flip | adult | 1.0000 | 1.0000 | 0.0000 | 0.1000 | within tolerance |
| distribution_flip | diabetes | 0.0000 | 0.0000 | 0.0000 | 0.0500 | within tolerance |
| distribution_flip | pooled | 1.0000 | 0.5000 | -0.5000 | 0.1000 | MATERIAL CHANGE |

## Non-comparable fields

The historical file has no raw runs, split hashes, sample records, background records, or environment metadata. Paired run-level changes and provenance fields are therefore not comparable.

## Interpretation

A material-change flag requires a technical explanation; it is not automatically a regression failure because the historical and refreshed protocols intentionally differ.
