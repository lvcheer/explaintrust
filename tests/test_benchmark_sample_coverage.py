"""All explained instances contribute to sensitivity and repeat stability."""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from experiments import benchmark_real_data as benchmark


@pytest.mark.parametrize('sensitivity_values', [[0.1, 0.4, 1.3], [0.1, np.inf, np.nan]])
def test_all_samples_are_evaluated_with_full_batch_repeats(monkeypatch, sensitivity_values):
    X = np.random.default_rng(12).normal(size=(40, 4))
    model = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=1).fit(X, X[:, 0] > 0)
    explained = X[:3]
    sensitivity_calls, lime_calls = [], []
    original_lime = benchmark.lime_attributions

    def sensitivity(explain, x, background, *, n_perturbations, seed):
        sensitivity_calls.append((x.copy(), n_perturbations, seed))
        return sensitivity_values[len(sensitivity_calls) - 1]

    def lime(model, instances, background, **kwargs):
        lime_calls.append((instances.copy(), kwargs['num_samples'], kwargs['seed']))
        return original_lime(model, instances, background, **kwargs)

    monkeypatch.setattr(benchmark, 'max_sensitivity', sensitivity)
    monkeypatch.setattr(benchmark, 'lime_attributions', lime)
    monkeypatch.setattr(benchmark, 'LIME_SAMPLES', 100)
    result = benchmark._run_metrics(model, explained, X, list('abcd'), X, seed=12)
    assert len(sensitivity_calls) == 3
    for i, (instance, budget, seed) in enumerate(sensitivity_calls):
        np.testing.assert_array_equal(instance, explained[i])
        assert (budget, seed) == (6, 12 + i)
    assert len(lime_calls) == 5
    for i, (instances, budget, seed) in enumerate(lime_calls):
        np.testing.assert_array_equal(instances, explained)
        assert (budget, seed) == (100, 12 + i)
    assert result['sensitivity'] == benchmark._mean(sensitivity_values)
    assert result['metric_counts']['sensitivity'] == benchmark._counts(sensitivity_values)
    for i, sample in enumerate(result['samples']):
        assert sample['metrics']['sensitivity'] == benchmark._measurement(sensitivity_values[i])
        repeats = result['stability_runs'][i]['runs']
        assert repeats[0]['contributions'] == sample['lime_contributions']
        stability = benchmark.cross_run_stability(
            lambda seed: repeats[seed]['contributions'], n_runs=5, top_k=3)
        for output, key in [('stability_rank', 'rank_corr'), ('stability_rank_topk', 'topk_rank_corr'),
                            ('stability_sign', 'sign_agreement')]:
            assert sample['metrics'][output] == benchmark._measurement(stability[key])
