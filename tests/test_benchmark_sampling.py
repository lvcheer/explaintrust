"""Reproducible holdout sampling and user-configurable explanation budgets."""
import json
import numpy as np
import pandas as pd
import pytest

from experiments import benchmark_real_data as benchmark
from experiments.real_datasets import RawDataset


def test_sampling_is_reproducible_unique_and_not_a_fixed_prefix():
    draw = benchmark._sample_test_positions(100, 20, 7)
    np.testing.assert_array_equal(draw, benchmark._sample_test_positions(100, 20, 7))
    assert len(set(draw)) == 20 and np.all((draw >= 0) & (draw < 100))
    assert not np.array_equal(draw, np.arange(20))
    assert not np.array_equal(draw, benchmark._sample_test_positions(100, 20, 8))


def test_oversized_request_uses_every_available_row_once():
    draw = benchmark._sample_test_positions(3, 100, 4)
    assert sorted(draw) == [0, 1, 2]


@pytest.mark.parametrize('value', [0, -1, True, 2.5, '4'])
def test_invalid_budget_rejected_before_loading(monkeypatch, value):
    def unexpected_load():
        pytest.fail('invalid configuration triggered dataset loading')
    monkeypatch.setattr(benchmark, 'load_adult', unexpected_load)
    with pytest.raises(ValueError, match='positive integer'):
        benchmark.main(n_explain=value)


def test_empty_holdout_is_rejected():
    with pytest.raises(ValueError, match='empty'):
        benchmark._sample_test_positions(0, 2, 0)


@pytest.mark.parametrize('args', [['--n-explain', '0'], ['--n-explain', '-3'], ['--n-explain', '2.5']])
def test_cli_rejects_invalid_counts(args):
    with pytest.raises(SystemExit) as error:
        benchmark._parse_args(args)
    assert error.value.code == 2


def test_cli_accepts_custom_count_and_default():
    assert benchmark._parse_args(['--n-explain', '25']).n_explain == 25
    assert benchmark._parse_args([]).n_explain == benchmark.N_EXPLAIN


def test_main_uses_selected_rows_for_all_models_independent_of_background_budget(monkeypatch, tmp_path):
    frame = pd.DataFrame(np.arange(80).reshape(20, 4), columns=list('abcd'))
    data = RawDataset(frame, np.arange(20) % 2, list('abcd'), [],
                      test_mask=np.arange(20) >= 10)
    monkeypatch.setattr(benchmark, 'load_adult', lambda: data)
    monkeypatch.setattr(benchmark, 'load_diabetes', lambda: data)
    monkeypatch.setattr(benchmark, 'SEEDS', range(2))
    monkeypatch.setattr(benchmark, 'HERE', tmp_path)
    class Model:
        def fit(self, X, y):
            return self
    monkeypatch.setattr(benchmark, 'MODELS', {'A': lambda seed: Model(), 'B': lambda seed: Model()})
    calls = []
    def metrics(model, explained, background, names, holdout, seed, *, test_positions):
        expected = benchmark._sample_test_positions(len(holdout), 3, seed)
        np.testing.assert_array_equal(test_positions, expected)
        np.testing.assert_array_equal(explained, holdout[expected])
        calls.append((seed, test_positions.tolist()))
        return {**{key: 0.5 for key in benchmark.METRICS}, 'shap_context': {}}
    monkeypatch.setattr(benchmark, '_run_metrics', metrics)
    for bg in [2, 5]:
        monkeypatch.setattr(benchmark, 'BG', bg)
        benchmark.main(n_explain=3)
        payload = json.loads((tmp_path / 'benchmark_results.json').read_text())
        assert payload['budget']['requested_explanations'] == 3
        for run in payload['runs']:
            assert run['sampling']['actual'] == run['sampling']['requested'] == 3
            assert run['sampling']['available'] == 10
            assert run['sampling']['unit'] == 'test_row'
    assert len(calls) == 16 and calls[:8] == calls[8:]
    for i in range(0, 16, 2):
        assert calls[i] == calls[i + 1]
