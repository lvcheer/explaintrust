"""Non-finite benchmark observations remain visible in standards-compliant JSON."""
import json
import numpy as np

from experiments.benchmark_real_data import _counts, _json_safe, _mean, _measurement, _pct


def test_infinite_failure_is_not_dropped_from_run_mean():
    values = [0.1, np.inf, np.nan]
    assert np.isposinf(_mean(values))
    assert _counts(values) == {'total': 3, 'finite': 1, 'nan': 1,
                               'positive_infinity': 1, 'negative_infinity': 0,
                               'not_computed': 0}
    assert _measurement(_mean(values)) == {'value': None, 'status': 'positive_infinity'}


def test_missing_only_run_is_unavailable_and_finite_percentile_counts_are_explicit():
    assert _measurement(_mean([np.nan])) == {'value': None, 'status': 'nan'}
    assert np.isnan(_pct([np.nan, np.inf], 50))
    assert _pct([1., 3., np.nan, np.inf], 50) == 2.
    assert _counts([1., 3., np.nan, np.inf])['finite'] == 2


def test_strict_json_distinguishes_nonfinite_states():
    values = [np.nan, np.inf, -np.inf, 1.]
    payload = {'metrics': [_measurement(v) for v in values], 'summary': np.nan}
    encoded = json.dumps(_json_safe(payload), allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded['summary'] is None
    assert [m['status'] for m in decoded['metrics']] == [
        'nan', 'positive_infinity', 'negative_infinity', 'finite']
    assert [m['value'] for m in decoded['metrics']] == [None, None, None, 1.]
