"""Exercise KernelSHAP's sampled (rather than fully enumerated) path."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import time

import numpy as np
import pytest

from explaintrust import shap_attributions


class NonlinearModel:
    def predict(self, X):
        return (
            np.sin(X[:, 0] * X[:, 1]) + X[:, 2] ** 2
            + X[:, 3] * X[:, 4] + np.cos(X[:, 5] + X[:, 6])
        )


@pytest.fixture(autouse=True)
def preserve_global_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def explain(seed=7, method="kernel", model=None):
    X = np.random.default_rng(42).normal(size=(100, 10))
    # 30 samples cannot enumerate the 2**10 - 2 nontrivial coalitions.
    return shap_attributions(
        model if model is not None else NonlinearModel(), X[:2], X[:15],
        method=method, nsamples=30, seed=seed,
    )


@pytest.mark.parametrize("method", ["kernel", "auto"])
def test_same_seed_is_independent_of_external_rng(method):
    np.random.seed(100)
    first = explain(method=method)
    np.random.seed(200)
    second = explain(method=method)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-12)


def test_different_seeds_change_sampled_attributions():
    # Reset the external RNG to show that the *API seed* controls the result.
    np.random.seed(100)
    first = explain(seed=7)
    np.random.seed(100)
    second = explain(seed=8)
    assert not np.allclose(first, second, rtol=0, atol=1e-8)


def assert_rng_state_equal(before):
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_kernel_preserves_callers_rng_state():
    np.random.seed(100)
    np.random.normal()  # Include the cached Gaussian in the saved state.
    before = np.random.get_state()
    explain()
    assert_rng_state_equal(before)


@pytest.mark.parametrize("stage", ["initialization", "sampling"])
def test_kernel_restores_rng_when_predictor_raises(stage):
    class FailingModel(NonlinearModel):
        def predict(self, X):
            if stage == "initialization" or len(X) > 15:
                np.random.random()
                raise RuntimeError("prediction failed")
            return super().predict(X)

    np.random.seed(100)
    before = np.random.get_state()
    with pytest.raises(RuntimeError, match="prediction failed"):
        explain(model=FailingModel())
    assert_rng_state_equal(before)


def test_concurrent_kernel_calls_match_serial_results():
    class SlowModel(NonlinearModel):
        def predict(self, X):
            time.sleep(0.01)  # Let the other caller enter while SHAP is active.
            return super().predict(X)

    seeds = [7, 8]
    expected = [explain(seed=seed) for seed in seeds]
    start = Barrier(2)

    def run(seed):
        start.wait(timeout=10)
        return explain(seed=seed, model=SlowModel())

    before = np.random.get_state()
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(run, seeds))
    for serial, concurrent in zip(expected, actual):
        np.testing.assert_allclose(serial, concurrent, rtol=0, atol=1e-12)
    assert_rng_state_equal(before)
