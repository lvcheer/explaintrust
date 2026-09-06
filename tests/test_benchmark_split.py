"""Holdout values must never fit benchmark preprocessing or cross patient groups."""
import numpy as np
import pandas as pd
import pytest

from experiments.real_datasets import RawDataset, _encode, prepare_split


def test_encoding_uses_training_medians_scaling_and_category_vocabulary():
    train = pd.DataFrame({'x': [0., 2., np.nan], 'empty': [np.nan]*3,
                          'cat': ['a', 'b', 'a']})
    test = pd.DataFrame({'x': [np.nan, 1000.], 'empty': [7., np.nan],
                         'cat': ['test-only', 'b']})
    tr, te, names = _encode(train, test, ['x', 'empty'], ['cat'])
    assert names == ['x', 'empty', 'cat_b']
    scale = np.std([0., 2., 1.]) + 1e-12
    np.testing.assert_allclose(tr[:, 0], [-1/scale, 1/scale, 0.])
    np.testing.assert_allclose(te[:, 0], [0., 999/scale])
    np.testing.assert_array_equal(tr[:, 1], 0.)
    np.testing.assert_array_equal(te[:, 1], [7., 0.])
    np.testing.assert_array_equal(te[:, 2], [0., 1.])
    changed = test.assign(x=-999999., empty=88., cat='another-unseen-category')
    changed_tr, _, changed_names = _encode(train, changed, ['x', 'empty'], ['cat'])
    np.testing.assert_array_equal(tr, changed_tr)
    assert names == changed_names


def test_official_partition_is_preserved_and_independent_of_seed():
    data = RawDataset(pd.DataFrame({'x': [0., 1., 2., 100., 200.]}),
                      np.array([0, 1, 0, 1, 1]), ['x'], [],
                      test_mask=np.array([False]*3 + [True]*2))
    a, b = prepare_split(data, 0), prepare_split(data, 9)
    for left, right in zip(a[:4], b[:4]):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(a[2], [0, 1, 0])
    np.testing.assert_array_equal(a[3], [1, 1])
    assert a[5]['protocol'] == 'adult-official-holdout'
    assert a[5]['train_indices_sha256'] == b[5]['train_indices_sha256']
    assert abs(a[0].mean()) < 1e-10


def test_patient_groups_do_not_cross_partitions_and_split_is_reproducible():
    groups = np.repeat(np.arange(20), 3)
    # Identity labels let this test independently recover the patient sets.
    data = RawDataset(pd.DataFrame({'x': np.arange(len(groups))}),
                      groups.copy(), ['x'], [], groups=groups)
    a, repeat, other = [prepare_split(data, seed) for seed in [7, 7, 8]]
    assert set(a[2]).isdisjoint(a[3])
    assert len(a[2]) + len(a[3]) == len(groups)
    assert a[5]['patient_overlap'] == 0
    assert a[5]['test_patients'] == 6
    np.testing.assert_array_equal(a[0], repeat[0])
    assert a[5] == repeat[5]
    assert a[5]['test_indices_sha256'] != other[5]['test_indices_sha256']
    assert a[5]['preprocessing_fit'] == 'train_only'


def test_missing_patient_ids_are_rejected():
    data = RawDataset(pd.DataFrame({'x': [1, 2, 3]}), np.array([0, 1, 0]),
                      ['x'], [], groups=np.array([1., np.nan, 2.]))
    with pytest.raises(ValueError, match='missing identifiers'):
        prepare_split(data, 0)


def test_loaders_preserve_partition_metadata_and_exclude_identifiers(monkeypatch):
    from experiments import real_datasets as datasets
    monkeypatch.setattr(datasets, '_download', lambda *args: None)
    monkeypatch.setattr(datasets.Path, 'exists', lambda self: True)
    adult = pd.DataFrame({'income': ['<=50K', '>50K'], 'fnlwgt': [1, 2],
                          'education': ['a', 'b'], 'age': [20, 30]})
    official_test = adult.copy()
    official_test['income'] = ['>50K.', '<=50K.']
    monkeypatch.setattr(datasets.pd, 'read_csv',
                        lambda path, **kw: official_test.copy() if path.name == 'adult.test' else adult.copy())
    loaded = datasets.load_adult()
    np.testing.assert_array_equal(loaded.test_mask, [False, False, True, True])
    np.testing.assert_array_equal(loaded.y, [0, 1, 1, 0])
    frame = pd.DataFrame({c: [1, 2, 3] for c in datasets.DIABETES_DROP})
    frame['patient_nbr'] = [11, 11, 22]
    frame['readmitted'] = ['NO', '<30', '>30']
    for column in datasets.DIABETES_MEDS:
        frame[column] = ['No', 'Steady', 'Up']
    frame['change'], frame['diabetesMed'] = ['No', 'Ch', 'Ch'], ['No', 'Yes', 'Yes']
    monkeypatch.setattr(datasets.pd, 'read_csv', lambda *args, **kw: frame.copy())
    loaded = datasets.load_diabetes()
    np.testing.assert_array_equal(loaded.groups, [11, 11, 22])
    assert 'patient_nbr' not in loaded.frame and 'encounter_id' not in loaded.frame
    assert 'readmitted' not in loaded.frame
    np.testing.assert_array_equal(loaded.y, [0, 1, 1])
