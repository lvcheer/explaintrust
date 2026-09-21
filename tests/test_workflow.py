from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.workflow import commit, load_data, split_data, feature_dictionary, explain_model, PRESETS
from app.tabular_utils import make_model
from app.streamlit_app import _run_battery
from explaintrust import shap_attributions, scalar_predictor


def test_upstream_commit_removes_all_downstream_artifacts():
    state = dict(data=1, split=2, trained=3, explained=4, analysis=5)
    commit(state, 'split', 6)
    assert state == {'data': 1, 'split': 6}


def test_synthetic_meanings_do_not_invent_business_variables():
    data = load_data(PRESETS[0])
    assert data['meanings']['x2'] != load_data(PRESETS[1])['meanings']['x2']
    assert len(feature_dictionary(data, data['names'])) == 10
    split = split_data(data, 5, 0.2)
    assert len(split['Xte']) == 300
    np.testing.assert_array_equal(split['Xte'], split_data(data, 5, 0.2)['Xte'])


@pytest.mark.parametrize('methods', [('SHAP',), ('LIME',), ('SHAP', 'LIME')])
def test_selected_methods_only_and_quality_reuses_explanations(monkeypatch, methods):
    import app.workflow as workflow
    import app.streamlit_app as app
    X = np.random.default_rng(5).normal(size=(60, 4))
    frame = pd.DataFrame(X, columns=list('abcd')).assign(target=(X[:, 0] > 0).astype(int))
    data = load_data('上传 CSV', frame=frame, target='target')
    split = split_data(data, 0, 0.3)
    model = make_model('classification', 'Random Forest', 5, 2, 0).fit(split['Xtr'], split['ytr'])
    def forbidden(*args, **kwargs):
        pytest.fail('unselected method was executed')
    if 'SHAP' not in methods:
        monkeypatch.setattr(workflow, 'shap_attributions', forbidden)
        monkeypatch.setattr(app, 'shap_attributions', forbidden)
    if 'LIME' not in methods:
        monkeypatch.setattr(workflow, 'lime_attributions', forbidden)
        monkeypatch.setattr(app, 'lime_attributions', forbidden)
    explained = explain_model({'model': model}, split, list(methods), 2, 100)
    out = _run_battery(model, explained['X'], explained['bg'], split['names'], 2, 100, 3, 0,
                       split['Xte'], 0, methods=methods, precomputed=explained)
    assert out['report'].context['selected_methods'] == list(methods)
    if 'SHAP' in methods:
        np.testing.assert_array_equal(out['shap_attr'], explained['shap_attr'])
    if 'LIME' in methods:
        np.testing.assert_array_equal(out['lime_contrib'], explained['lime_contrib'])
    if len(methods) == 1:
        assert out['report'].overall.startswith('INSUFFICIENT')
        assert all(m.verdict == 'info' for m in out['report'].metric_results if m.name.startswith('SHAP vs LIME'))


def test_upload_regression_and_target_removal():
    frame = pd.DataFrame({'age': np.arange(60), 'x': np.arange(60)**2, 'outcome': np.arange(60)*0.2})
    data = load_data('上传 CSV', frame=frame, target='outcome', task='regression')
    assert data['names'] == ['age', 'x']
    split = split_data(data, 0, 0.3)
    model = make_model('regression', 'Linear Regression').fit(split['Xtr'], split['ytr'])
    explained = explain_model({'model': model}, split, ['SHAP'], 2, 100)
    assert explained['shap_attr'].shape == (2, 2)
    assert explained['output_space'] == 'raw'


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_xgboost_training_and_shap_output_alignment(task):
    pytest.importorskip('xgboost')
    X = np.random.default_rng(1).normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int) if task == 'classification' else X[:, 0] * 3
    model = make_model(task, 'XGBoost', 10, 2, 0).fit(X, y)
    values, context = shap_attributions(model, X[:2], X_background=X[:20], method='auto', return_context=True)
    np.testing.assert_allclose(values.sum(axis=1) + context['expected_value'], scalar_predictor(model)(X[:2]), atol=1e-5)
    if task == 'classification':
        np.testing.assert_allclose(scalar_predictor(model, class_index=0)(X[:2]), -scalar_predictor(model)(X[:2]))


def test_button_workflow_and_retraining_invalidation():
    from streamlit.testing.v1 import AppTest
    app_path = Path(__file__).resolve().parents[1] / 'app' / 'streamlit_app.py'
    app = AppTest.from_file(app_path).run(timeout=40)

    def click(label):
        next(button for button in app.button if button.label == label).click()
        app.run(timeout=60)
        assert not app.exception, [error.message for error in app.exception]
        assert not app.error, [error.value for error in app.error]

    assert 'data' not in app.session_state['workflow']
    for label in ['加载数据', '确认数据与字段含义 → 数据划分', '执行数据划分', '下一步 → 训练模型']:
        click(label)
    next(slider for slider in app.slider if slider.label == '树的数量（树模型）').set_value(10)
    click('训练模型')
    click('下一步 → 解释预测')
    next(slider for slider in app.slider if slider.label == 'LIME 每样本采样数').set_value(500)
    click('生成解释')
    assert any(metric.label == '真实目标值' for metric in app.metric)
    assert any(metric.label == '模型预测' for metric in app.metric)
    click('下一步 → 分析解释质量')
    next(slider for slider in app.slider if slider.label == 'LIME 稳定性重复次数').set_value(3)
    click('运行质量分析')
    assert 'analysis' in app.session_state['workflow']
    click('3. 训练模型')
    click('训练模型')
    assert 'trained' in app.session_state['workflow']
    assert 'explained' not in app.session_state['workflow']
    assert 'analysis' not in app.session_state['workflow']
