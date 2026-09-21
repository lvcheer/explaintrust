from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from app.tabular_utils import make_model, evaluate_model
from app.workflow import explain_model


class ScoredClassifier:
    classes_ = np.array([0, 1])
    def predict_proba(self, X):
        return np.column_stack([1-X[:,0], X[:,0]])
    def predict(self, X):
        return (X[:,0] > 0.5).astype(int)


def test_auc_uses_scores_and_confusion_has_fixed_class_order():
    X = np.array([[.1],[.8],[.4],[.7]])
    y = np.array([0,0,1,1])
    result = evaluate_model(ScoredClassifier(), X, y, 'classification')
    assert result['ROC AUC'] == roc_auc_score(y,X[:,0]) == .5
    assert result['confusion_matrix'] == [[1,1],[1,1]]
    assert result['Precision'] == result['Recall'] == result['F1'] == .5
    missing = evaluate_model(ScoredClassifier(), X, np.zeros(4), 'classification')
    assert missing['ROC AUC'] is None and missing['fpr'] == []
    assert np.array(missing['confusion_matrix']).shape == (2,2)


def test_regression_metrics_have_no_classification_fields():
    X=np.arange(10).reshape(-1,1)
    model=make_model('regression','Linear Regression').fit(X,2*X[:,0]+1)
    result=evaluate_model(model,X,2*X[:,0]+1,'regression')
    assert set(result) == {'R²','MAE','RMSE'}
    assert result['R²'] == 1. and result['RMSE'] < 1e-10


def test_mlp_scaling_is_train_only_and_shap_is_on_pipeline_output():
    X=np.random.default_rng(3).normal(size=(60,3))
    model=make_model('classification','MLP',seed=0,hidden_layers=(8,4),learning_rate=.01,max_iter=100)
    model.fit(X[:40],(X[:40,0]>0).astype(int))
    np.testing.assert_allclose(model.named_steps['standardscaler'].mean_, X[:40].mean(axis=0))
    split=dict(Xtr=X[:40],Xte=X[40:],names=list('abc'),seed=0)
    out=explain_model({'model':model},split,['SHAP','LIME'],1,100,background_size=10,shap_samples=50)
    from explaintrust import scalar_predictor
    np.testing.assert_allclose(out['shap_attr'].sum(axis=1)+out['shap_context']['expected_value'],
                               scalar_predictor(model)(out['X']),atol=1e-5)
    assert out['lime_attr'].shape == (1,3)
    assert out['shap_context']['method'] == 'kernel'


def test_model_specific_parameters_reach_estimators():
    assert make_model('classification','Logistic Regression',regularization=.2,max_iter=123).C == .2
    assert make_model('regression','Linear Regression',fit_intercept=False).fit_intercept is False
    assert make_model('classification','Gradient Boosting',learning_rate=.02).learning_rate == .02
    mlp=make_model('regression','MLP',hidden_layers=(16,8),alpha=.02,max_iter=123)
    assert mlp.named_steps['mlpregressor'].hidden_layer_sizes == (16,8)
    assert mlp.named_steps['mlpregressor'].alpha == .02


def test_quality_settings_reach_metric_computations(monkeypatch):
    import app.streamlit_app as app
    X=np.random.default_rng(8).normal(size=(40,4))
    model=make_model('classification','Random Forest',5,2,0).fit(X,(X[:,0]>0).astype(int))
    split=dict(Xtr=X,Xte=X[:10],names=list('abcd'),seed=0)
    precomputed=explain_model({'model':model},split,['SHAP','LIME'],2,100)
    calls={'sensitivity':[],'infidelity':[],'random':[]}
    def sensitivity(*args, **kwargs):
        calls['sensitivity'].append(kwargs); return .5
    def infidelity(*args, **kwargs):
        calls['infidelity'].append(kwargs); return .2
    def random_removal(*args, **kwargs):
        calls['random'].append(kwargs); return 2.
    monkeypatch.setattr(app,'max_sensitivity',sensitivity)
    monkeypatch.setattr(app,'infidelity',infidelity)
    monkeypatch.setattr(app,'comprehensiveness_ratio',random_removal)
    config=dict(top_k=1,sensitivity_samples=2,sensitivity_perturbations=7,sensitivity_radius=.2,
                infidelity_perturbations=75,comprehensiveness_random=9)
    out=app._run_battery(model,precomputed['X'],precomputed['bg'],list('abcd'),2,100,3,0,X,0,
                         precomputed=precomputed,quality_config=config)
    assert len(calls['sensitivity'])==2
    assert [c['seed'] for c in calls['sensitivity']]==[0,1]
    assert all(c['n_perturbations']==7 and c['radius']==.2 for c in calls['sensitivity'])
    assert all(c['n_perturbations']==75 for c in calls['infidelity'])
    assert all(c['n_random']==9 and c['top_k']==1 for c in calls['random'])
    assert out['report'].context['quality_config']==config
    assert out['report'].context['sensitivity_instances']==2


def test_language_switch_preserves_model_parameters_and_training(monkeypatch):
    from streamlit.elements.lib import policies
    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    from streamlit.testing.v1 import AppTest
    app_path = Path(__file__).resolve().parents[1] / 'app' / 'streamlit_app.py'
    app = AppTest.from_file(app_path).run(timeout=40)

    def click(label):
        next(b for b in app.button if b.label == label).click()
        app.run(timeout=60)
        assert not app.exception
        assert not app.error

    for label in ['加载数据', '确认数据与字段含义 → 数据划分', '执行数据划分', '下一步 → 训练模型']:
        click(label)
    app.selectbox[0].set_value('Logistic Regression').run()
    assert not any('树的数量' in s.label for s in app.slider)
    app.number_input[0].set_value(2.0).run()
    app.radio[0].set_value('en').run()
    assert app.selectbox[0].value == 'Logistic Regression'
    assert app.number_input[0].label == 'Regularization parameter C'
    assert app.number_input[0].value == 2.0
    click('Train model')
    trained = app.session_state['workflow']['trained']
    assert trained['parameters']['regularization'] == 2.0
    assert any(m.label == 'ROC AUC' for m in app.metric)
    app.radio[0].set_value('zh').run()
    assert app.session_state['workflow']['trained']['model'] is trained['model']
    assert app.number_input[0].value == 2.0
    app.selectbox[0].set_value('MLP').run()
    assert any(s.label == '隐藏层数' for s in app.slider)
    assert not any('树的数量' in s.label for s in app.slider)
    app.radio[0].set_value('en').run()
    assert app.selectbox[0].value == 'MLP'
    assert any(s.label == 'Hidden layers' for s in app.slider)

    next(s for s in app.slider if s.label == 'Hidden layers').set_value(3).run()
    app.radio[0].set_value('zh').run()
    assert next(s for s in app.slider if s.label == '隐藏层数').value == 3
    assert not policies._shown_default_value_warning
    assert not any('Session State API' in w.value for w in app.warning)
