# -*- coding: utf-8 -*-
"""Button-driven analysis workflow; upstream commits invalidate downstream artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from app.i18n import st
from sklearn.model_selection import train_test_split

from app.tabular_utils import prepare_tabular, make_model, CLASSIFICATION_MODELS, REGRESSION_MODELS, segment_feature, evaluate_model
from explaintrust import make_collinear_dataset, shap_attributions, lime_attributions, to_contribution_scale, prediction_output_space, scalar_predictor
from experiments.real_datasets import load_adult, load_diabetes, prepare_split

STEPS = ['选择数据', '划分数据', '训练模型', '解释预测', '分析解释质量']
ARTIFACTS = ['data', 'split', 'trained', 'explained', 'analysis']
PRESETS = ['模拟：相关代理特征', '模拟：独立噪声特征', 'Adult 收入分类', 'Diabetes 再入院分类', '上传 CSV']
MEANINGS = {
    'age': '年龄；Diabetes 中为年龄区间', 'education-num': '教育程度的数值编码',
    'capital-gain': '资本收益', 'capital-loss': '资本损失', 'hours-per-week': '每周工作小时数',
    'workclass': '工作类别', 'marital-status': '婚姻状态', 'occupation': '职业类别',
    'relationship': '家庭关系类别', 'race': '数据集记录的种族类别', 'sex': '数据集记录的性别',
    'gender': '数据集记录的性别', 'native-country': '原籍国家',
    'time_in_hospital': '本次住院天数', 'num_lab_procedures': '检验次数',
    'num_procedures': '非检验操作次数', 'num_medications': '用药数量',
    'number_outpatient': '此前一年门诊次数', 'number_emergency': '此前一年急诊次数',
    'number_inpatient': '此前一年住院次数', 'number_diagnoses': '记录的诊断数量',
    'admission_type_id': '入院类型编码', 'max_glu_serum': '血糖检查结果类别',
    'A1Cresult': '糖化血红蛋白检查结果类别', 'change': '糖尿病用药是否调整（1=调整）',
    'diabetesMed': '是否使用糖尿病药物（1=是）',
}


def commit(state, key, value):
    """Replace one completed stage and remove every dependent artifact."""
    index = ARTIFACTS.index(key)
    for dependent in ARTIFACTS[index + 1:]:
        state.pop(dependent, None)
    state[key] = value


def load_data(source, seed=0, frame=None, target=None, task='classification'):
    if source.startswith('模拟'):
        X, y, names = make_collinear_dataset(n=1500, seed=seed)
        independent = source == PRESETS[1]
        if independent:
            X[:, 2] = np.random.default_rng(seed).normal(size=len(X))
        meanings = {name: '独立噪声变量；不参与标签生成' for name in names}
        meanings.update({'x0': '人工变量：参与标签生成，含线性和平方项',
                         'x1': '人工变量：参与标签生成，并与 x0 交互',
                         'x2': '独立噪声变量；不参与标签生成' if independent else 'x0 的相关代理；不直接参与标签生成'})
        return {'X': X, 'y': y, 'names': names, 'task': 'classification', 'meanings': meanings,
                'preview': pd.DataFrame(X, columns=names).assign(target=y), 'source': source,
                'target': '人工生成的二分类标签；1 为模型所解释的目标类别', 'preprocessing': {}}
    if source in PRESETS[2:4]:
        raw = load_adult() if source == PRESETS[2] else load_diabetes()
        names = raw.numeric + raw.categorical
        from experiments.real_datasets import DIABETES_MEDS
        meanings = {name: MEANINGS.get(name, '是否使用该药物（No=0，其他记录=1）' if name in DIABETES_MEDS else '数据集字段') for name in names}
        return {'raw': raw, 'y': raw.y, 'names': names, 'task': 'classification',
                'meanings': meanings, 'preview': raw.frame[names].head(100).assign(target=raw.y[:100]),
                'source': source, 'target': '1：年收入 >50K；0：≤50K' if source == PRESETS[2] else '1：再次入院（<30 或 >30）；0：未再次入院', 'preprocessing': {}}
    X, y, names, prep = prepare_tabular(frame, target, task)
    if len(X) < 40 or len(names) < 2:
        raise ValueError('至少需要 40 条完整记录和 2 个数值特征。')
    if not np.isfinite(X).all() or (task == 'regression' and not np.isfinite(y).all()):
        raise ValueError('特征和回归目标必须为有限数值。')
    if task == 'classification' and (len(np.unique(y)) != 2 or np.min(np.bincount(y)) < 2):
        raise ValueError('当前支持二分类；每类至少需要两条记录。')
    return {'X': X, 'y': y, 'names': names, 'task': task, 'meanings': {n: '待填写业务含义' for n in names},
            'preview': frame.head(100), 'source': source, 'target': target, 'preprocessing': prep}


def split_data(data, seed, test_size):
    if 'raw' in data:
        Xtr, Xte, ytr, yte, names, context = prepare_split(data['raw'], seed, test_size)
    else:
        Xtr, Xte, ytr, yte = train_test_split(data['X'], data['y'], test_size=test_size,
                                            random_state=seed, stratify=data['y'] if data['task'] == 'classification' else None)
        names = data['names']
        context = {'protocol': 'stratified_random_rows' if data['task'] == 'classification' else 'random_rows',
                   'seed': seed, 'train_rows': len(Xtr), 'test_rows': len(Xte)}
    return dict(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, names=names, context=context, seed=seed)


def feature_dictionary(data, names):
    rows = []
    for name in names:
        raw_name = name if name in data['meanings'] else next((n for n in sorted(data['meanings'], key=len, reverse=True) if name.startswith(n + '_')), None)
        meaning = data['meanings'].get(raw_name, '未提供说明')
        if raw_name and name != raw_name:
            meaning += f'；类别指示：{name[len(raw_name)+1:]}（1=该类别）'
        rows.append({'字段': name, '含义': meaning})
    return pd.DataFrame(rows)


def explain_model(trained, split, methods, n_explain, lime_samples, background_size=100, shap_samples=100):
    if not methods:
        raise ValueError('至少选择一种解释方法。')
    rng = np.random.default_rng(split['seed'])
    positions = rng.choice(len(split['Xte']), min(n_explain, len(split['Xte'])), replace=False)
    bg = split['Xtr'][rng.choice(len(split['Xtr']), min(background_size, len(split['Xtr'])), replace=False)]
    X = split['Xte'][positions]
    model = trained['model']
    space = prediction_output_space(model)
    out = dict(X=X, bg=bg, positions=positions, methods=methods, lime_samples=lime_samples, output_space=space, shap_samples=shap_samples)
    if 'SHAP' in methods:
        out['shap_attr'], out['shap_context'] = shap_attributions(model, X, X_background=bg, method='auto', nsamples=shap_samples, seed=split['seed'], return_context=True)
    if 'LIME' in methods:
        out['lime_attr'] = lime_attributions(model, X, bg, feature_names=split['names'], num_samples=lime_samples,
                                             seed=split['seed'], output_space=space)
        out['lime_contrib'] = to_contribution_scale(out['lime_attr'], X, bg)
    return out



def _go(stage, data=None):
    if data is not None:
        commit(st.session_state['workflow'], 'data', data)
    state = st.session_state["workflow"]
    if stage > 0 and ARTIFACTS[stage - 1] not in state:
        stage = next((i for i, key in enumerate(ARTIFACTS) if key not in state), 4)
    st.session_state.workflow_stage = stage

def render_workflow(battery):
    import streamlit as native_st
    def preserve_inputs():
        saved = {}
        for key in list(native_st.session_state):
            if key.startswith(('ui_selectbox_', 'ui_multiselect_', 'ui_radio_', 'ui_slider_', 'ui_number_input_', 'ui_checkbox_', 'ui_text_input_')):
                saved[key] = native_st.session_state[key]
                del native_st.session_state[key]
        native_st.session_state['_language_inputs'] = saved

    native_st.sidebar.radio('Language / 语言', ['zh', 'en'], format_func=lambda v: '中文' if v == 'zh' else 'English', key='language', horizontal=True, on_change=preserve_inputs)
    state = st.session_state.setdefault('workflow', {})
    stage = st.session_state.get('workflow_stage', 0)
    st.sidebar.title('🔍 explaintrust')
    st.sidebar.caption('数据 → 划分 → 训练 → 解释 → 质量分析')
    for i, title in enumerate(STEPS):
        allowed = i == 0 or ARTIFACTS[i - 1] in state
        st.sidebar.button(f'{i + 1}. {title}', disabled=not allowed, type='primary' if stage == i else 'secondary', width='stretch', on_click=_go, args=(i,))
    st.title(f'{stage + 1} · {STEPS[stage]}')
    st.progress((stage + 1) / 5)
    st.caption('点击按钮执行当前步骤。重新执行上游步骤会清除后续结果；修改输入后请先提交当前步骤。')

    if stage == 0:
        source = st.selectbox('数据集', PRESETS)
        frame, target, task = None, None, 'classification'
        generation_seed = 0
        if source.startswith('模拟'):
            st.info('x0、x1、x2 是人工构造的变量，不对应年龄、收入等真实业务字段。此处保留原始编号，展示其生成角色。')
            generation_seed = st.number_input('数据生成种子', 0, 10000, 0)
        elif source == '上传 CSV':
            uploaded = st.file_uploader('上传数据表', type=['csv'])
            if uploaded is not None:
                try:
                    frame = pd.read_csv(uploaded)
                    target = st.selectbox('目标列', list(frame.columns))
                    task = st.selectbox('任务类型', ['classification', 'regression'], format_func=lambda v: '二分类' if v == 'classification' else '回归')
                    st.dataframe(frame.head(10), hide_index=True, translate=False)
                except Exception as exc:
                    st.error(f'无法读取 CSV：{exc}')
            st.caption('当前上传流程仅保留数值特征，排除含缺失值的行。字段含义可在加载后填写。')
        else:
            st.caption('首次使用需下载 UCI 数据集。Adult 保留官方测试集；Diabetes 按患者划分。')
        if st.button('加载数据', type='primary', disabled=source == '上传 CSV' and frame is None):
            state.clear()
            st.session_state.pop("feature_meanings", None)
            with st.spinner('正在加载数据…'):
                try:
                    commit(state, 'data', load_data(source, generation_seed, frame, target, task))
                except Exception as exc:
                    st.error(f'加载失败：{exc}')
        if 'data' in state:
            data = state['data']
            st.success(f"已加载：{data['source']} · {len(data['y']):,} 条记录 · {len(data['names'])} 个原始特征")
            st.write(f"预测目标：{data['target']}")
            if data['source'] in PRESETS[2:4]:
                source_url = 'https://archive.ics.uci.edu/dataset/2/adult' if data['source'] == PRESETS[2] else 'https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008'
                st.caption(f'[UCI 数据来源与字段说明]({source_url})')
            with st.expander('预览前 10 行数据'):
                st.dataframe(data['preview'].head(10), width='stretch', hide_index=True, translate=False)
            st.subheader('特征含义')
            dictionary = feature_dictionary(data, data['names'])
            if data['source'] == '上传 CSV':
                dictionary = st.data_editor(dictionary, disabled=['字段'], hide_index=True, key='feature_meanings', translate_values=False)
            else:
                st.dataframe(dictionary, width='stretch', hide_index=True, translate_values=data['source'] != '上传 CSV')
            if data['preprocessing']:
                st.json(data['preprocessing'])
            confirmed = {**data, 'meanings': dict(zip(dictionary['字段'], dictionary['含义']))}
            st.button('确认数据与字段含义 → 数据划分', on_click=_go, args=(1, confirmed))
    elif stage == 1:
        data = state['data']
        official = data['source'] == PRESETS[2]
        st.write('Adult 使用官方训练／测试划分。' if official else 'Diabetes 按患者分组，同一患者不会跨集合。' if data['source'] == PRESETS[3] else '分类任务按标签分层随机划分；回归任务随机划分。')
        with st.form('split_form'):
            fraction = st.slider('测试集比例', 0.1, 0.5, 0.3, 0.05, disabled=official)
            seed = st.number_input('划分与评估种子', 0, 10000, 0)
            submit = st.form_submit_button('执行数据划分', type='primary')
        if submit:
            for key in ARTIFACTS[1:]: state.pop(key, None)
            try:
                commit(state, 'split', split_data(data, seed, fraction))
            except Exception as exc:
                st.error(f'划分失败：{exc}')
        if 'split' in state:
            split = state['split']
            c1, c2 = st.columns(2)
            c1.metric('训练集记录数', len(split['Xtr']))
            c2.metric('测试集记录数', len(split['Xte']))
            st.caption('真实预设数据的填补、标准化和类别编码仅在训练集拟合。')
            with st.expander('划分记录'): st.json(split['context'])
            st.button('下一步 → 训练模型', on_click=_go, args=(2,))
    elif stage == 2:
        data, split = state['data'], state['split']
        model_name = st.selectbox('模型', CLASSIFICATION_MODELS if data['task'] == 'classification' else REGRESSION_MODELS)
        with st.form('training_form_' + model_name):
            params = {}
            trees, depth = 100, 4
            if model_name in ('Random Forest', 'XGBoost', 'Gradient Boosting'):
                trees = st.slider('树的数量（树模型）', 10, 200, 60, 10)
                depth = st.slider('最大深度（树模型）', 2, 10, 4)
                if model_name != 'Random Forest':
                    params['learning_rate'] = st.number_input('学习率', 0.001, 1.0, 0.1, format='%.3f')
            elif model_name == 'MLP':
                layers = st.slider('隐藏层数', 1, 4, 2)
                width = st.selectbox('每层神经元数', [16, 32, 64, 128], index=1)
                params.update(hidden_layers=tuple([width] * layers),
                              learning_rate=st.number_input('学习率', 0.0001, 0.1, 0.001, format='%.4f'),
                              alpha=st.number_input('L2 正则强度', 0.0, 1.0, 0.0001, format='%.4f'),
                              max_iter=st.slider('最大迭代次数', 100, 2000, 300, 100))
                st.caption('多层感知机：使用训练集拟合标准化；SHAP 使用模型无关采样解释，计算较慢。')
            else:
                params['fit_intercept'] = st.checkbox('拟合截距', value=True)
                if model_name == 'Logistic Regression':
                    params['regularization'] = st.number_input('正则化参数 C', 0.001, 100.0, 1.0)
                    params['max_iter'] = st.slider('最大迭代次数', 100, 2000, 1000, 100)
            submit = st.form_submit_button('训练模型', type='primary')
        if submit:
            for key in ARTIFACTS[2:]: state.pop(key, None)
            with st.spinner('正在训练模型…'):
                try:
                    model = make_model(data['task'], model_name, trees, depth, split['seed'], **params)
                    model.fit(split['Xtr'], split['ytr'])
                    commit(state, 'trained', {'model': model, 'model_name': model_name,
                                             'score': model.score(split['Xte'], split['yte']),
                                             'evaluation': evaluate_model(model, split['Xte'], split['yte'], data['task']),
                                             'parameters': {**({'n_estimators': trees, 'max_depth': depth} if model_name in ('Random Forest', 'XGBoost', 'Gradient Boosting') else {}), **params}})
                except Exception as exc:
                    st.error(f'训练失败：{exc}')
                    if model_name == 'XGBoost': st.code('pip install -e ".[app,xgboost]"')
        if 'trained' in state:
            trained = state['trained']
            st.success(f"已训练：{trained['model_name']}")
            if trained['model_name'] == 'MLP':
                estimator = trained['model'].steps[-1][1]
                if estimator.n_iter_ >= estimator.max_iter:
                    st.warning('MLP 已达到最大迭代次数；请结合收敛情况决定是否增加训练预算。')
            evaluation = trained['evaluation']
            keys = ['Accuracy', 'Balanced accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'] if data['task'] == 'classification' else ['R²', 'MAE', 'RMSE']
            columns = st.columns(3)
            for j, name in enumerate(keys):
                columns[j % 3].metric(name, '—' if evaluation[name] is None else f"{evaluation[name]:.3f}")
            if data['task'] == 'classification':
                st.subheader('混淆矩阵')
                st.caption('Precision、Recall、F1 和 AUC 以编码 1 为正类；混淆矩阵的行是真实类别，列是预测类别。')
                labels = [str(v) for v in evaluation['classes']]
                st.dataframe(pd.DataFrame(evaluation['confusion_matrix'], index=labels, columns=labels), width='stretch')
                if evaluation['ROC AUC'] is None:
                    st.info('测试集只有一个类别，无法计算 ROC AUC。')
                else:
                    st.subheader('ROC 曲线')
                    curve = go.Figure(go.Scatter(x=evaluation['fpr'], y=evaluation['tpr'], name='ROC'))
                    curve.add_scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash'))
                    curve.update_layout(xaxis_title='False positive rate', yaxis_title='True positive rate', height=300)
                    st.plotly_chart(curve, width='stretch')
            st.caption('这是预测表现，不是解释质量。后续解释针对类别编码 1 的输出，回归则针对预测值。')
            if data['preprocessing'].get('class_mapping'): st.json(data['preprocessing']['class_mapping'])
            st.button('下一步 → 解释预测', on_click=_go, args=(3,))
    elif stage == 3:
        data, split, trained = state['data'], state['split'], state['trained']
        with st.form('explanation_form'):
            methods = st.multiselect('解释方法', ['SHAP', 'LIME'], default=['SHAP', 'LIME'])
            count = st.slider('解释样本数', 1, min(20, len(split['Xte'])), 1)
            lime_samples = st.slider('LIME 每样本采样数', 500, 5000, 1000, 500)
            background_size = st.slider('背景样本数', 10, 200, 50, 10)
            shap_samples = st.slider('KernelSHAP 采样预算', 50, 500, 100, 50)
            st.caption('KernelSHAP 预算用于 MLP 等模型；树模型和线性模型使用对应的专用解释器。')
            submit = st.form_submit_button('生成解释', type='primary')
        st.caption('从测试集可复现地抽样。仅运行所选方法；只选择一种时，未覆盖的质量检查会标为缺少结果。')
        if submit:
            for key in ARTIFACTS[3:]: state.pop(key, None)
            with st.spinner('正在生成解释…'):
                try: commit(state, 'explained', explain_model(trained, split, methods, count, lime_samples, background_size, shap_samples))
                except Exception as exc: st.error(f'解释失败：{exc}')
        if 'explained' in state:
            explained = state['explained']
            i = st.selectbox('查看样本', range(len(explained['X'])), format_func=lambda i: f"样本 {i+1}（测试行 {explained['positions'][i]}）")
            dictionary = feature_dictionary(data, split['names'])
            c1, c2, c3 = st.columns(3)
            c1.metric('真实目标值', f"{split['yte'][explained['positions'][i]]:g}")
            c2.metric('模型预测', f"{trained['model'].predict(explained['X'][i:i+1])[0]:g}")
            c3.metric(f"被解释的输出（{explained['output_space']}）", f"{scalar_predictor(trained['model'])(explained['X'][i:i+1])[0]:.3f}")
            fig = go.Figure()
            for method, key in [('SHAP', 'shap_attr'), ('LIME', 'lime_contrib')]:
                if method in explained['methods']:
                    fig.add_bar(name=method, x=split['names'], y=explained[key][i], customdata=dictionary['含义'], hovertemplate='%{x}<br>%{customdata}<br>贡献：%{y:.3f}<extra>%{fullData.name}</extra>')
            fig.update_layout(barmode='group', yaxis_title=f"贡献（{explained['output_space']}）", height=380)
            st.plotly_chart(fig, width='stretch', translate_descriptions=data['source'] != '上传 CSV')
            dictionary['当前输入值'] = explained['X'][i]
            st.dataframe(dictionary, width='stretch', hide_index=True, translate_values=data['source'] != '上传 CSV')
            st.caption('真实预设数据的数值已标准化，类别列为编码值；特征含义保持原始字段语义。')
            st.button('下一步 → 分析解释质量', on_click=_go, args=(4,))
    else:
        data, split, trained, explained = (state[k] for k in ARTIFACTS[:4])
        with st.form('quality_form'):
            use_shap, use_lime = 'SHAP' in explained['methods'], 'LIME' in explained['methods']
            repeats = st.slider('LIME 稳定性重复次数', 3, 10, 5, disabled=not use_lime)
            top_k = st.number_input('重要特征数量 top-k', 1, max(1,len(split['names'])-1), min(3,max(1,len(split['names'])-1)))
            sens_count = st.number_input('敏感度评估样本数', 1, len(explained['X']), 1, disabled=not use_shap)
            sens_budget = st.slider('每样本敏感度扰动次数', 1, 50, 10, disabled=not use_shap)
            radius = st.number_input('敏感度扰动半径', 0.001, 2.0, 0.1, format='%.3f', disabled=not use_shap)
            infidelity_budget = st.slider('局部拟合扰动次数', 50, 1000, 200, 50, disabled=not use_lime)
            random_budget = st.slider('随机移除对照次数', 5, 100, 20, 5, disabled=not use_shap)
            subgroup_limit = st.number_input('子群分析样本上限', 2, max(2,len(split['Xte'])), min(100,len(split['Xte'])), disabled=not use_shap)
            group_feature = st.selectbox('子群分组特征', split['names'], disabled=not use_shap)
            quality = dict(top_k=top_k, sensitivity_samples=sens_count, sensitivity_perturbations=sens_budget,
                           sensitivity_radius=radius, infidelity_perturbations=infidelity_budget,
                           comprehensiveness_random=random_budget)
            submit = st.form_submit_button('运行质量分析', type='primary')
        st.caption('复用上一步的模型、样本和解释。半径采用背景标准化坐标；增加样本或扰动次数会增加计算时间。')
        if submit:
            state.pop('analysis', None)
            with st.spinner('正在评估解释质量…'):
                try:
                    rng = np.random.default_rng(split['seed'])
                    dist = split['Xte'][rng.choice(len(split['Xte']), min(subgroup_limit, len(split['Xte'])), replace=False)]
                    out = battery(trained['model'], explained['X'], explained['bg'], split['names'], len(explained['X']), explained['lime_samples'], repeats, split['seed'], dist, split['names'].index(group_feature), methods=explained['methods'], precomputed=explained, quality_config=quality)
                    out.update(accuracy=trained['score'], task=data['task'], ground_truth=None, n_explain=len(explained['X']),
                               X_explain=explained['X'], preprocessing=data['preprocessing'], lime_samples=explained['lime_samples'], n_runs=repeats,
                               feature_dictionary=feature_dictionary(data, split['names']))
                    out['report'].context.update(split=split['context'], selected_test_positions=explained['positions'].tolist(),
                                                  dataset=data['source'], target=data['target'], model=trained['model_name'], model_evaluation=trained['evaluation'], model_parameters=trained['parameters'],
                                                  subgroup_limit=subgroup_limit,
                                                  feature_dictionary=out['feature_dictionary'].to_dict('records'))
                    commit(state, 'analysis', out)
                except Exception as exc: st.error(f'分析失败：{exc}')
        return state.get('analysis')
    return None
