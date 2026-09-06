# -*- coding: utf-8 -*-
"""Presentation-only translation: canonical option values and exported schemas stay stable."""
import streamlit as _st
import pandas as pd
import re
import inspect
from functools import lru_cache

# Long phrases are replaced first, so dynamic strings reuse the same vocabulary.
EN = {
'MLP 已达到最大迭代次数；请结合收敛情况决定是否增加训练预算。':'MLP reached its iteration limit; check convergence before increasing the training budget.',
'混淆矩阵':'Confusion matrix', 'ROC 曲线':'ROC curve',
'选择数据':'Choose data', '划分数据':'Split data', '训练模型':'Train model', '解释预测':'Explain predictions', '分析解释质量':'Assess explanation quality',
'数据 → 划分 → 训练 → 解释 → 质量分析':'Data → Split → Train → Explain → Assess',
'点击按钮执行当前步骤。重新执行上游步骤会清除后续结果；修改输入后请先提交当前步骤。':'Run each step using its button. Re-running an earlier step clears dependent results. Submit edited settings before continuing.',
'模拟：相关代理特征':'Synthetic: correlated proxy', '模拟：独立噪声特征':'Synthetic: independent noise', 'Adult 收入分类':'Adult income classification', 'Diabetes 再入院分类':'Diabetes readmission classification',
'上传 CSV':'Upload CSV', '上传数据表':'Upload a data table', '数据集':'Dataset', '数据生成种子':'Data generation seed',
'x0、x1、x2 是人工构造的变量，不对应年龄、收入等真实业务字段。此处保留原始编号，展示其生成角色。':'x0, x1 and x2 are synthetic variables, not age or income. Their original names are retained and their roles in data generation are shown.',
'目标列':'Target column', '任务类型':'Task', '二分类':'Binary classification', '回归':'Regression',
'当前上传流程仅保留数值特征，排除含缺失值的行。字段含义可在加载后填写。':'Uploads currently retain numeric features and remove rows with missing values. Add feature descriptions after loading.',
'首次使用需下载 UCI 数据集。Adult 保留官方测试集；Diabetes 按患者划分。':'UCI datasets are downloaded on first use. Adult keeps its official holdout; Diabetes is split by patient.',
'加载数据':'Load data', '正在加载数据…':'Loading data…', '加载失败：':'Loading failed: ', '无法读取 CSV：':'Cannot read CSV: ',
'已加载：':'Loaded: ', ' 条记录 · ':' rows · ', ' 个原始特征':' raw features', '预测目标：':'Prediction target: ',
'预览前 10 行数据':'Preview first 10 rows', '特征含义':'Feature descriptions', '字段':'Feature', '含义':'Description',
'确认数据与字段含义 → 数据划分':'Confirm data and descriptions → Split data', 'UCI 数据来源与字段说明':'UCI data source and variable descriptions',
'人工生成的二分类标签；1 为模型所解释的目标类别':'Synthetic binary target; explanations concern class 1',
'1：年收入 >50K；0：≤50K':'1: annual income >50K; 0: ≤50K', '1：再次入院（<30 或 >30）；0：未再次入院':'1: readmitted (<30 or >30); 0: not readmitted',
'人工变量：参与标签生成，含线性和平方项':'Synthetic variable: enters the target mechanism with linear and squared terms',
'人工变量：参与标签生成，并与 x0 交互':'Synthetic variable: enters the target mechanism and interacts with x0',
'x0 的相关代理；不直接参与标签生成':'Correlated proxy for x0; does not directly enter the target mechanism',
'独立噪声变量；不参与标签生成':'Independent noise; does not enter the target mechanism',
'待填写业务含义':'Add a domain description', '未提供说明':'No description provided', '数据集字段':'Dataset variable',
'年龄；Diabetes 中为年龄区间':'Age; age band in Diabetes', '教育程度的数值编码':'Numeric education-level code',
'资本收益':'Capital gain', '资本损失':'Capital loss', '每周工作小时数':'Hours worked per week', '工作类别':'Work class',
'婚姻状态':'Marital status', '职业类别':'Occupation', '家庭关系类别':'Household relationship',
'数据集记录的种族类别':'Race category recorded in the dataset', '数据集记录的性别':'Sex/gender recorded in the dataset',
'原籍国家':'Native country', '本次住院天数':'Days in the current hospital stay', '检验次数':'Laboratory procedure count',
'非检验操作次数':'Non-laboratory procedure count', '用药数量':'Medication count', '此前一年门诊次数':'Outpatient visits in the preceding year',
'此前一年急诊次数':'Emergency visits in the preceding year', '此前一年住院次数':'Inpatient visits in the preceding year',
'记录的诊断数量':'Number of recorded diagnoses', '入院类型编码':'Admission-type code', '血糖检查结果类别':'Serum glucose result category',
'糖化血红蛋白检查结果类别':'HbA1c result category', '糖尿病用药是否调整（1=调整）':'Diabetes medication changed (1=yes)',
'是否使用糖尿病药物（1=是）':'Diabetes medication prescribed (1=yes)', '是否使用该药物（No=0，其他记录=1）':'Medication prescribed (No=0; other records=1)',
'；类别指示：':'; category indicator: ', '（1=该类别）':' (1=this category)',
'Adult 使用官方训练／测试划分。':'Adult uses its official training/test split.',
'Diabetes 按患者分组，同一患者不会跨集合。':'Diabetes is split by patient, with no patient shared between partitions.',
'分类任务按标签分层随机划分；回归任务随机划分。':'Classification uses a stratified random split; regression uses a random split.',
'测试集比例':'Test fraction', '划分与评估种子':'Split and evaluation seed', '执行数据划分':'Run split', '划分失败：':'Split failed: ',
'训练集记录数':'Training rows', '测试集记录数':'Test rows', '划分记录':'Split details',
'真实预设数据的填补、标准化和类别编码仅在训练集拟合。':'For real presets, imputation, scaling and categorical encoding are fitted on training rows only.',
'下一步 → 训练模型':'Next → Train model', '下一步 → 解释预测':'Next → Explain predictions', '下一步 → 分析解释质量':'Next → Assess explanation quality',
'模型':'Model', '树的数量（树模型）':'Number of trees', '最大深度（树模型）':'Maximum tree depth', '学习率':'Learning rate',
'隐藏层数':'Hidden layers', '每层神经元数':'Neurons per layer', 'L2 正则强度':'L2 regularization', '最大迭代次数':'Maximum iterations',
'拟合截距':'Fit intercept', '正则化参数 C':'Regularization parameter C',
'多层感知机：使用训练集拟合标准化；SHAP 使用模型无关采样解释，计算较慢。':'Multilayer perceptron: scaling is fitted on training data. Model-agnostic SHAP sampling can be slow.',
'正在训练模型…':'Training model…', '训练失败：':'Training failed: ', '已训练：':'Trained: ',
'测试集准确率':'Test accuracy', '测试集 R²':'Test R²',
'Precision、Recall、F1 和 AUC 以编码 1 为正类；混淆矩阵的行是真实类别，列是预测类别。':'Precision, recall, F1 and AUC use encoded class 1 as positive. Confusion-matrix rows are true classes; columns are predicted classes.',
'测试集只有一个类别，无法计算 ROC AUC。':'ROC AUC is unavailable because the test set contains only one class.',
'这是预测表现，不是解释质量。后续解释针对类别编码 1 的输出，回归则针对预测值。':'These are predictive performance metrics. Explanations concern encoded class 1 for classification and the predicted value for regression.',
'解释方法':'Explanation methods', '解释样本数':'Instances to explain', 'LIME 每样本采样数':'LIME samples per instance',
'背景样本数':'Background rows', 'KernelSHAP 采样预算':'KernelSHAP sampling budget',
'KernelSHAP 预算用于 MLP 等模型；树模型和线性模型使用对应的专用解释器。':'The KernelSHAP budget applies to models such as MLP. Tree and linear models use their specialized explainers.',
'生成解释':'Generate explanations', '正在生成解释…':'Generating explanations…', '解释失败：':'Explanation failed: ',
'从测试集可复现地抽样。仅运行所选方法；只选择一种时，未覆盖的质量检查会标为缺少结果。':'Instances are sampled reproducibly from the holdout. Only selected methods run; uncovered checks are marked unavailable.',
'查看样本':'View instance', '样本 ':'Instance ', '（测试行 ':' (test row ', '真实目标值':'Observed target', '模型预测':'Model prediction',
'被解释的输出（':'Explained output (', '贡献（':'Contribution (', '贡献：':'Contribution: ',
'当前输入值':'Current input value',
'真实预设数据的数值已标准化，类别列为编码值；特征含义保持原始字段语义。':'Real-preset numeric values are standardized and category columns are encoded; descriptions retain the original field semantics.',
'LIME 稳定性重复次数':'LIME stability repeats', '重要特征数量 top-k':'Important feature count (top-k)',
'敏感度评估样本数':'Instances for sensitivity', '每样本敏感度扰动次数':'Sensitivity perturbations per instance',
'敏感度扰动半径':'Sensitivity radius', '局部拟合扰动次数':'Infidelity perturbations', '随机移除对照次数':'Random-removal comparisons',
'子群分析样本上限':'Subgroup sample limit', '子群分组特征':'Subgroup feature',
'复用上一步的模型、样本和解释。半径采用背景标准化坐标；增加样本或扰动次数会增加计算时间。':'Reuses the model, instances and explanations from the previous step. Radius uses background-standardized coordinates; larger budgets take longer.',
'运行质量分析':'Run quality analysis', '正在评估解释质量…':'Assessing explanation quality…', '分析失败：':'Analysis failed: ',
'至少选择一种解释方法。':'Select at least one explanation method.', '至少需要 40 条完整记录和 2 个数值特征。':'At least 40 complete rows and two numeric features are required.',
'当前支持二分类；每类至少需要两条记录。':'Binary classification requires at least two rows per class.',
'特征和回归目标必须为有限数值。':'Features and regression targets must be finite.',
'解释评估':'Explanation assessment', 'explaintrust · 了解解释的表现，以及哪些结果需要进一步核查':'explaintrust · Understand explanation performance and what needs further investigation',
'当前检查未发现问题':'No issues detected by current checks', '评估证据不完整，请先查看缺失与失败的检查':'Evidence is incomplete; review missing and failed checks',
'部分检查需要关注':'Some checks need attention', '多项检查未通过':'Multiple checks failed',
'结论来自忠实性、敏感度和重复运行稳定性检查；方法间分歧与子群差异单独呈现。':'The conclusion uses faithfulness, sensitivity and repeat stability. Method and subgroup differences are shown separately.',
'通过的检查':'Checks passed', '已解释样本':'Explained instances', '模型测试准确率':'Model test accuracy', '模型测试 R²':'Model test R²',
'覆盖范围：忠实性 ':'Coverage: faithfulness ', ' 个样本 · 稳定性 ':' instances · stability ', ' 个样本 · 敏感度 ':' instances · sensitivity ',
' 个样本。通过检查不等于证明解释正确；模型预测表现也不等于解释质量。':' instances. Passing checks does not prove correctness; predictive performance is distinct from explanation quality.',
'质量检查':'Quality checks', '单个样本':'Individual instance', '方法与子群差异':'Method and subgroup differences', '评估详情与导出':'Details and export',
'先看需要关注的检查':'Checks needing attention', '当前没有失败、提醒或缺失的质量检查。':'No failed, warning or missing checks.',
'查看全部质量检查':'View all quality checks', '状态根据当前配置阈值生成；不适用项不计入检查总数。具体阈值与原始说明可在评估详情查看。':'Statuses use configured thresholds; inapplicable checks are excluded from the count. See details for thresholds and original metric definitions.',
'这个样本的解释是什么？':'What explains this prediction?', '选择样本':'Select instance',
'柱高表示对模型输出的贡献；正负表示贡献方向。两种方法分配贡献的差异需要结合模型与数据理解。':'Bar height shows contribution to model output; sign shows direction. Interpret attribution differences in the context of the model and data.',
'归因贡献':'Attribution contribution', '查看逐特征对比':'View feature comparison', '按 SHAP–LIME 绝对差值排序；差异标记不等同于解释错误。':'Sorted by absolute SHAP–LIME difference; a flag does not imply an incorrect explanation.',
'特征':'Feature', '绝对差值':'Absolute gap', '未触发差异标记':'No difference flag', '信噪比':'Signal-to-noise ratio',
'当前只选择了一种解释方法，因此不生成方法间差异标记。':'Only one method was selected; cross-method flags are not produced.',
'差异描述，不参与总体判定':'Descriptive differences, excluded from the conclusion',
'方法可能关注不同的特征，子群也可能存在真实差异。差异较大不直接意味着不可靠，一致也不代表正确。':'Methods may emphasize different features and subgroups may genuinely differ. Large differences do not establish unreliability, and agreement does not prove correctness.',
'查看子群特征重要性':'View subgroup feature importance', '分组特征：':'Grouping feature: ', ' · 覆盖 ':' · coverage ',
' 个样本 · ':' instances · ', ' 个子群。数值为各子群的平均绝对归因。':' subgroups. Values are mean absolute attributions per subgroup.',
'子群':'Subgroup', '组 ':'Group ', '这里描述选定子群之间的异质性，不是源数据到目标数据的分布漂移检验。':'This describes heterogeneity across selected subgroups, not a source-to-target distribution-shift test.',
'评估范围与计算设置':'Evaluation scope and settings', '已选方法：':'Selected methods: ', '；背景样本 ':'; background rows ', ' 行。':'.',
'LIME 每个样本采样 ':'LIME samples per instance: ', ' 次，稳定性重复 ':'; stability repeats: ', ' 次。':'.',
'原始结论、指标与阈值':'Original conclusion, metrics and thresholds', '完整方法与计算参数':'Full method and computation parameters',
'数据预处理记录':'Preprocessing audit', '当前版本排除非数值特征列和含缺失值的行，请结合任务核查这些排除是否合适。':'Non-numeric features and rows with missing values are excluded. Check that these exclusions suit your task.',
'合成数据的已知机制':'Known synthetic mechanism', '角色':'Role',
'x0、x1 参与标签生成；x2 是相关代理特征。SHAP 与 LIME 描述模型贡献，不提供因果证据。':'x0 and x1 enter target generation; x2 is a correlated proxy. SHAP and LIME describe model contributions, not causal evidence.',
'下载完整报告 JSON':'Download full report JSON', '报告保留原始指标和计算参数。':'The report retains original metrics and settings. ',
'逐特征对比对应当前样本 ':'Feature comparison corresponds to instance ', '未选择的方法以缺失检查呈现。':'Unselected methods appear as unavailable checks.',
'检查项':'Check', '状态':'Status', '数值':'Value', '如何理解':'Interpretation', '通过':'Passed', '需关注':'Attention', '未通过':'Failed', '缺少结果':'Unavailable', '不适用':'Not applicable', '描述性指标':'Descriptive',
'SHAP 移除效果':'SHAP removal effect', 'SHAP 关键特征检查':'SHAP comprehensiveness', 'LIME 局部拟合':'LIME local fidelity',
'解释敏感度':'Explanation sensitivity', '重复运行排名稳定性':'Repeat rank stability', '重复运行方向稳定性':'Repeat sign stability',
'方法间方向差异':'Cross-method sign difference', '方法间排名一致性':'Cross-method rank agreement', '方法间重要特征重合':'Cross-method top-feature overlap',
'方法间贡献大小差异':'Cross-method magnitude difference', '子群排名一致性':'Subgroup rank agreement', '子群重要特征变化率':'Subgroup top-feature flip rate',
'特征归因是否与移除该特征后的模型输出变化一致。':'Does attribution agree with the output change after feature removal?',
'移除重要特征是否比随机移除同样数量的特征影响更大。':'Does removing important features change predictions more than removing random features?',
'局部解释能否近似模型的输出变化；数值越低，误差越小。':'How well does the local explanation approximate output changes? Lower values mean less error.',
'各已评估样本的扰动最大归因变化的均值；覆盖数量见摘要。':'Mean sampled maximum attribution change across evaluated instances; see coverage in the summary.',
'不同随机种子下，重要特征的排名是否一致。':'Do important-feature rankings agree across random seeds?',
'不同随机种子下，重要特征的贡献方向是否一致。':'Do important-feature contribution signs agree across seeds?',
'两种方法给出不同贡献方向的特征比例。':'Fraction of features assigned different contribution signs.',
'重要特征排名有多相似；相似程度不代表解释质量。':'Similarity of important-feature rankings; similarity is not explanation quality.',
'两种方法选出的重要特征有多少重合。':'Overlap between the important features selected by each method.',
'重要特征的相对归因差值；需要结合方法与任务解释。':'Relative attribution gap for important features; interpret in method and task context.',
'各子群的重要特征排序有多相似。':'Similarity of importance rankings across subgroups.',
'与参考子群相比，重要特征集合发生变化的比例。':'Fraction of subgroups whose top-feature set differs from the reference.',
'当前特征数或 top-k 设置下，此项比较不适用；详见原始指标说明。':'Not applicable for the current feature count or top-k; see original metric definitions.',
'未计算或返回了无法解释的非有限值；详见原始指标说明。':'Not computed or returned an uninterpretable non-finite value; see original metric definitions.',
}
ZH = {'Random Forest':'随机森林', 'Gradient Boosting':'梯度提升', 'Logistic Regression':'逻辑回归', 'Linear Regression':'线性回归', 'MLP':'多层感知机（MLP）',
      'Accuracy':'准确率', 'Balanced accuracy':'平衡准确率', 'Precision':'精确率', 'Recall':'召回率',
      'ROC AUC':'ROC AUC', 'MAE':'平均绝对误差（MAE）', 'RMSE':'均方根误差（RMSE）',
      'False positive rate':'假阳性率', 'True positive rate':'真阳性率', 'Random':'随机基线'}


@lru_cache(maxsize=2)
def _pattern(language):
    mapping = EN if language == "en" else ZH
    return re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))

def tr(text, language=None):
    if not isinstance(text, str): return text
    language = language or _st.session_state.get('language', 'zh')
    mapping = EN if language == 'en' else ZH
    if text in mapping: return mapping[text]
    # One-pass replacement prevents translations from being translated again.
    return _pattern(language).sub(lambda m: mapping[m.group()], text)


class UI:
    """Translate presentation calls while preserving data and callback arguments."""
    def __init__(self, target): self.target = target
    def __enter__(self): self.target.__enter__(); return self
    def __exit__(self, *args): return self.target.__exit__(*args)
    def __getattr__(self, name):
        attr = getattr(self.target, name)
        if name == 'sidebar': return UI(attr)
        if name in ('session_state', 'cache_data', 'cache_resource'): return attr
        if name == 'columns':
            return lambda *a, **k: [UI(c) for c in attr(*a, **k)]
        if name in ('form', 'container', 'expander', 'spinner'):
            def context(*a, **k):
                if name in ('expander', 'spinner') and a: a = (tr(a[0]), *a[1:])
                return UI(attr(*a, **k))
            return context
        if name == 'tabs': return lambda labels: [UI(t) for t in attr([tr(v) for v in labels])]
        if name in ('selectbox', 'multiselect', 'radio'):
            def choice(label, options, *a, **k):
                formatter = k.pop('format_func', lambda v: v)
                language = _st.session_state.get('language', 'zh')
                k.setdefault('key', f'ui_{name}_{label}')
                saved = _st.session_state.get('_language_inputs', {})
                if k['key'] in saved:
                    value = saved.pop(k['key'])
                    if name == 'multiselect':
                        k['default'] = value
                    else:
                        k['index'] = list(options).index(value) if value in options else 0
                return attr(tr(label), options, *a, format_func=lambda v: tr(formatter(v), language), **k)
            return choice
        if name in ('slider', 'number_input', 'checkbox', 'button', 'file_uploader', 'text_input'):
            def widget(label, *a, **k):
                k.setdefault('key', f'ui_{name}_{label}')
                saved = _st.session_state.get('_language_inputs', {})
                if k['key'] in saved:
                    bound = inspect.signature(attr).bind_partial(tr(label), *a, **k)
                    bound.arguments['value'] = saved.pop(k['key'])
                    return attr(*bound.args, **bound.kwargs)
                return attr(tr(label), *a, **k)
            return widget
        if name in ('dataframe', 'data_editor'):
            def table(data, *a, **k):
                translate = k.pop('translate', True)
                translate_values = k.pop('translate_values', True)
                old_columns = list(data.columns) if isinstance(data, pd.DataFrame) else []
                if name == 'data_editor' and translate:
                    k.setdefault('column_config', {c: tr(c) for c in old_columns})
                    translate = False
                if isinstance(data, pd.DataFrame) and translate:
                    old_columns = list(data.columns)
                    data = data.copy()
                    for col in data:
                        if translate_values and col in ('含义', '检查项', '状态', '如何理解', '角色') and data[col].dtype == object:
                            data[col] = data[col].map(lambda v: tr(v) if isinstance(v,str) else v)
                    data.columns = [tr(c) for c in old_columns]
                    if 'disabled' in k and isinstance(k['disabled'],list): k['disabled'] = [tr(v) for v in k['disabled']]
                result = attr(data, *a, **k)
                if name == 'data_editor' and isinstance(result, pd.DataFrame): result.columns = old_columns
                return result
            return table
        if name == 'plotly_chart':
            def chart(fig, *a, **k):
                import plotly.graph_objects as go
                translate_descriptions = k.pop('translate_descriptions', True)
                fig = go.Figure(fig)
                for trace in fig.data:
                    if trace.name: trace.name = tr(trace.name)
                    if trace.hovertemplate: trace.hovertemplate = tr(trace.hovertemplate)
                    if trace.customdata is not None and translate_descriptions:
                        trace.customdata = [tr(v) if isinstance(v,str) and v in EN else v for v in trace.customdata]
                for axis in ('xaxis','yaxis'):
                    title = getattr(fig.layout, axis).title.text
                    if title: getattr(fig.layout, axis).title.text = tr(title)
                return attr(fig, *a, **k)
            return chart
        if name in ('title','header','subheader','caption','write','markdown','info','success','warning','error','metric','form_submit_button','download_button'):
            def text(*a, **k):
                if a: a = (tr(a[0]), *a[1:])
                return attr(*a, **k)
            return text
        return attr


st = UI(_st)
