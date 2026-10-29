"""读取分析结果，生成离线 HTML 报告。"""
from base64 import b64encode
from datetime import datetime, timezone, timedelta
from html import escape
from pathlib import Path
from textwrap import dedent, indent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / 'outputs'
ACCENT = '#1e3a5f'
MODEL_NAMES = {'baseline': '基准模型', 'nonlinear': '二次项扩展', 'pay_categorical': '还款状态分类'}


def configure_plots():
    """设置图表配色和中文字体。"""
    plt.rcParams.update(
        {
            'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans'],
            'axes.unicode_minus': False,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.prop_cycle': cycler(color=['#1e3a5f', '#c2410c', '#0f766e', '#8b5e83']),
            'axes.facecolor': '#fffdf8',
            'figure.facecolor': '#fffdf8',
            'axes.labelcolor': '#334155',
            'text.color': '#172033',
            'axes.titlepad': 15,
            'axes.grid': True,
            'grid.alpha': .18,
            'font.size': 10,
            'lines.linewidth': 2,
            'savefig.bbox': 'tight',
        },
    )


def read_table(name, columns):
    """读取结果表，检查文件是否为空、必要列是否齐全。"""
    path = OUT / name
    if not path.is_file():
        raise FileNotFoundError(f'缺少分析结果 {name}，请先运行本项目的 run.py。')
    frame = pd.read_csv(path)
    if frame.empty or not set(columns).issubset(frame.columns):
        raise ValueError(f'{name}为空或缺少必要字段：{columns}')
    return frame


def table_html(frame, names):
    """选择并重命名展示列；转义单元格中的 HTML 字符。"""
    data = frame[list(names)].copy().rename(columns=names)
    table = data.to_html(
        index=False,
        border=0,
        escape=True,
        float_format=lambda value: f'{value:,.4f}',
        na_rep='—',
    )
    return (
        '<div class="table-scroll" tabindex="0" aria-label="结果表，可横向滚动">\n'
        + indent(table, '    ')
        + '\n</div>'
    )


def figure_html(name, title, note):
    """将 PNG 嵌入页面，离线打开时无需另带图片。"""
    encoded = b64encode((OUT / name).read_bytes()).decode('ascii')
    return dedent(f"""\
        <figure>
            <figcaption>{escape(title)}</figcaption>
            <img src="data:image/png;base64,{encoded}" alt="{escape(title)}">
            <p class="note">{escape(note)}</p>
        </figure>
        """)


def figures_html(*figures):
    """把图表放入现有的双列布局，窄屏下由 CSS 切换为单列。"""
    return (
        '<div class="two">\n'
        + indent('\n'.join(figure.strip() for figure in figures), '    ')
        + '\n</div>'
    )

CSS = """
:root {
    --accent: ACCENT; --ink: #172033; --muted: #586577; --line: #dcdedc; --paper: #fffdf8;
    --bg: #f7f3ea;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.75 "Microsoft YaHei", "PingFang SC", sans-serif;
}
a { color: var(--accent); text-underline-offset: 4px; }
a:focus-visible, summary:focus-visible, [tabindex]:focus-visible {
    outline: 3px solid #c2410c; outline-offset: 4px;
}
.skip {
    position: absolute; top: -100px; left: 20px; background: white; padding: 8px;
}
.skip:focus { top: 10px; }
.wrap { max-width: 1220px; margin: auto; padding: 0 36px; }
.mast {
    display: flex; justify-content: space-between; gap: 14px; padding: 24px 0;
    border-bottom: 2px solid var(--accent); font-size: 13px; letter-spacing: .08em;
}
.brand { font-weight: 800; color: var(--accent); }
.hero {
    display: grid; grid-template-columns: 1.7fr 1fr; gap: 64px; align-items: center;
    padding: 46px 0 30px;
}
.eyebrow {
    font-size: 13px; color: var(--accent); font-weight: 700; letter-spacing: .1em;
}
h1 { font: 700 clamp(30px, 4vw, 46px)/1.35 "SimSun", "Songti SC", serif; margin: 12px 0 20px; }
h1 span {
    display: block; font-size: .62em; font-weight: 400; margin-top: 8px; color: var(--muted);
}
.lead { color: var(--muted); max-width: 650px; margin: 0; }
.hero-stat { border-left: 4px solid var(--accent); padding: 16px 0 16px 26px; }
.hero-stat strong {
    font: 600 66px/1.15 "Consolas", monospace; letter-spacing: -.07em; color: var(--accent);
}
.hero-stat p { margin: 4px 0; font-size: 14px; }
.hero-stat .note { max-width: 300px; }
.kpis {
    display: grid; grid-template-columns: repeat(4, 1fr); background: var(--paper);
    border: 1px solid var(--line); margin: 8px 0 24px;
}
.kpi { padding: 20px 22px; border-right: 1px solid var(--line); }
.kpi:last-child { border: 0; }
.kpi span { display: block; font-size: 13px; color: var(--muted); }
.kpi b { display: block; font: 600 27px/1.6 "Consolas", monospace; font-variant-numeric: tabular-nums; }
.kpi small { font-size: 12px; color: var(--muted); }
nav {
    display: flex; gap: 26px; flex-wrap: wrap; border-bottom: 1px solid var(--line);
    padding: 6px 0 18px; margin-bottom: 24px; font-size: 14px;
}
nav a { text-decoration: none; font-weight: 700; }
section { padding: 26px 0; border-bottom: 1px solid var(--line); scroll-margin-top: 20px; }
h2 { font-size: 24px; line-height: 1.4; margin: 0 0 8px; }
h3 { font-size: 18px; margin: 8px 0 12px; }
.section-note { color: var(--muted); margin: 0 0 20px; }
.finding {
    background: #e9eef2; border-left: 4px solid var(--accent); padding: 18px 24px; margin: 18px 0;
}
.finding p { margin: 0; }
.two {
    display: grid; grid-template-columns: 1fr 1fr; gap: 22px; align-items: start; min-width: 0;
}
.two>* { min-width: 0; }
figure {
    margin: 0; background: var(--paper); border: 1px solid var(--line); padding: 20px; min-width: 0;
}
figcaption { font-weight: 700; font-size: 17px; margin-bottom: 14px; }
figure img { display: block; width: 100%; height: auto; }
.note {
    font-size: 13px; line-height: 1.65; color: var(--muted); margin: 12px 0 0;
}
.table-scroll {
    overflow-x: auto; max-width: 100%; background: var(--paper); border: 1px solid var(--line);
    margin: 16px 0;
}
table {
    border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums;
}
th, td {
    padding: 12px 16px; text-align: right; white-space: nowrap; border-bottom: 1px solid #e3e5e7;
}
th { background: var(--accent); color: white; font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
tbody tr:nth-child(even) { background: #f0f3f4; }
tbody tr:last-child td { border-bottom: 0; }
.methods { background: var(--paper); padding: 22px; border: 1px solid var(--line); }
.methods dl { margin: 0; }
.methods dt { font-weight: 700; margin-top: 16px; }
.methods dt:first-child { margin-top: 0; }
.methods dd { margin: 3px 0 0; color: var(--muted); }
details {
    margin: 14px 0; padding: 14px 18px; background: var(--paper); border: 1px solid var(--line);
}
summary { cursor: pointer; font-weight: 700; min-height: 28px; }
ul { padding-left: 22px; }
li { margin: 8px 0; }
.downloads {
    display: flex; flex-wrap: wrap; gap: 10px 22px; margin: 16px 0;
}
.downloads a { font-size: 14px; }
footer { padding: 28px 0 38px; font-size: 12px; color: var(--muted); }

@media (max-width: 760px) {
    .wrap { padding: 0 18px; }
    .mast { font-size: 11px; flex-direction: column; gap: 4px; }
    .hero { grid-template-columns: 1fr; gap: 20px; padding: 28px 0; }
    .hero-stat { padding: 10px 0 10px 20px; }
    .hero-stat strong { font-size: 50px; }
    .kpis { grid-template-columns: 1fr 1fr; }
    .kpi { padding: 14px; border-bottom: 1px solid var(--line); }
    .kpi:nth-child(2) { border-right: 0; }
    .kpi b { font-size: 23px; }
    .two { grid-template-columns: 1fr; }
    nav { gap: 14px 20px; }
    section { padding: 24px 0; }
    figure { padding: 14px; }
    h2 { font-size: 22px; }
    th, td { padding: 10px 12px; }
}

@media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto; }
}

@media print {
    body { background: white; }
    .wrap { max-width: none; }
    nav, .downloads { display: none; }
    figure, .kpis { break-inside: avoid; }
}
"""


def write_report(title, subject, intro, hero_value, hero_label, hero_note, kpis, nav, body):
    """组装页面并保存报告。"""
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
    cards = '\n'.join(
        dedent(f"""\
            <div class="kpi">
                <span>{escape(label)}</span>
                <b>{escape(value)}</b>
                <small>{escape(note)}</small>
            </div>
            """).strip()
        for label, value, note in kpis
    )
    links = '\n'.join(f'<a href="#{key}">{escape(label)}</a>' for key, label in nav)
    html = dedent(f"""\
        <!doctype html>
        <html lang="zh-CN">
            <head>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width,initial-scale=1">
                <meta
                    http-equiv="Content-Security-Policy"
                    content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"
                >
                <title>{escape(title)}</title>
                <style>
                    {indent(CSS.replace('ACCENT', ACCENT), " " * 20).lstrip()}
                </style>
            </head>
            <body>
                <a class="skip" href="#content">跳到研究结果</a>
                <div class="wrap">
                    <header>
                        <div class="mast">
                            <span class="brand">{escape(subject)} · 研究结果</span>
                            <span>UCI公开数据 / 离线可视化报告</span>
                        </div>
                        <div class="hero">
                            <div>
                                <div class="eyebrow">{escape(title)}</div>
                                <h1>{title.split('：')[0]}<span>{escape(intro)}</span></h1>
                                <p class="lead">比较三种 Logistic 回归模型的测试表现，查看系数和诊断结果。完整结果表见页末。</p>
                            </div>
                            <aside class="hero-stat">
                                <p>{escape(hero_label)}</p>
                                <strong>{escape(hero_value)}</strong>
                                <p class="note">{escape(hero_note)}</p>
                            </aside>
                        </div>
                        <div class="kpis">
                            {indent(cards, " " * 28).lstrip()}
                        </div>
                        <nav aria-label="报告章节">
                            {indent(links, " " * 28).lstrip()}
                        </nav>
                    </header>
                    <main id="content">
                        {indent(body, " " * 24).lstrip()}
                    </main>
                    <footer>数据来源：UCI Machine Learning Repository · 报告生成时间：{stamp}（北京时间）<br/>页面图像已内嵌，可单独离线打开；CSV下载需保留同目录的结果文件。</footer>
                </div>
            </body>
        </html>
        """)
    target = OUT / '结果报告.html'
    target.write_text(html, encoding='utf-8')
    return target


def build_report():
    """检查模型指标和样本量，生成报告。"""
    configure_plots()
    summary = read_table('summary.csv', ['n', 'default_rate', 'train_n', 'test_n', 'lr_pvalue']).iloc[0]
    metrics = read_table(
        'model_comparison.csv',
        ['model', 'test_auc', 'test_pr_auc', 'test_brier', 'converged'],
    )
    if set(metrics.model) != set(MODEL_NAMES) or len(metrics) != 3 or not metrics.converged.eq(True).all():
        raise ValueError('模型必须包含三种唯一设定，且每个模型都收敛。')
    for col in ['test_auc', 'test_pr_auc', 'test_brier', 'test_accuracy_0_5']:
        if not np.isfinite(metrics[col]).all() or not metrics[col].between(0, 1).all():
            raise ValueError(f'模型指标{col}存在缺失或超出0—1范围。')
    if not (metrics.train_n.eq(summary.train_n).all() and metrics.test_n.eq(summary.test_n).all()
            and summary.train_n + summary.test_n == summary.n):
        raise ValueError('三模型样本量或训练测试人数不一致。')
    by_model = metrics.set_index('model')
    base, expanded, categorical = (by_model.loc[key] for key in MODEL_NAMES)
    display = metrics.copy()
    display['model'] = display.model.map(MODEL_NAMES)
    comparison = table_html(
        display,
        {
            'model': '模型设定',
            'test_auc': '测试AUC ↑',
            'test_pr_auc': '平均精度AP ↑',
            'test_brier': 'Brier误差 ↓',
            'test_accuracy_0_5': '准确率（阈值0.5）',
        },
    )
    figures = figures_html(
        figure_html('roc_curves.png', 'ROC曲线', '三个模型使用同一测试集，虚线为随机排序基准。'),
        figure_html('pr_curves.png', '查准率与召回率', 'AP为平均精度；水平线为测试集违约率。'),
    )
    diagnostics = figures_html(
        figure_html('calibration.png', '预测违约概率与实际违约率', '按预测概率分组比较，尚未检验校准截距和斜率。'),
        figure_html('coefficients.png', '基准模型的优势比', '变量每增加一个训练集标准差对应的条件优势比；采用常规Wald置信区间，反映关联而非因果关系。'),
    )
    coeff = read_table('baseline_coefficients.csv', ['term', 'or', 'or_ci_low', 'or_ci_high'])
    terms = ['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1']
    core = coeff[coeff.term.isin(terms)]
    core_table = table_html(
        core,
        {
            'term': '代表性变量',
            'or': '优势比OR',
            'or_ci_low': '95%区间下限',
            'or_ci_high': '95%区间上限',
            'p_value': 'p值',
        },
    )
    vif = read_table('vif.csv', ['term', 'vif'])
    vif_table = table_html(vif.head(6), {'term': '变量', 'vif': 'VIF（现有口径）'})
    calibration = read_table('calibration.csv', ['model', 'n', 'mean_predicted', 'observed_rate'])
    if calibration.groupby('model').n.sum().to_dict() != {key: summary.test_n for key in MODEL_NAMES}:
        raise ValueError('校准分箱人数与共同测试集人数不一致。')
    evidence = dedent(f"""\
        <section id="comparison">
            <h2>还款状态编码与模型表现</h2>
            <p class="section-note">三个模型共用同一次训练集与测试集划分，比较加入二次项和改变还款状态编码后的表现。</p>
            <div class="finding">
                <p>把还款状态由连续数值改为分类后，测试AUC从 <b>{base.test_auc:.4f}</b> 变为 <b>{categorical.test_auc:.4f}</b>，Brier从 <b>{base.test_brier:.4f}</b> 变为 <b>{categorical.test_brier:.4f}</b>。这是共同测试集上的描述性比较，尚未估计性能差异的置信区间。</p>
            </div>
            {indent(comparison, " " * 12).lstrip()}
            <p class="note">↑ 越大越好，↓ 越小越好。准确率依赖0.5阈值，不能代替排序与概率评价。</p>
            {indent(figures, " " * 12).lstrip()}
        </section>
        <section id="diagnostics">
            <h2>效应解释与诊断</h2>
            <p class="section-note">回归解释、排序能力与概率校准回答不同问题。</p>
            {indent(diagnostics, " " * 12).lstrip()}
            <details>
                <summary>查看5个变量的优势比和置信区间</summary>
                {indent(core_table, " " * 16).lstrip()}
            </details>
            <details>
                <summary>查看VIF最高的6项</summary>
                {indent(vif_table, " " * 16).lstrip()}
                <p class="note">VIF的辅助回归未包含截距，因此不能直接使用常规中心化VIF的阈值判断共线性。</p>
            </details>
        </section>
        <section id="method">
            <h2>样本、方法与局限</h2>
            <div class="two">
                <div class="methods">
                    <dl>
                        <dt>样本与时期</dt>
                        <dd>台湾地区历史信用卡样本；特征约为2005年4—9月，标签为次月违约。</dd>
                        <dt>训练与测试</dt>
                        <dd>按标签分层，75%训练 / 25%测试；随机种子42。标准化参数仅由训练集估计。</dd>
                        <dt>二次项模型</dt>
                        <dd>年龄与额度加入平方项；联合似然比检验p={summary.lr_pvalue:.4f}。测试AUC变化为{expanded.test_auc - base.test_auc:+.6f}，预测排序改善很小。</dd>
                    </dl>
                </div>
                <div class="methods">
                    <h3>分析局限</h3>
                    <ul>
                        <li>测试客户随机取自同一历史样本，尚未进行跨月份验证。</li>
                        <li>PAY按数值处理时假定相邻编码等距；分类模型中部分稀疏类别的系数估计不稳定，解释效应需谨慎。</li>
                        <li>系数采用常规标准误，校准采用分箱比较；尚未计算稳健标准误、检验校准斜率或进行重复验证。</li>
                    </ul>
                </div>
            </div>
        </section>
        """)
    files = [
        ('model_comparison.csv', '模型比较表'),
        ('baseline_coefficients.csv', '基准完整系数'),
        ('all_coefficients.csv', '三模型全部系数'),
        ('calibration.csv', '校准分箱'),
        ('vif.csv', 'VIF明细'),
        ('influence_top100.csv', '影响点明细'),
        ('summary.csv', '样本与检验摘要'),
    ]
    for filename, _ in files:
        if not (OUT / filename).is_file():
            raise FileNotFoundError(f'结果明细缺失：{filename}')
    downloads = '\n'.join(f'<a href="{name}" download>{label}</a>' for name, label in files)
    evidence += dedent(f"""\
        <section id="downloads">
            <h2>结果明细</h2>
            <p class="section-note">以下CSV包含完整系数、模型评价和诊断结果。</p>
            <div class="downloads">
                {indent(downloads, " " * 16).lstrip()}
            </div>
            <a href="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients">UCI数据来源</a>
        </section>
        """)
    return write_report(
        '信用卡违约研究：Logistic回归',
        '统计模型',
        '比较函数形式、还款状态编码与预测表现',
        f'{categorical.test_auc:.4f}',
        '还款状态分类模型 · 测试AUC',
        '采用相同留出客户进行比较；分数来自历史公开样本。',
        [
            ('客户样本', f'{int(summary.n):,}', '客户级观测'),
            ('违约率', f'{summary.default_rate:.2%}', '全样本标签比例'),
            ('训练 / 测试', f'{int(summary.train_n):,} / {int(summary.test_n):,}', '按标签分层，随机种子42'),
            ('编码后的AUC变化', f'{categorical.test_auc - base.test_auc:+.4f}', '相对连续编码基准模型'),
        ],
        [('comparison', '模型比较'), ('diagnostics', '效应与诊断'), ('method', '方法与范围'), ('downloads', '结果明细')],
        evidence,
    )


if __name__ == '__main__':
    print(f'结果核验通过：{build_report()}')
