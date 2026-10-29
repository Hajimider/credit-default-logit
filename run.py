"""信用卡违约 Logistic 回归分析。

读取 UCI 数据，比较三种模型设定，结果保存到 outputs。
"""

from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')  # 仅保存图片，不打开绘图窗口。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
from report import build_report, configure_plots


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "credit_default.zip"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
CONT_COLS = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
configure_plots()
MODEL_NAMES = {'baseline': '基准模型', 'nonlinear': '二次项扩展', 'pay_categorical': '还款状态分类'}


def load_data():
    """从压缩包读取 Excel，并检查分析所需字段是否齐全。"""
    if not RAW.exists():
        raise FileNotFoundError('找不到原始数据，请将 credit_default.zip 放入本项目的 data/raw 文件夹。')
    with zipfile.ZipFile(RAW) as zf:
        name = next((n for n in zf.namelist() if n.lower().endswith(".xls")), None)
        if not name:
            raise ValueError("The credit-card archive has no XLS workbook")
        with zf.open(name) as source:
            df = pd.read_excel(source, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    required = set(CAT_COLS + PAY_COLS + CONT_COLS + ["default payment next month"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing credit-card columns: {sorted(missing)}")
    return df


def prepare_frame(raw):
    """删除仅用于标识的 ID，整理分类编码，并拆出违约标签。"""
    df = raw.drop(columns=["ID"], errors="ignore").copy()
    target = df.pop("default payment next month").astype(int)
    for col in ["EDUCATION", "MARRIAGE"]:
        upper = 4 if col == "EDUCATION" else 3
        df[col] = df[col].where(df[col].between(1, upper), 0)
    return df, target


def build_design(df, train_i, pay_mode="continuous", nonlinear=False):
    """把原始字段转换为回归设计矩阵。

    用训练集的均值和标准差做标准化。
    pay_mode 指定还款状态按数值处理，或合并为四档后转为哑变量。
    """
    work = df.copy()
    dummy_cols = CAT_COLS.copy()
    if pay_mode == "categorical":
        # 本模型将 -2、-1、0 合为 no_delay，1、2 各为一档，3 及以上合为 three_plus。
        # 合并后每档样本更多，但原编码之间的差异也会丢失。
        for col in PAY_COLS:
            work[col] = np.select(
                [work[col] <= 0, work[col] == 1, work[col] == 2],
                ["no_delay", "one_month", "two_months"],
                default="three_plus",
            )
        dummy_cols += PAY_COLS
    # 类别转成 0/1 哑变量，每个字段省略一类作为参照组。
    x = pd.get_dummies(work, columns=dummy_cols, drop_first=True, dtype=float).astype(float)
    continuous = [c for c in CONT_COLS if c in x.columns]
    if pay_mode == "continuous":
        continuous += [c for c in PAY_COLS if c in x.columns]
    # 测试集沿用训练集参数，防止测试信息进入模型训练。
    means = x.iloc[train_i][continuous].mean()
    stds = x.iloc[train_i][continuous].std().replace(0, 1)
    x.loc[:, continuous] = (x[continuous] - means) / stds
    if nonlinear:
        # 加入平方项，检验年龄、额度与违约对数优势的非线性关系。
        x["AGE_sq"] = x["AGE"] ** 2
        x["LIMIT_BAL_sq"] = x["LIMIT_BAL"] ** 2
    return sm.add_constant(x.astype(float), has_constant="add"), means, stds


def fit_and_metrics(y, x, train_i, test_i, label, pay_mode, nonlinear):
    """拟合二项 Logistic 模型，并返回系数、预测概率和测试指标。"""
    # Binomial 默认使用 logit 链接，对应 Logistic 回归。
    fit = sm.GLM(y.iloc[train_i], x.iloc[train_i], family=sm.families.Binomial()).fit(maxiter=200)
    p_train = np.clip(fit.predict(x.iloc[train_i]), 1e-8, 1 - 1e-8)
    p_test = np.clip(fit.predict(x.iloc[test_i]), 1e-8, 1 - 1e-8)
    # AUC 衡量排序能力，AP 汇总查准率与召回率，Brier 衡量概率误差。
    # 准确率按预测概率 >= 0.5 判为违约计算。
    metrics = {"model": label, "pay_mode": pay_mode, "nonlinear": nonlinear, "train_n": len(train_i), "test_n": len(test_i), "train_aic": fit.aic, "train_loglik": fit.llf, "test_auc": roc_auc_score(y.iloc[test_i], p_test), "test_pr_auc": average_precision_score(y.iloc[test_i], p_test), "test_brier": brier_score_loss(y.iloc[test_i], p_test), "test_accuracy_0_5": accuracy_score(y.iloc[test_i], p_test >= 0.5), "converged": bool(getattr(fit, "converged", True)), "iterations": int(getattr(fit, "fit_history", {}).get("iteration", -1))}
    ci = fit.conf_int()
    coeff = pd.DataFrame({"model": label, "pay_mode": pay_mode, "term": fit.params.index, "coef": fit.params.values, "se": fit.bse.values, "p_value": fit.pvalues.values, "or": np.exp(np.clip(fit.params.values, -20, 20)), "or_ci_low": np.exp(np.clip(ci[0].values, -20, 20)), "or_ci_high": np.exp(np.clip(ci[1].values, -20, 20))})
    return fit, p_train, p_test, metrics, coeff


def calibration_rows(y, p, model):
    """按预测概率分十组，比较每组平均预测值与实际违约率。"""
    frame = pd.DataFrame({"y": np.asarray(y), "predicted": np.asarray(p)})
    frame["bin"] = pd.qcut(frame["predicted"], 10, duplicates="drop")
    out = frame.groupby("bin", observed=True).agg(n=("y", "size"), mean_predicted=("predicted", "mean"), observed_rate=("y", "mean")).reset_index(drop=True)
    out.insert(0, "model", model)
    return out


def vif_table(x, train_i):
    """计算训练集各变量的 VIF；辅助回归不含截距。"""
    values = x.iloc[train_i].drop(columns=["const"]).astype(float)
    rows = []
    for i, col in enumerate(values.columns):
        rows.append({"term": col, "vif": float(variance_inflation_factor(values.values, i))})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def influence_table(y, x, train_i, fit):
    """计算训练观测的影响度，并保留原始行号方便回查。"""
    glm = sm.GLM(y.iloc[train_i], x.iloc[train_i], family=sm.families.Binomial()).fit()
    frame = glm.get_influence().summary_frame().reset_index(drop=True)
    frame.insert(0, "row_index", np.asarray(train_i))
    frame.insert(1, "observed", y.iloc[train_i].to_numpy())
    frame["model_logit_deviance"] = float(fit.deviance)
    return frame.sort_values("cooks_d", ascending=False)


def plot_model_curves(y_test, predictions):
    """绘制三种模型在共同测试集上的 ROC 和 PR 曲线。"""
    plt.figure(figsize=(7.5, 6))
    for name, p in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, label=f"{MODEL_NAMES[name]} (AUC={roc_auc_score(y_test, p):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="#777")
    plt.xlabel("假阳性率")
    plt.ylabel("真阳性率（召回率）")
    plt.title("信用卡违约模型 ROC 曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "roc_curves.png", dpi=180)
    plt.close()
    plt.figure(figsize=(7.5, 6))
    for name, p in predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, p)
        plt.plot(recall, precision, label=f"{MODEL_NAMES[name]} (AP={average_precision_score(y_test, p):.3f})")
    plt.axhline(np.mean(y_test), linestyle="--", color="#777", label="测试集违约率")
    plt.xlabel("召回率")
    plt.ylabel("查准率")
    plt.title("信用卡违约模型 PR 曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "pr_curves.png", dpi=180)
    plt.close()


def main():
    """运行完整信用卡分析并保存所有结果文件。"""
    print('统计模型：正在读取信用卡数据……', flush=True)
    raw = load_data()
    df, y = prepare_frame(raw)
    if len(df) != 30000 or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Unexpected credit-card sample shape or target values")
    # 按违约标签分层抽样，使训练集和测试集的违约率接近。
    train_i, test_i = train_test_split(np.arange(len(y)), test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    base_x, means, stds = build_design(df, train_i, pay_mode="continuous", nonlinear=False)
    nonlinear_x, _, _ = build_design(df, train_i, pay_mode="continuous", nonlinear=True)
    categorical_x, _, _ = build_design(df, train_i, pay_mode="categorical", nonlinear=False)

    # 三个模型共用同一次样本划分。
    fits = {}
    outputs = []
    coeffs = []
    for label, x, pay_mode, nonlinear in [("baseline", base_x, "continuous", False), ("nonlinear", nonlinear_x, "continuous", True), ("pay_categorical", categorical_x, "categorical", False)]:
        print(f'正在估计：{MODEL_NAMES[label]}', flush=True)
        fit, p_train, p_test, metrics, coeff = fit_and_metrics(y, x, train_i, test_i, label, pay_mode, nonlinear)
        fits[label] = (fit, x, p_train, p_test)
        outputs.append(metrics)
        coeffs.append(coeff)
        # 分别保存各模型的系数和评价指标。
        coeff.drop(columns=["model", "pay_mode"]).to_csv(OUT / f"{label}_coefficients.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([metrics]).to_csv(OUT / f"{label}_metrics.csv", index=False, encoding="utf-8-sig")

    # 二次项模型包含基准模型，可做嵌套模型的似然比检验。
    # PAY 分类模型与基准模型不嵌套，不参加此检验。
    lr_stat = 2 * (fits["nonlinear"][0].llf - fits["baseline"][0].llf)
    lr_df = int(fits["nonlinear"][0].df_model - fits["baseline"][0].df_model)
    comparison = pd.DataFrame(outputs)
    comparison["lr_vs_baseline_stat"] = np.nan
    comparison["lr_vs_baseline_df"] = np.nan
    comparison["lr_vs_baseline_p"] = np.nan
    comparison.loc[comparison.model == "nonlinear", ["lr_vs_baseline_stat", "lr_vs_baseline_df", "lr_vs_baseline_p"]] = [lr_stat, lr_df, chi2.sf(lr_stat, lr_df)]
    comparison.to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(coeffs, ignore_index=True).to_csv(OUT / "all_coefficients.csv", index=False, encoding="utf-8-sig")

    # 在测试集上比较预测概率与实际违约率。
    calibration = pd.concat([calibration_rows(y.iloc[test_i], fits[name][3], name) for name in ["baseline", "nonlinear", "pay_categorical"]], ignore_index=True)
    calibration.to_csv(OUT / "calibration.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(7.5, 6))
    for name in ["baseline", "nonlinear", "pay_categorical"]:
        part = calibration[calibration.model == name]
        plt.plot(part.mean_predicted, part.observed_rate, marker="o", label=MODEL_NAMES[name])
    plt.plot([0, 1], [0, 1], "--", color="#777")
    plt.xlabel("分组平均预测概率")
    plt.ylabel("分组实际违约率")
    plt.title("信用卡违约模型校准曲线（测试集十等分）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "calibration.png", dpi=180)
    plt.close()

    print('正在计算校准、共线性及影响点诊断……', flush=True)
    vif_table(base_x, train_i).to_csv(OUT / "vif.csv", index=False, encoding="utf-8-sig")
    influence_table(y, base_x, train_i, fits["baseline"][0]).head(100).to_csv(OUT / "influence_top100.csv", index=False, encoding="utf-8-sig")
    plot_model_curves(y.iloc[test_i], {name: fits[name][3] for name in ["baseline", "nonlinear", "pay_categorical"]})

    coeff = pd.read_csv(OUT / "baseline_coefficients.csv")
    # 图中展示五个代表性变量，完整系数见 baseline_coefficients.csv。
    coeff = coeff[coeff.term.isin(['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1'])].sort_values("coef")
    labels = {'LIMIT_BAL': '信用额度', 'AGE': '年龄', 'PAY_0': '最近一期还款状态', 'BILL_AMT1': '最近一期账单金额', 'PAY_AMT1': '最近一期还款金额'}
    plt.figure(figsize=(8.5, 4.8))
    plt.errorbar(coeff["or"], np.arange(len(coeff)), xerr=[coeff["or"] - coeff["or_ci_low"], coeff["or_ci_high"] - coeff["or"]], fmt="o", color="#2878a5")
    plt.axvline(1, color="#444", lw=.8)
    plt.yticks(np.arange(len(coeff)), [labels[t] for t in coeff.term])
    plt.xscale("log")
    plt.xlabel("优势比及95%置信区间（每增加一个训练集标准差）")
    plt.title("基准模型：代表性变量的条件优势比")
    plt.tight_layout()
    plt.savefig(OUT / "coefficients.png", dpi=180)
    plt.close()

    summary = {
        "n": len(df), "default_rate": float(y.mean()), "train_n": len(train_i), "test_n": len(test_i),
        "lr_stat": float(lr_stat), "lr_df": lr_df, "lr_pvalue": float(chi2.sf(lr_stat, lr_df)),
        "baseline_train_aic": float(fits["baseline"][0].aic), "nonlinear_train_aic": float(fits["nonlinear"][0].aic),
        "feature_count_baseline": int(base_x.shape[1] - 1), "missing_cells": int(df.isna().sum().sum()),
        "random_state": RANDOM_STATE, "pay_0_sd_train": float(stds["PAY_0"]),
        "pay_0_or_per_sd": float(np.exp(fits["baseline"][0].params["PAY_0"])),
        "pay_0_or_per_raw_level": float(np.exp(fits["baseline"][0].params["PAY_0"] / stds["PAY_0"])),
        "diagnostics": ["vif", "calibration_deciles", "glm_influence_top100"],
    }
    summary['diagnostics'] = '；'.join(summary['diagnostics'])
    pd.DataFrame([summary]).to_csv(OUT / 'summary.csv', index=False, encoding='utf-8-sig')
    path = build_report()
    print(f'完成：{len(df):,}名客户；测试集最佳AUC为{comparison.test_auc.max():.4f}。')
    print(f'报告：{path}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='信用卡违约分析与结果报告')
    parser.add_argument('--check', action='store_true', help='核验已有结果并重建报告，不重估模型')
    args = parser.parse_args()
    if args.check:
        print(f'结果核验通过，可视化报告：{build_report()}')
    else:
        main()
