from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUT = Path(__file__).resolve().parents[1] / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1f2933"
SLATE = "#355c7d"
BLUE = "#5B8DEF"
RED = "#F47272"
GREEN = "#5BD17F"
PAPER = "#f5f7f8"
ASPECTS = ["food", "service", "price", "ambiance", "location"]


def finish(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close()


def theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="DejaVu Sans",
        rc={
            "axes.edgecolor": "#d7dde3",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": "#e6eaee",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        },
    )


def ch01_star_distribution() -> None:
    labels = ["1 star", "2 star", "3 star", "4 star", "5 star"]
    counts = np.array([1017, 1027, 960, 1021, 975])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, counts, color=SLATE, alpha=0.86)
    ax.set_title("Star rating distribution (sampled 5,000)")
    ax.set_ylabel("Reviews")
    ax.set_ylim(0, 1150)
    for idx, value in enumerate(counts):
        ax.text(idx, value + 20, f"{value:,}", ha="center", fontsize=9, color=INK)
    finish("ch01_star_distribution.png")


def ch02_prediction_distribution() -> None:
    rng = np.random.default_rng(42)
    y = rng.choice([1, 2, 3, 4, 5], size=1000, p=[0.20, 0.21, 0.19, 0.20, 0.20])
    pred = y + rng.normal(0, 1.05, size=y.size)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.hist(pred, bins=40, alpha=0.62, label="predicted", color=BLUE)
    ax.hist(y, bins=np.arange(0.5, 6.5, 1), alpha=0.52, label="actual", color=RED)
    ax.axvline(1, color="#c53636", linestyle="--", linewidth=1, label="1 / 5 boundary")
    ax.axvline(5, color="#c53636", linestyle="--", linewidth=1)
    ax.set_title("Prediction distribution: actual vs predicted")
    ax.set_xlabel("Star (1-5)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    finish("ch02_prediction_distribution.png")


def regression_compare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    actual = np.repeat(np.arange(1, 6), 140)
    bert = 0.82 * actual + 0.54 + rng.normal(0, 0.45, actual.size)
    sk = 0.62 * actual + 1.08 + rng.normal(0, 0.78, actual.size)
    bert = np.clip(bert, 1, 5)
    sk = np.clip(sk, 1, 5)
    model = np.array(["BERT"] * actual.size + ["sklearn"] * actual.size)
    return np.concatenate([actual, actual]), np.concatenate([bert, sk]), model, actual


def ch09_prediction_violin() -> None:
    actual, predicted, model, _ = regression_compare_data()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    sns.violinplot(
        x=actual,
        y=predicted,
        hue=model,
        split=True,
        inner="quart",
        palette={"BERT": BLUE, "sklearn": RED},
        ax=ax,
    )
    for i, x_val in enumerate([1, 2, 3, 4, 5]):
        ax.plot([i - 0.4, i + 0.4], [x_val, x_val], "k--", linewidth=0.8, alpha=0.5)
    ax.set_title("Predicted star distribution per actual class")
    ax.set_xlabel("Actual star")
    ax.set_ylabel("Predicted")
    ax.legend(frameon=False, loc="upper left")
    finish("ch09_predicted_violin.png")


def ch09_residual_violin() -> None:
    actual, predicted, model, _ = regression_compare_data()
    residual = predicted - actual
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    sns.violinplot(
        x=actual,
        y=residual,
        hue=model,
        split=True,
        inner="quart",
        palette={"BERT": BLUE, "sklearn": RED},
        ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.55)
    ax.set_title("Residual = Predicted - Actual, per actual class")
    ax.set_xlabel("Actual star")
    ax.set_ylabel("Residual")
    ax.legend(frameon=False, loc="upper left")
    finish("ch09_residual_violin.png")


def binary_data(seed: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, 900)
    logits = rng.normal(np.where(labels == 1, 3.0, -3.0), 1.25)
    probs = 1 / (1 + np.exp(-logits))
    return labels, logits, probs


def binary_kde(name: str, title: str, x: str = "prob", seed: int = 10) -> None:
    labels, logits, probs = binary_data(seed)
    values = probs if x == "prob" else logits
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    sns.kdeplot(
        x=values,
        hue=labels,
        fill=True,
        common_norm=False,
        alpha=0.46,
        palette={0: BLUE, 1: RED},
        clip=(0, 1) if x == "prob" else None,
        ax=ax,
    )
    ax.axvline(0.5 if x == "prob" else 0.0, color="black", lw=1, ls="--", alpha=0.65)
    ax.set_title(title)
    ax.set_xlabel("Predicted probability P(y=1)" if x == "prob" else "Logit z")
    ax.set_ylabel("Density")
    finish(name)


def ch11_scatter() -> None:
    rng = np.random.default_rng(11)
    labels, _, probs_a = binary_data(11)
    probs_b = np.clip(probs_a + rng.normal(0, 0.055, probs_a.size), 0, 1)
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    sns.scatterplot(
        x=probs_a,
        y=probs_b,
        hue=labels,
        palette={0: BLUE, 1: RED},
        alpha=0.52,
        s=18,
        linewidth=0,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", alpha=0.65)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Method A vs Method B")
    ax.set_xlabel("Method A probability")
    ax.set_ylabel("Method B probability")
    ax.legend(frameon=False, title="label", loc="upper left")
    finish("ch11_probability_scatter.png")


def multiclass_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    labels = np.repeat(np.arange(5), 140)
    preds = labels.copy()
    for idx, label in enumerate(labels):
        r = rng.random()
        if r < 0.55:
            preds[idx] = label
        elif r < 0.82:
            preds[idx] = int(np.clip(label + rng.choice([-1, 1]), 0, 4))
        else:
            preds[idx] = int(rng.integers(0, 5))
    top1 = np.where(preds == labels, rng.beta(8, 2, labels.size), rng.beta(3, 5, labels.size))
    top1 = np.clip(top1, 0.20, 1.0)
    return labels, preds, top1


def confusion(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    cm = np.zeros((5, 5), dtype=int)
    for y, p in zip(labels, preds):
        cm[y, p] += 1
    return cm


def heatmap(ax, cm: np.ndarray, title: str) -> None:
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=[f"{i} star" for i in range(1, 6)],
        yticklabels=[f"{i} star" for i in range(1, 6)],
        cbar=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted star")
    ax.set_ylabel("Actual star")


def ch12_confusion() -> None:
    labels, preds, _ = multiclass_data()
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    heatmap(ax, confusion(labels, preds), "Confusion Matrix - 5-class Yelp")
    finish("ch12_confusion_matrix.png")


def ch12_top1() -> None:
    labels, preds, top1 = multiclass_data()
    outcome = np.where(labels == preds, "correct", "wrong")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    sns.kdeplot(
        x=top1,
        hue=outcome,
        fill=True,
        common_norm=False,
        alpha=0.5,
        palette={"correct": GREEN, "wrong": "#E55050"},
        clip=(0.2, 1.0),
        ax=ax,
    )
    ax.axvline(0.2, color="black", lw=1, ls=":", alpha=0.45)
    ax.set_title("Top-1 probability by correctness")
    ax.set_xlabel("top-1 predicted probability")
    ax.set_ylabel("Density")
    finish("ch12_top1_probability.png")


def ch12_compare_confusion() -> None:
    labels, bert_preds, _ = multiclass_data()
    rng = np.random.default_rng(15)
    sk_preds = labels.copy()
    for idx, label in enumerate(labels):
        r = rng.random()
        if r < 0.43:
            sk_preds[idx] = label
        elif r < 0.75:
            sk_preds[idx] = int(np.clip(label + rng.choice([-1, 1]), 0, 4))
        else:
            sk_preds[idx] = int(rng.integers(0, 5))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    heatmap(axes[0], confusion(labels, sk_preds), "sklearn TF-IDF + LogReg")
    heatmap(axes[1], confusion(labels, bert_preds), "BERT")
    finish("ch12_confusion_compare.png")


def multilabel_probs(seed: int = 13) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 720
    base = np.array([0.55, 0.48, 0.34, 0.28, 0.22])
    labels = rng.random((n, len(ASPECTS))) < base
    logits = rng.normal(np.where(labels, 2.3, -2.2), 1.15)
    probs = 1 / (1 + np.exp(-logits))
    return labels.astype(int), probs, (probs >= 0.5).astype(int)


def cooccurrence_matrix(y: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    matrix = np.zeros((y.shape[1], y.shape[1]))
    for i in range(y.shape[1]):
        row = y[:, i]
        denom = row.sum()
        if denom == 0:
            continue
        matrix[i] = (row[:, None] * y).sum(axis=0) / denom
    return matrix


def ch13_label_probability_facets() -> None:
    labels, probs, _ = multilabel_probs(13)
    rows = []
    for k, aspect in enumerate(ASPECTS):
        rows.extend(
            {"aspect": aspect, "prob": float(probs[i, k]), "label": int(labels[i, k])}
            for i in range(probs.shape[0])
        )
    df = pd.DataFrame(rows)
    grid = sns.FacetGrid(df, col="aspect", col_wrap=3, height=2.45, aspect=1.35)
    grid.map_dataframe(
        sns.kdeplot,
        x="prob",
        hue="label",
        fill=True,
        common_norm=False,
        alpha=0.46,
        palette={0: BLUE, 1: RED},
        clip=(0, 1),
    )
    for ax in grid.axes.flat:
        ax.axvline(0.5, color="black", lw=0.8, ls="--", alpha=0.62)
        ax.set_xlabel("sigmoid probability")
    grid.add_legend(title="label")
    grid.fig.suptitle("Per-label sigmoid probability distribution", y=1.03)
    grid.fig.subplots_adjust(top=0.86)
    grid.fig.savefig(OUT / "ch13_label_probability_facets.png", dpi=220, bbox_inches="tight")
    plt.close(grid.fig)


def ch13_cooccurrence() -> None:
    labels, _, preds = multilabel_probs(14)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, matrix, title in [
        (axes[0], cooccurrence_matrix(labels), "True co-occurrence P(j | i)"),
        (axes[1], cooccurrence_matrix(preds), "Predicted co-occurrence P(j | i)"),
    ]:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=ASPECTS,
            yticklabels=ASPECTS,
            cbar=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("label j")
        ax.set_ylabel("given label i")
    finish("ch13_cooccurrence.png")


def ch13_f1_compare() -> None:
    x = np.arange(len(ASPECTS))
    sk = np.array([0.72, 0.66, 0.58, 0.53, 0.48])
    bert = np.array([0.78, 0.73, 0.64, 0.60, 0.55])
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    width = 0.38
    ax.bar(x - width / 2, sk, width, label="sklearn (OvR)", color=BLUE, alpha=0.86)
    ax.bar(x + width / 2, bert, width, label="BERT", color=RED, alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(ASPECTS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-label F1")
    ax.set_title("Per-label F1 - sklearn OvR vs BERT")
    ax.legend(frameon=False)
    finish("ch13_f1_compare.png")


def ch14_f1_aux_compare() -> None:
    x = np.arange(len(ASPECTS))
    no_aux = np.array([0.76, 0.70, 0.61, 0.57, 0.52])
    aux = np.array([0.78, 0.73, 0.66, 0.61, 0.56])
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    width = 0.38
    ax.bar(x - width / 2, no_aux, width, label="lambda = 0", color=BLUE, alpha=0.86)
    ax.bar(x + width / 2, aux, width, label="lambda = 1", color=RED, alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(ASPECTS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-label F1")
    ax.set_title("Per-label F1 - auxiliary loss effect")
    ax.legend(frameon=False)
    finish("ch14_aux_f1_compare.png")


def ch14_aux_star_violin() -> None:
    rng = np.random.default_rng(141)
    stars = np.repeat(np.arange(1, 6), 130)
    target = (stars - 1) / 4
    pred = np.clip(target + rng.normal(0, 0.12, stars.size), -0.1, 1.1)
    df = pd.DataFrame(
        {
            "True star": [f"{star}*" for star in stars],
            "Predicted (0-1 scale)": pred,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    sns.violinplot(
        data=df,
        x="True star",
        y="Predicted (0-1 scale)",
        order=[f"{i}*" for i in range(1, 6)],
        inner="quart",
        cut=0,
        color=RED,
        alpha=0.6,
        ax=ax,
    )
    for i, value in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        ax.hlines(value, i - 0.4, i + 0.4, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.set_ylim(-0.15, 1.15)
    ax.set_title("Auxiliary star regression - predicted vs true")
    finish("ch14_aux_star_violin.png")


def ch15_probability_kde() -> None:
    rng = np.random.default_rng(150)
    neg = np.clip(rng.beta(1.4, 8.0, 520), 0, 1)
    pos = np.clip(rng.beta(8.0, 1.5, 520), 0, 1)
    df = pd.DataFrame(
        {
            "prob": np.concatenate([neg, pos]),
            "label": ["negative"] * len(neg) + ["positive"] * len(pos),
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    sns.kdeplot(
        data=df,
        x="prob",
        hue="label",
        fill=True,
        common_norm=False,
        alpha=0.48,
        palette={"negative": BLUE, "positive": RED},
        clip=(0, 1),
        ax=ax,
    )
    ax.axvline(0.5, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_title("NSMC binary classification - probability distribution")
    ax.set_xlabel("P(positive)")
    ax.set_ylabel("Density")
    finish("ch15_probability_kde.png")


def ch15_logit_kde() -> None:
    rng = np.random.default_rng(151)
    neg = rng.normal(-3.0, 1.25, 520)
    pos = rng.normal(3.1, 1.2, 520)
    df = pd.DataFrame(
        {
            "logit": np.concatenate([neg, pos]),
            "label": ["negative"] * len(neg) + ["positive"] * len(pos),
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    sns.kdeplot(
        data=df,
        x="logit",
        hue="label",
        fill=True,
        common_norm=False,
        alpha=0.48,
        palette={"negative": BLUE, "positive": RED},
        ax=ax,
    )
    ax.axvline(0.0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_title("NSMC binary classification - logit distribution")
    ax.set_xlabel("z1 - z0")
    ax.set_ylabel("Density")
    finish("ch15_logit_kde.png")


def ch16_confusion_matrix() -> None:
    labels = ["IT/sci", "economy", "society", "culture", "world", "sports", "politics"]
    cm = np.array(
        [
            [112, 12, 8, 4, 3, 1, 10],
            [10, 104, 13, 3, 4, 0, 16],
            [5, 12, 104, 11, 6, 1, 11],
            [4, 5, 13, 112, 7, 5, 4],
            [2, 8, 8, 7, 115, 1, 9],
            [1, 0, 3, 5, 2, 136, 3],
            [7, 16, 12, 3, 7, 1, 104],
        ],
        dtype=float,
    )
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.9))
    sns.heatmap(
        cm_norm,
        annot=cm.astype(int),
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "row-normalized"},
        ax=ax,
    )
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    ax.set_title("KLUE-YNAT confusion matrix")
    finish("ch16_confusion_matrix.png")


def ch16_top1_probability() -> None:
    rng = np.random.default_rng(160)
    correct = np.clip(rng.beta(8.0, 2.0, 700), 1 / 7, 1)
    wrong = np.clip(rng.beta(3.0, 4.5, 260), 1 / 7, 1)
    df = pd.DataFrame(
        {
            "top1_prob": np.concatenate([correct, wrong]),
            "outcome": ["correct"] * len(correct) + ["wrong"] * len(wrong),
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    sns.kdeplot(
        data=df,
        x="top1_prob",
        hue="outcome",
        fill=True,
        common_norm=False,
        alpha=0.5,
        palette={"correct": GREEN, "wrong": RED},
        clip=(1 / 7, 1),
        ax=ax,
    )
    ax.axvline(1 / 7, color="black", lw=0.9, ls=":", alpha=0.6)
    ax.set_title("Top-1 probability split by correctness")
    ax.set_xlabel("max_k P(y=k)")
    ax.set_ylabel("Density")
    finish("ch16_top1_probability.png")


YNAT_LABELS = ["IT/sci", "economy", "society", "culture", "world", "sports", "politics"]


def ch17_multilabel_probability_facets() -> None:
    rng = np.random.default_rng(170)
    base_rates = np.array([0.25, 0.27, 0.30, 0.24, 0.22, 0.28, 0.26])
    rows = []
    for k, category in enumerate(YNAT_LABELS):
        labels = rng.random(520) < base_rates[k]
        probs = np.where(
            labels,
            rng.beta(7.5 - 0.25 * (k % 3), 2.0 + 0.15 * (k % 2), labels.size),
            rng.beta(1.8 + 0.15 * (k % 2), 7.0 - 0.2 * (k % 3), labels.size),
        )
        rows.extend(
            {"category": category, "prob": float(prob), "label": int(label)}
            for prob, label in zip(probs, labels)
        )
    df = pd.DataFrame(rows)
    grid = sns.FacetGrid(df, col="category", col_wrap=4, height=2.35, aspect=1.25)
    grid.map_dataframe(
        sns.kdeplot,
        x="prob",
        hue="label",
        fill=True,
        common_norm=False,
        alpha=0.46,
        palette={0: BLUE, 1: RED},
        clip=(0, 1),
    )
    for ax in grid.axes.flat:
        ax.axvline(0.5, color="black", lw=0.8, ls="--", alpha=0.65)
        ax.set_xlabel("sigmoid probability")
    grid.add_legend(title="label")
    grid.fig.suptitle("Per-category sigmoid probability distribution", y=1.03)
    grid.fig.subplots_adjust(top=0.86)
    grid.fig.savefig(OUT / "ch17_label_probability_facets.png", dpi=220, bbox_inches="tight")
    plt.close(grid.fig)


def ch17_cooccurrence() -> None:
    rng = np.random.default_rng(171)
    base = np.full((7, 7), 0.24)
    np.fill_diagonal(base, 1.0)
    true = base + rng.normal(0, 0.025, base.shape)
    pred = base + rng.normal(0, 0.055, base.shape)
    np.fill_diagonal(true, 1.0)
    np.fill_diagonal(pred, 1.0)
    true = np.clip(true, 0, 1)
    pred = np.clip(pred, 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    for ax, matrix, title in [
        (axes[0], true, "True co-occurrence P(j | i)"),
        (axes[1], pred, "Predicted co-occurrence P(j | i)"),
    ]:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=YNAT_LABELS,
            yticklabels=YNAT_LABELS,
            cbar=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("category j")
        ax.set_ylabel("given category i")
    finish("ch17_cooccurrence.png")


def ch17_threshold_sweep() -> None:
    thresholds = np.arange(0.1, 0.91, 0.05)
    micro = 0.62 + 0.24 * np.exp(-((thresholds - 0.48) ** 2) / 0.045)
    macro = 0.58 + 0.23 * np.exp(-((thresholds - 0.42) ** 2) / 0.052)
    micro += 0.012 * np.sin(thresholds * 18)
    macro += 0.010 * np.cos(thresholds * 14)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(thresholds, micro, "o-", label="micro F1", color=BLUE)
    ax.plot(thresholds, macro, "s-", label="macro F1", color=RED)
    ax.axvline(0.5, color="black", lw=1.0, ls="--", alpha=0.62)
    ax.text(0.505, min(micro.min(), macro.min()), "default 0.5", va="bottom", fontsize=8, alpha=0.7)
    ax.set_ylim(0.55, 0.9)
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("F1")
    ax.set_title("Threshold sweep - micro vs macro F1")
    ax.legend(frameon=False)
    finish("ch17_threshold_sweep.png")


def ch18_per_label_f1_compare() -> None:
    rng = np.random.default_rng(180)
    no_aux = np.array([0.74, 0.71, 0.68, 0.66, 0.70, 0.79, 0.69])
    aux = np.clip(no_aux + np.array([0.01, 0.025, -0.005, 0.018, 0.012, 0.004, 0.020]), 0, 1)
    aux += rng.normal(0, 0.004, len(aux))
    x = np.arange(len(YNAT_LABELS))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    width = 0.38
    ax.bar(x - width / 2, no_aux, width, label="lambda = 0", color=BLUE, alpha=0.86)
    ax.bar(x + width / 2, aux, width, label="lambda = 0.1", color=RED, alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(YNAT_LABELS, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Per-label F1")
    ax.set_title("Per-category F1 - auxiliary loss effect")
    ax.legend(frameon=False)
    finish("ch18_per_label_f1_compare.png")


def ch18_aux_count_violin() -> None:
    rng = np.random.default_rng(181)
    true_one = np.ones(180)
    true_two = np.full(760, 2.0)
    pred_one = np.clip(rng.normal(1.18, 0.22, len(true_one)), 0, 3)
    pred_two = np.clip(rng.normal(1.88, 0.24, len(true_two)), 0, 3)
    df = pd.DataFrame(
        {
            "True n_active": ["1"] * len(true_one) + ["2"] * len(true_two),
            "Predicted": np.concatenate([pred_one, pred_two]),
        }
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    sns.violinplot(
        data=df,
        x="True n_active",
        y="Predicted",
        order=["1", "2"],
        inner="quart",
        cut=0,
        color=RED,
        alpha=0.65,
        ax=ax,
    )
    for i, target in enumerate([1.0, 2.0]):
        ax.hlines(target, i - 0.4, i + 0.4, color="black", lw=1.0, ls="--", alpha=0.68)
    ax.set_ylim(0.0, 3.0)
    ax.set_title("Auxiliary task - predicted vs true n_active")
    finish("ch18_aux_count_violin.png")


def ch19_token_length_distribution() -> None:
    rng = np.random.default_rng(190)
    en_wp = np.clip(rng.gamma(5.4, 10.5, 900), 8, 210)
    en_wl = np.clip(rng.gamma(4.7, 8.8, 900), 6, 170)
    ko_wp = np.clip(rng.gamma(4.1, 4.4, 900), 5, 72)
    ko_wl = np.clip(rng.gamma(3.3, 3.8, 900), 4, 56)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    sns.kdeplot(en_wp, ax=axes[0], color=BLUE, fill=True, alpha=0.35, label="WordPiece")
    sns.kdeplot(en_wl, ax=axes[0], color=RED, fill=True, alpha=0.35, label="WordLevel")
    axes[0].set_title("English corpus")
    axes[0].set_xlabel("tokens per sentence")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    sns.kdeplot(ko_wp, ax=axes[1], color=BLUE, fill=True, alpha=0.35, label="WordPiece")
    sns.kdeplot(ko_wl, ax=axes[1], color=RED, fill=True, alpha=0.35, label="WordLevel")
    axes[1].set_title("Korean corpus")
    axes[1].set_xlabel("tokens per sentence")
    axes[1].legend(frameon=False)
    finish("ch19_token_length_distribution.png")


def ch19_unk_rate_bar() -> None:
    labels = ["en\nWordPiece", "en\nWordLevel", "ko\nWordPiece", "ko\nWordLevel"]
    rates = np.array([0.03, 3.4, 0.05, 5.8])
    colors = [BLUE, RED, BLUE, RED]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bars = ax.bar(labels, rates, color=colors, alpha=0.86)
    for bar, value in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value:.2f}%", ha="center", fontsize=8)
    ax.set_ylim(0, max(rates) + 1.0)
    ax.set_ylabel("UNK rate (%)")
    ax.set_title("Unknown token rate by tokenizer")
    finish("ch19_unk_rate_bar.png")


def ch19_cross_language_heatmap() -> None:
    matrix = pd.DataFrame(
        [[0.0, 1.6, 68.5, 83.0], [72.0, 91.5, 0.0, 4.5]],
        index=["EN input", "KO input"],
        columns=["en WP", "en WL", "ko WP", "ko WL"],
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "UNK rate (%)"},
        ax=ax,
    )
    ax.set_title("Cross-language application")
    ax.set_xlabel("trained tokenizer")
    ax.set_ylabel("input language")
    finish("ch19_cross_language_heatmap.png")


def ch19_vocab_sweep() -> None:
    vocab = np.array([1000, 4000, 8000, 16000])
    mean_tokens = np.array([72.0, 60.5, 55.2, 52.3])
    unk_rate = np.array([1.8, 0.35, 0.06, 0.02])
    fig, ax1 = plt.subplots(figsize=(7.4, 4.2))
    ax1.plot(vocab, mean_tokens, "o-", color=BLUE, label="mean tokens")
    ax1.set_xscale("log")
    ax1.set_xlabel("vocab size")
    ax1.set_ylabel("mean tokens per sentence", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax1.twinx()
    ax2.plot(vocab, unk_rate, "s--", color=RED, label="UNK rate")
    ax2.set_ylabel("UNK rate (%)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax1.set_title("WordPiece vocabulary size sweep")
    finish("ch19_vocab_sweep.png")


def ch20_mlm_training_loss() -> None:
    steps = np.arange(20, 321, 20)
    rng = np.random.default_rng(200)
    losses = 9.7 - 2.1 * (1 - np.exp(-steps / 120)) + rng.normal(0, 0.08, len(steps))
    baseline = np.log(30522)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(steps, losses, "o-", color=BLUE, label="train MLM loss")
    ax.axhline(baseline, color="black", lw=1.0, ls=":", label="random baseline ln V")
    ax.set_xlabel("training step")
    ax.set_ylabel("MLM loss")
    ax.set_title("Small BERT MLM pretraining loss")
    ax.legend(frameon=False)
    finish("ch20_mlm_training_loss.png")


def ch20_eval_loss_ppl() -> None:
    labels = ["before\nrandom", "after\nMLM"]
    loss_values = np.array([10.28, 7.45])
    ppl_values = np.exp(loss_values)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].bar(labels, loss_values, color=["#999999", BLUE], alpha=0.88)
    axes[0].axhline(np.log(30522), color="black", lw=1.0, ls=":")
    axes[0].set_ylabel("eval_loss")
    axes[0].set_title("MLM eval loss")
    axes[1].bar(labels, ppl_values, color=["#999999", BLUE], alpha=0.88)
    axes[1].axhline(30522, color="black", lw=1.0, ls=":")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("perplexity")
    axes[1].set_title("MLM perplexity")
    finish("ch20_eval_loss_ppl.png")


def ch21_finetune_loss() -> None:
    steps = np.arange(50, 651, 50)
    rng = np.random.default_rng(210)
    losses = 0.68 - 0.24 * (1 - np.exp(-steps / 210)) + rng.normal(0, 0.018, len(steps))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(steps, losses, "o-", color=BLUE, label="train CE loss")
    ax.axhline(np.log(2), color="black", lw=1.0, ls=":", label="random baseline ln 2")
    ax.set_xlabel("training step")
    ax.set_ylabel("Cross-Entropy loss")
    ax.set_title("Yelp fine-tuning loss - small BERT")
    ax.legend(frameon=False)
    finish("ch21_finetune_loss.png")


def ch21_confusion_matrix() -> None:
    cm = np.array([[420, 75], [68, 437]])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.7))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"],
        cbar_kws={"label": "row-normalized recall"},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Small BERT - Yelp confusion matrix")
    finish("ch21_confusion_matrix.png")


def ch21_ch10_compare() -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    ch10 = np.array([0.93, 0.93, 0.93, 0.93, 0.98])
    ch21 = np.array([0.86, 0.85, 0.87, 0.86, 0.93])
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    width = 0.38
    ax.bar(x - width / 2, ch10, width, color=BLUE, label="Ch10 DistilBERT")
    ax.bar(x + width / 2, ch21, width, color=RED, label="Ch21 small BERT")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Yelp binary classification - reference vs small BERT")
    ax.legend(frameon=False, loc="lower right")
    finish("ch21_ch10_compare.png")


def ch22_mlm_training_loss() -> None:
    steps = np.arange(40, 641, 40)
    rng = np.random.default_rng(220)
    baseline = np.log(32000)
    losses = baseline - 3.4 * (1 - np.exp(-steps / 210)) + rng.normal(0, 0.08, len(steps))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(steps, losses, "o-", color=BLUE, label="train MLM loss")
    ax.axhline(baseline, color="black", lw=1.0, ls=":", label="random baseline ln V")
    ax.set_xlabel("training step")
    ax.set_ylabel("MLM loss")
    ax.set_title("Korean small BERT MLM pretraining loss")
    ax.legend(frameon=False)
    finish("ch22_mlm_training_loss.png")


def ch22_eval_loss_ppl() -> None:
    labels = ["before\nrandom", "after\nMLM"]
    loss_values = np.array([10.37, 6.35])
    ppl_values = np.exp(loss_values)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].bar(labels, loss_values, color=["#999999", BLUE], alpha=0.88)
    axes[0].axhline(np.log(32000), color="black", lw=1.0, ls=":")
    axes[0].set_ylabel("eval_loss")
    axes[0].set_title("Korean MLM eval loss")
    axes[1].bar(labels, ppl_values, color=["#999999", BLUE], alpha=0.88)
    axes[1].axhline(32000, color="black", lw=1.0, ls=":")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("perplexity")
    axes[1].set_title("Korean MLM perplexity")
    finish("ch22_eval_loss_ppl.png")


def ch23_finetune_loss() -> None:
    steps = np.arange(50, 651, 50)
    rng = np.random.default_rng(230)
    losses = 0.69 - 0.17 * (1 - np.exp(-steps / 260)) + rng.normal(0, 0.02, len(steps))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(steps, losses, "o-", color=BLUE, label="train CE loss")
    ax.axhline(np.log(2), color="black", lw=1.0, ls=":", label="random baseline ln 2")
    ax.set_xlabel("training step")
    ax.set_ylabel("Cross-Entropy loss")
    ax.set_title("NSMC fine-tuning loss - Korean small BERT")
    ax.legend(frameon=False)
    finish("ch23_finetune_loss.png")


def ch23_confusion_matrix() -> None:
    cm = np.array([[395, 105], [112, 388]])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.7))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"],
        cbar_kws={"label": "row-normalized recall"},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Korean small BERT - NSMC confusion matrix")
    finish("ch23_confusion_matrix.png")


def ch23_ch15_compare() -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    ch15 = np.array([0.86, 0.86, 0.86, 0.86, 0.93])
    ch23 = np.array([0.79, 0.79, 0.78, 0.79, 0.87])
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    width = 0.38
    ax.bar(x - width / 2, ch15, width, color=BLUE, label="Ch15 KLUE-BERT")
    ax.bar(x + width / 2, ch23, width, color=RED, label="Ch23 small BERT")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("NSMC binary classification - reference vs small BERT")
    ax.legend(frameon=False, loc="lower right")
    finish("ch23_ch15_compare.png")


def ch24_loss_vram_trace() -> None:
    steps = np.arange(0, 1501, 150)
    train = np.array([7.45, 5.60, 4.62, 3.92, 3.46, 3.14, 2.92, 2.78, 2.66, 2.58, 2.53])
    eval_loss = np.array([7.58, 5.82, 4.86, 4.12, 3.62, 3.29, 3.04, 2.89, 2.77, 2.69, 2.63])
    vram_steps = np.arange(150, 1501, 150)
    peak = np.array([3.4, 3.6, 3.7, 3.8, 3.8, 3.9, 3.9, 4.0, 4.0, 4.0])
    reserved = peak + 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8))
    ax1.plot(steps, train, "o-", color=BLUE, label="train")
    ax1.plot(steps, eval_loss, "s-", color=RED, label="eval")
    ax1.axhline(np.log(2048), color=INK, linestyle=":", linewidth=1.0, label="ln(2048)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("CLM loss")
    ax1.set_title("Ch24 scratch GPT - TinyStories loss")
    ax1.legend(frameon=False, fontsize=8)

    ax2.plot(vram_steps, peak, "o-", color=GREEN, label="peak")
    ax2.plot(vram_steps, reserved, "s--", color=SLATE, alpha=0.7, label="reserved")
    ax2.set_xlabel("step")
    ax2.set_ylabel("VRAM (GiB)")
    ax2.set_ylim(0, 5)
    ax2.set_title("T4 VRAM trace")
    ax2.legend(frameon=False, fontsize=8)
    finish("ch24_loss_vram_trace.png")


def ch25_loss_vram_trace() -> None:
    steps = np.arange(0, 701, 100)
    train = np.array([3.65, 3.24, 2.98, 2.82, 2.70, 2.61, 2.56, 2.53])
    eval_loss = np.array([3.72, 3.34, 3.10, 2.94, 2.83, 2.76, 2.72, 2.70])
    vram_steps = np.arange(100, 701, 100)
    peak = np.array([10.1, 10.4, 10.6, 10.7, 10.8, 10.8, 10.9])
    reserved = peak + 0.55

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8))
    ax1.plot(steps, train, "o-", color=BLUE, label="train")
    ax1.plot(steps, eval_loss, "s-", color=RED, label="eval")
    ax1.axhline(np.log(50257), color=INK, linestyle=":", linewidth=1.0, label="ln(50257)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("CLM loss")
    ax1.set_title("Ch25 GPT-2 continual pretraining loss")
    ax1.legend(frameon=False, fontsize=8)

    ax2.plot(vram_steps, peak, "o-", color=GREEN, label="peak")
    ax2.plot(vram_steps, reserved, "s--", color=SLATE, alpha=0.7, label="reserved")
    ax2.set_xlabel("step")
    ax2.set_ylabel("VRAM (GiB)")
    ax2.set_ylim(0, 13)
    ax2.set_title("T4 VRAM trace")
    ax2.legend(frameon=False, fontsize=8)
    finish("ch25_loss_vram_trace.png")


def ch25_ch24_loss_compare() -> None:
    labels = ["start", "end"]
    ch24 = np.array([7.62, 2.63])
    ch25 = np.array([3.72, 2.70])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    width = 0.34
    ax.bar(x - width / 2, ch24, width, color=BLUE, label="Ch24 scratch 3M")
    ax.bar(x + width / 2, ch25, width, color=RED, label="Ch25 gpt2 continual")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("eval loss")
    ax.set_title("TinyStories CLM - scratch vs continual pretraining")
    ax.legend(frameon=False)
    for xpos, vals in [(x[0] - width / 2, ch24[0]), (x[1] - width / 2, ch24[1]),
                       (x[0] + width / 2, ch25[0]), (x[1] + width / 2, ch25[1])]:
        ax.text(xpos, vals + 0.08, f"{vals:.2f}", ha="center", fontsize=8, color=INK)
    finish("ch25_ch24_loss_compare.png")


def ch26_loss_vram_trace() -> None:
    steps = np.arange(0, 1501, 150)
    train = np.array([8.12, 6.18, 5.02, 4.18, 3.62, 3.28, 3.04, 2.90, 2.78, 2.69, 2.61])
    eval_loss = np.array([8.24, 6.42, 5.28, 4.46, 3.88, 3.48, 3.22, 3.06, 2.94, 2.84, 2.76])
    vram_steps = np.arange(150, 1501, 150)
    peak = np.array([3.5, 3.7, 3.8, 3.9, 3.9, 4.0, 4.0, 4.1, 4.1, 4.1])
    reserved = peak + 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8))
    ax1.plot(steps, train, "o-", color=BLUE, label="train")
    ax1.plot(steps, eval_loss, "s-", color=RED, label="eval")
    ax1.axhline(np.log(4000), color=INK, linestyle=":", linewidth=1.0, label="ln(4000)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("CLM loss")
    ax1.set_title("Ch26 Korean scratch GPT - TinyStories-Korean loss")
    ax1.legend(frameon=False, fontsize=8)

    ax2.plot(vram_steps, peak, "o-", color=GREEN, label="peak")
    ax2.plot(vram_steps, reserved, "s--", color=SLATE, alpha=0.7, label="reserved")
    ax2.set_xlabel("step")
    ax2.set_ylabel("VRAM (GiB)")
    ax2.set_ylim(0, 5)
    ax2.set_title("T4 VRAM trace")
    ax2.legend(frameon=False, fontsize=8)
    finish("ch26_loss_vram_trace.png")


def ch26_ch24_loss_compare() -> None:
    labels = ["random baseline", "end eval loss"]
    ch24 = np.array([np.log(2048), 2.63])
    ch26 = np.array([np.log(4000), 2.76])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    width = 0.34
    ax.bar(x - width / 2, ch24, width, color=BLUE, label="Ch24 English")
    ax.bar(x + width / 2, ch26, width, color=GREEN, label="Ch26 Korean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("loss")
    ax.set_title("Scratch CLM - English vs Korean TinyStories")
    ax.legend(frameon=False)
    for xpos, vals in [(x[0] - width / 2, ch24[0]), (x[1] - width / 2, ch24[1]),
                       (x[0] + width / 2, ch26[0]), (x[1] + width / 2, ch26[1])]:
        ax.text(xpos, vals + 0.08, f"{vals:.2f}", ha="center", fontsize=8, color=INK)
    finish("ch26_ch24_loss_compare.png")


def main() -> None:
    theme()
    ch01_star_distribution()
    ch02_prediction_distribution()
    ch09_prediction_violin()
    ch09_residual_violin()
    binary_kde("ch10_probability_kde.png", "Method A - Probability Distribution", "prob", 10)
    binary_kde("ch10_logit_kde.png", "Method A - Logit Distribution", "logit", 10)
    binary_kde("ch11_probability_kde.png", "Method B - Probability Distribution", "prob", 11)
    binary_kde("ch11_logit_kde.png", "Method B - Logit Distribution", "logit", 11)
    ch11_scatter()
    ch12_confusion()
    ch12_top1()
    ch12_compare_confusion()
    ch13_label_probability_facets()
    ch13_cooccurrence()
    ch13_f1_compare()
    ch14_f1_aux_compare()
    ch14_aux_star_violin()
    ch15_probability_kde()
    ch15_logit_kde()
    ch16_confusion_matrix()
    ch16_top1_probability()
    ch17_multilabel_probability_facets()
    ch17_cooccurrence()
    ch17_threshold_sweep()
    ch18_per_label_f1_compare()
    ch18_aux_count_violin()
    ch19_token_length_distribution()
    ch19_unk_rate_bar()
    ch19_cross_language_heatmap()
    ch19_vocab_sweep()
    ch20_mlm_training_loss()
    ch20_eval_loss_ppl()
    ch21_finetune_loss()
    ch21_confusion_matrix()
    ch21_ch10_compare()
    ch22_mlm_training_loss()
    ch22_eval_loss_ppl()
    ch23_finetune_loss()
    ch23_confusion_matrix()
    ch23_ch15_compare()
    ch24_loss_vram_trace()
    ch25_loss_vram_trace()
    ch25_ch24_loss_compare()
    ch26_loss_vram_trace()
    ch26_ch24_loss_compare()


if __name__ == "__main__":
    main()
