"""Visualization helpers for hand strength statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns

# Ensure 3D projection is registered
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  pylint: disable=unused-import


sns.set_theme(style='whitegrid')

_FEATURE_COLUMNS = (
    'trump_count',
    'trump_strength_sum',
    'strongest_trump_strength',
    'max_suit_run',
    'color_aces',
    'total_points',
)


def _prepare_dataframe(
    csv_path: Path,
    min_trumps: Optional[int] = None,
    max_trumps: Optional[int] = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")
    df = df.copy()
    if 'strongest_trump_strength' not in df and 'strongest_trump_rank' in df:
        df['strongest_trump_strength'] = df['strongest_trump_rank']
    if 'weakest_trump_strength' not in df and 'weakest_trump_rank' in df:
        df['weakest_trump_strength'] = df['weakest_trump_rank']
    df['strongest_trump_strength'] = pd.to_numeric(df.get('strongest_trump_strength'), errors='coerce')
    df['weakest_trump_strength'] = pd.to_numeric(df.get('weakest_trump_strength'), errors='coerce')
    if 'trump_strength_sum' in df:
        df['trump_strength_sum'] = pd.to_numeric(df['trump_strength_sum'], errors='coerce')
    df['total_points'] = pd.to_numeric(df['total_points'], errors='coerce')
    df['win_rate'] = pd.to_numeric(df['win_rate'], errors='coerce')
    df['games'] = pd.to_numeric(df['games'], errors='coerce').fillna(0)
    df = df.dropna(subset=['win_rate'])
    if min_trumps is not None:
        df = df[df['trump_count'] >= min_trumps]
    if max_trumps is not None:
        df = df[df['trump_count'] <= max_trumps]
    if df.empty:
        raise ValueError("No rows left after applying trump filters")
    return df


def create_hand_stats_figure(
    csv_path: Path,
    output_path: Optional[Path] = None,
    min_trumps: Optional[int] = None,
    max_trumps: Optional[int] = None,
) -> Path:
    df = _prepare_dataframe(csv_path, min_trumps, max_trumps)
    fig = plt.figure(figsize=(16, 10), dpi=150)

    ax1 = fig.add_subplot(2, 2, 1)
    trump_summary = df.groupby('trump_count')['win_rate'].agg(['mean', 'std', 'count']).reset_index()
    ax1.errorbar(
        trump_summary['trump_count'],
        trump_summary['mean'],
        yerr=trump_summary['std'].fillna(0),
        fmt='o-',
        capsize=4,
        color='steelblue',
        ecolor='lightsteelblue',
    )
    ax1.set_title('Trump Count vs Win Rate')
    ax1.set_xlabel('Number of Trumps')
    ax1.set_ylabel('Win Rate')
    ax1.set_xticks(sorted(df['trump_count'].unique()))

    ax2 = fig.add_subplot(2, 2, 2)
    scatter2 = ax2.scatter(
        df['total_points'],
        df['win_rate'],
        c=df['trump_count'],
        cmap='viridis',
        edgecolor='black',
        alpha=0.7,
    )
    ax2.set_title('Total Points vs Win Rate')
    ax2.set_xlabel('Total Points in Hand')
    ax2.set_ylabel('Win Rate')
    cbar2 = fig.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Trump Count')

    ax3 = fig.add_subplot(2, 2, 3)
    trumps_only = df[df['strongest_trump_strength'].notna()]
    scatter3 = ax3.scatter(
        trumps_only['strongest_trump_strength'],
        trumps_only['win_rate'],
        c=trumps_only['trump_count'],
        cmap='plasma',
        edgecolor='black',
        alpha=0.7,
    )
    ax3.set_title('Strongest Trump vs Win Rate')
    ax3.set_xlabel('Strongest Trump Strength (14 = best)')
    ax3.set_ylabel('Win Rate')
    cbar3 = fig.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Trump Count')

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    normalized_games = df['games'] / df['games'].max() if df['games'].max() else df['games']
    sizes = 40 + 120 * normalized_games
    colors = df['strongest_trump_strength'].fillna(df['strongest_trump_strength'].max() + 1)
    y_axis = df['trump_strength_sum'] if 'trump_strength_sum' in df else df['total_points']
    scatter4 = ax4.scatter(
        df['trump_count'],
        y_axis,
        df['win_rate'],
        c=colors,
        cmap='coolwarm',
        s=sizes,
        alpha=0.8,
        edgecolor='black',
    )
    ax4.set_title('Multi-dimensional View')
    ax4.set_xlabel('Trump Count')
    ax4.set_ylabel('Sum of Trump Strength' if 'trump_strength_sum' in df else 'Total Points')
    ax4.set_zlabel('Win Rate')
    cbar4 = fig.colorbar(scatter4, ax=ax4, pad=0.1)
    cbar4.set_label('Strongest Trump Strength')

    fig.suptitle('Schafkopf Sucherb Hand Strength Overview', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    target = output_path or csv_path.with_suffix('.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches='tight')
    plt.close(fig)
    return target


def create_metric_comparison_figure(
    csv_path: Path,
    output_path: Optional[Path] = None,
    min_trumps: Optional[int] = None,
    max_trumps: Optional[int] = None,
) -> Path:
    df = _prepare_dataframe(csv_path, min_trumps, max_trumps)
    if 'color_aces' not in df:
        df['color_aces'] = 0
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)

    sns.boxplot(ax=axes[0, 0], data=df, x='trump_count', y='win_rate', color='lightsteelblue')
    axes[0, 0].set_title('Win Rate Distribution by Trump Count')
    axes[0, 0].set_xlabel('Trump Count')
    axes[0, 0].set_ylabel('Win Rate')

    strongest = df[df['strongest_trump_strength'].notna()]
    sns.regplot(
        ax=axes[0, 1],
        data=strongest,
        x='strongest_trump_strength',
        y='win_rate',
        scatter_kws={'s': 20, 'alpha': 0.4, 'edgecolor': 'none'},
        line_kws={'color': 'black'},
    )
    axes[0, 1].set_title('Strongest Trump Strength vs Win Rate')
    axes[0, 1].set_xlabel('Strongest Trump Strength (14 = Ober of Acorns)')
    axes[0, 1].set_ylabel('Win Rate')

    y_axis = 'trump_strength_sum' if 'trump_strength_sum' in df else 'total_points'
    sns.regplot(
        ax=axes[1, 0],
        data=df,
        x=y_axis,
        y='win_rate',
        scatter_kws={'s': 18, 'alpha': 0.35, 'edgecolor': 'none'},
        line_kws={'color': 'black'},
    )
    axes[1, 0].set_title(('Trump Strength Sum' if y_axis == 'trump_strength_sum' else 'Total Points') + ' vs Win Rate')
    axes[1, 0].set_xlabel('Sum of Trump Strength' if y_axis == 'trump_strength_sum' else 'Total Points in Hand')
    axes[1, 0].set_ylabel('Win Rate')

    sns.boxplot(ax=axes[1, 1], data=df, x='color_aces', y='win_rate', color='lightblue')
    axes[1, 1].set_title('Win Rate Distribution by Number of Color Aces')
    axes[1, 1].set_xlabel('Color Aces (Excluding Trump Suit)')
    axes[1, 1].set_ylabel('Win Rate')

    fig.suptitle('Win Rate Comparisons Across Key Trump Metrics', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    target = output_path or csv_path.with_name(csv_path.stem + '_comparisons.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches='tight')
    plt.close(fig)
    return target


def _bucket_points(series: pd.Series, step: int = 5) -> pd.Series:
    bucketed = (series.fillna(0) // step) * step
    return bucketed.astype(int)


def create_heatmap_figure(
    csv_path: Path,
    output_path: Optional[Path] = None,
    min_trumps: Optional[int] = None,
    max_trumps: Optional[int] = None,
) -> Path:
    df = _prepare_dataframe(csv_path, min_trumps, max_trumps)
    if 'color_aces' not in df:
        df['color_aces'] = 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    trump_strength_heat = (
        df.pivot_table(
            index='strongest_trump_strength',
            columns='trump_count',
            values='win_rate',
            aggfunc='mean',
        )
        .sort_index(ascending=False)
        .sort_index(axis=1)
    )
    sns.heatmap(
        trump_strength_heat,
        ax=axes[0],
        cmap='YlOrRd',
        cbar_kws={'label': 'Win Rate'},
        annot=False,
    )
    axes[0].set_title('Win Rate by Trump Count & Strongest Trump')
    axes[0].set_xlabel('Trump Count')
    axes[0].set_ylabel('Strongest Trump Strength')

    df['total_points_bucket'] = _bucket_points(df['total_points'], step=5)
    points_aces_heat = (
        df.pivot_table(
            index='total_points_bucket',
            columns='color_aces',
            values='win_rate',
            aggfunc='mean',
        )
        .sort_index(ascending=False)
        .sort_index(axis=1)
    )
    sns.heatmap(
        points_aces_heat,
        ax=axes[1],
        cmap='BuGn',
        cbar_kws={'label': 'Win Rate'},
        annot=False,
    )
    axes[1].set_title('Win Rate by Total Points & Color Aces')
    axes[1].set_xlabel('Color Aces (non-trump)')
    axes[1].set_ylabel('Total Points (bucketed)')

    fig.suptitle('Heatmaps Highlighting Signature Correlations', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    target = output_path or csv_path.with_name(csv_path.stem + '_heatmaps.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches='tight')
    plt.close(fig)
    return target


def _simple_kmeans(
    data: np.ndarray,
    clusters: int,
    rng_seed: int = 0,
    max_iter: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    if len(data) == 0:
        return np.array([], dtype=int), np.empty((0, data.shape[1] if data.ndim > 1 else 0))
    clusters = max(1, min(clusters, len(data)))
    if len(data) == clusters:
        return np.arange(len(data), dtype=int), data.copy()

    rng = np.random.default_rng(rng_seed)
    centers = data[rng.choice(len(data), size=clusters, replace=False)]
    labels = np.zeros(len(data), dtype=int)
    for _ in range(max_iter):
        distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        new_centers = []
        for idx in range(clusters):
            members = data[labels == idx]
            if len(members) == 0:
                new_centers.append(data[rng.integers(len(data))])
            else:
                new_centers.append(members.mean(axis=0))
        new_centers = np.vstack(new_centers)
        if np.allclose(new_centers, centers, atol=1e-4):
            break
        centers = new_centers
    return labels, centers


def _cluster_signatures(frame: pd.DataFrame, rng_seed: int, max_clusters: int = 4) -> pd.DataFrame:
    clustered, _ = _fit_cluster_model(frame, rng_seed, max_clusters)
    return clustered


def _fit_cluster_model(
    frame: pd.DataFrame,
    rng_seed: int,
    max_clusters: int = 4,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    if frame.empty:
        return frame.copy(), {}
    df = frame.copy()
    if 'color_aces' not in df:
        df['color_aces'] = 0
    if 'trump_strength_sum' not in df:
        df['trump_strength_sum'] = df['total_points']

    feature_cols = [col for col in _FEATURE_COLUMNS if col in df.columns]
    if len(feature_cols) < 2:
        raise ValueError("Not enough feature columns available for clustering")

    values = df[feature_cols].fillna(0).to_numpy(dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1.0
    normalized = (values - means) / stds
    labels, centers = _simple_kmeans(normalized, clusters=min(max_clusters, len(values)), rng_seed=rng_seed)
    df['cluster'] = labels
    model: Dict[str, object] = {
        'feature_cols': feature_cols,
        'means': means,
        'stds': stds,
        'centers': centers,
    }
    return df, model


def _assign_cluster_model(frame: pd.DataFrame, model: Dict[str, object]) -> pd.DataFrame:
    if frame.empty or not model:
        return frame.copy()
    df = frame.copy()
    feature_cols = model['feature_cols']
    values = df[feature_cols].fillna(0).to_numpy(dtype=float)
    means = model['means']
    stds = np.where(model['stds'] == 0, 1.0, model['stds'])
    normalized = (values - means) / stds
    centers = model['centers']
    if centers.size == 0:
        df['cluster'] = 0
        return df
    distances = ((normalized[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    df['cluster'] = distances.argmin(axis=1)
    return df


def _cluster_descriptions(frame: pd.DataFrame) -> Dict[int, str]:
    if frame.empty or 'cluster' not in frame:
        return {}
    stats = frame.groupby('cluster').agg(
        trump_count=('trump_count', 'mean'),
        color_aces=('color_aces', 'mean'),
        strongest=('strongest_trump_strength', 'mean'),
        max_suit=('max_suit_run', 'mean'),
    )
    desc: Dict[int, str] = {}
    remaining = set(stats.index)

    if remaining:
        idx = stats['trump_count'].idxmax()
        desc[idx] = 'heavy-trump artillery'
        remaining.discard(idx)
    if remaining:
        subset = stats.loc[list(remaining)]
        idx = subset['color_aces'].idxmax()
        desc[idx] = 'point hoarders'
        remaining.discard(idx)
    if remaining:
        subset = stats.loc[list(remaining)]
        idx = subset['max_suit'].idxmax()
        desc[idx] = 'lean trumps, long suit'
        remaining.discard(idx)
    for idx in remaining:
        desc[idx] = 'middling mix'
    return desc


def create_cluster_figure(
    csv_path: Path,
    output_path: Optional[Path] = None,
    min_trumps: Optional[int] = None,
    max_trumps: Optional[int] = None,
) -> Path:
    df = _prepare_dataframe(csv_path, min_trumps, max_trumps)

    winners = df[df['win_rate'] >= 0.5]
    losers = df[df['win_rate'] < 0.5]
    winners_clustered = _cluster_signatures(winners, rng_seed=7)
    losers_clustered = _cluster_signatures(losers, rng_seed=13)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    def _plot_panel(ax, data: pd.DataFrame, title: str) -> None:
        if data.empty:
            ax.text(0.5, 0.5, 'Not enough samples', ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            return
        x = data['trump_strength_sum']
        y = data['strongest_trump_strength']
        sizes = 30 + 90 * data['win_rate']
        scatter = ax.scatter(
            x,
            y,
            c=data['cluster'],
            cmap='tab10',
            s=sizes,
            alpha=0.75,
            edgecolor='black',
            linewidth=0.2,
        )
        ax.set_title(title)
        ax.set_xlabel('Sum of Trump Strength')
        ax.set_ylabel('Strongest Trump Strength')
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster ID')

    _plot_panel(axes[0], winners_clustered, 'Winning Signatures (win_rate ≥ 0.5)')
    _plot_panel(axes[1], losers_clustered, 'Losing Signatures (win_rate < 0.5)')

    fig.suptitle('Clustered Signature Landscapes', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    target = output_path or csv_path.with_name(csv_path.stem + '_clusters.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches='tight')
    plt.close(fig)
    return target


def _scatter_trump_vs_win(
    ax: plt.Axes,
    data: pd.DataFrame,
    cmap,
    title: str,
) -> None:
    if data.empty:
        ax.text(0.5, 0.5, 'Not enough samples', ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        return
    games = data['games']
    if games.max() > 0:
        sizes = 40 + 120 * (games / games.max())
    else:
        sizes = np.full(len(data), 80)
    ax.scatter(
        data['trump_count'],
        data['win_rate'],
        c=data['cluster'],
        cmap=cmap,
        s=sizes,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.2,
    )
    ax.set_xlabel('Trump Count')
    ax.set_ylabel('Win Rate')
    ax.set_ylim(0.4, 1.01)
    ax.set_title(title)


def create_cluster_trump_win_figure(
    csv_path: Path,
    output_path: Optional[Path] = None,
    min_trumps: Optional[int] = None,
    max_trumps: Optional[int] = None,
) -> Path:
    df = _prepare_dataframe(csv_path, min_trumps, max_trumps)
    winners = df[df['win_rate'] >= 0.5]
    clustered, _ = _fit_cluster_model(winners, rng_seed=7)
    if clustered.empty:
        raise ValueError("No winning signatures available for clustering")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    unique_clusters = sorted(clustered['cluster'].unique())
    cmap = plt.cm.get_cmap('tab10', len(unique_clusters))
    _scatter_trump_vs_win(ax, clustered, cmap, 'Winning Signature Clusters: Trump Count vs Win Rate')

    descriptions = _cluster_descriptions(clustered)
    handles = []
    for idx in unique_clusters:
        label = f"Cluster {idx}: {descriptions.get(idx, 'signature family')}"
        handles.append(Patch(facecolor=cmap(idx), edgecolor='black', label=label))
    ax.legend(handles=handles, title='Cluster ID', loc='lower right', frameon=True)

    fig.tight_layout()
    target = output_path or csv_path.with_name(csv_path.stem + '_cluster_trump_win.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches='tight')
    plt.close(fig)
    return target


def create_cluster_trump_win_all_figure(
    csv_path: Path,
    output_path: Optional[Path] = None,
    min_trumps: Optional[int] = None,
    max_trumps: Optional[int] = None,
) -> Path:
    df = _prepare_dataframe(csv_path, min_trumps, max_trumps)
    winners = df[df['win_rate'] >= 0.5]
    losers = df[df['win_rate'] < 0.5]
    winners_clustered, model = _fit_cluster_model(winners, rng_seed=7)
    if not model:
        raise ValueError("No data available to build cluster model")
    losers_clustered = _assign_cluster_model(losers, model)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150, sharey=True)
    unique_clusters = sorted(winners_clustered['cluster'].unique()) if not winners_clustered.empty else [0]
    cmap = plt.cm.get_cmap('tab10', len(unique_clusters))

    _scatter_trump_vs_win(axes[0], winners_clustered, cmap, 'Winning Signatures (win_rate ≥ 0.5)')
    _scatter_trump_vs_win(axes[1], losers_clustered, cmap, 'Losing Signatures (win_rate < 0.5)')

    descriptions = _cluster_descriptions(winners_clustered)
    handles = []
    for idx in unique_clusters:
        label = f"Cluster {idx}: {descriptions.get(idx, 'signature family')}"
        handles.append(Patch(facecolor=cmap(idx), edgecolor='black', label=label))
    fig.legend(handles=handles, title='Cluster ID', loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2)

    fig.suptitle('Trump Count vs Win Rate — Winning vs Losing Clusters', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    target = output_path or csv_path.with_name(csv_path.stem + '_cluster_trump_win_all.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches='tight')
    plt.close(fig)
    return target
