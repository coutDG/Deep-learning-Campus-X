from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer


def main():
    rng = np.random.default_rng(42)
    positive = rng.lognormal(mean=1, sigma=0.9, size=4000)
    signed = rng.lognormal(mean=1, sigma=0.9, size=4000) - 4
    signed[0] = 0

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for row, (values, method, label, color) in enumerate([
        (positive, 'box-cox', 'Box–Cox: positive data', '#2563eb'),
        (signed, 'yeo-johnson', 'Yeo–Johnson: negative, zero and positive data', '#0d9488'),
    ]):
        transformer = PowerTransformer(method=method, standardize=True)
        transformed = transformer.fit_transform(values.reshape(-1, 1)).ravel()
        for col, (data, title) in enumerate([
            (values, 'Before'), (transformed, 'After'),
        ]):
            ax = axes[row, col]
            ax.hist(data, bins=65, density=True, color=color, alpha=0.85,
                    edgecolor='white', linewidth=0.3)
            ax.set_title(f'{title} | skewness = {skew(data):.2f}', loc='left')
            ax.set_xlabel('Original value' if col == 0 else 'Transformed value (standardized)')
            ax.set_ylabel('Density')
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=0.15)
        axes[row, 0].text(0, 1.20, label, transform=axes[row, 0].transAxes,
                          fontsize=12, weight='bold')
        axes[row, 1].text(0.98, 0.94, f'Fitted lambda = {transformer.lambdas_[0]:.2f}',
                          transform=axes[row, 1].transAxes, ha='right', va='top')

    fig.suptitle('Power transformations reduce skewness', fontsize=19, weight='bold', y=0.99)
    fig.text(0.5, 0.015, 'Synthetic illustration • Each row uses its own dataset • Normality is not guaranteed',
             ha='center', color='#475569', fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94], h_pad=4)
    output = Path(__file__).parent / 'output' / 'power_transform_before_after.png'
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches='tight')
    print(output)


if __name__ == '__main__':
    main()
