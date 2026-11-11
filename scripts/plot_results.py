"""Render figures from the published aggregate results. No network access."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pjm-matplotlib')
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backtest', type=Path, help='Optional hourly backtest to regenerate aggregates')
    args = parser.parse_args()
    results, figures = ROOT / 'results', ROOT / 'figures'
    figures.mkdir(exist_ok=True)
    if args.backtest:
        data = pd.read_csv(args.backtest)
        data['month'] = pd.to_datetime(data.opr_date).dt.month
        for field in ('regression', 'prediction'):
            data[f'{field}_absolute_error'] = (data[field] - data.actual).abs()
            data[f'{field}_squared_error'] = (data[field] - data.actual) ** 2
        data.groupby('month').agg(hours=('actual', 'size'), regression_MAE=('regression_absolute_error', 'mean'),
                                 hybrid_MAE=('prediction_absolute_error', 'mean'),
                                 regression_MSE=('regression_squared_error', 'mean'), hybrid_MSE=('prediction_squared_error', 'mean')).to_csv(results / 'monthly_2025.csv')
        data.groupby('hour_beginning').agg(hours=('actual', 'size'), regression_MAE=('regression_absolute_error', 'mean'),
                                         hybrid_MAE=('prediction_absolute_error', 'mean')).to_csv(results / 'hourly_2025.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})
    monthly = pd.read_csv(results / 'monthly_2025.csv')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly.month, monthly.regression_MAE, marker='o', color='#8996a6', label='Regression backbone')
    ax.plot(monthly.month, monthly.hybrid_MAE, marker='o', color='#075985', label='Final hybrid')
    ax.set(xticks=range(1, 13), xticklabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ylabel='MAE ($/MWh)', title='2025 monthly forecast error')
    ax.legend(frameon=False); ax.grid(axis='y', alpha=.15)
    fig.text(.12, .015, 'Monthly expanding fits · 8,735 scored hours · all prediction routes', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(figures / 'monthly_error.png', dpi=180); plt.close(fig)
    hourly = pd.read_csv(results / 'hourly_2025.csv')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hourly.hour_beginning + 1, hourly.regression_MAE, marker='o', color='#8996a6', label='Regression backbone')
    ax.plot(hourly.hour_beginning + 1, hourly.hybrid_MAE, marker='o', color='#075985', label='Final hybrid')
    ax.set(xticks=[1,4,8,12,16,20,24], xlabel='Hour ending (Eastern prevailing time)', ylabel='MAE ($/MWh)', title='2025 forecast error by delivery hour')
    ax.legend(frameon=False); ax.grid(axis='y', alpha=.15)
    fig.text(.12,.015,'HE1 and HE24 use the geographic regression route when same-day ramp neighbors are unavailable.',fontsize=9,color='#475569')
    fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(figures / 'hourly_error.png', dpi=180); plt.close(fig)


if __name__ == '__main__': main()
