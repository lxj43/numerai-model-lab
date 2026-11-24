"""Create seeded, fictional weekly observations for exercising the full pipeline."""
import json

import numpy as np
import pandas as pd

from paths import ROOT, SOURCE


def make_frame(rng, start, periods, tickers, features):
    weeks = pd.date_range(start, periods=periods, freq='W-FRI')
    values = rng.integers(0, 5, (len(weeks) * len(tickers), len(features)))
    frame = pd.DataFrame(values, columns=features)
    frame.insert(0, 'numerai_ticker', np.tile(tickers, len(weeks)))
    frame.insert(0, 'date', np.repeat(weeks.strftime('%Y-%m-%d'), len(tickers)))
    signal = (values[:, 0] - values[:, 1]) * .12 + rng.normal(0, .45, len(values))
    frame['target_jupiter_60'] = pd.Series(signal).groupby(frame['date']).rank(pct=True)
    return frame


def main():
    if ROOT == SOURCE:
        raise SystemExit('Set NUMERAI_WORK_DIR to a separate directory for synthetic data.')
    data = ROOT / 'data'
    data.mkdir(parents=True, exist_ok=True)
    marker = data / 'synthetic.marker'
    if not marker.exists() and any(data.iterdir()):
        raise SystemExit('Use an empty workspace for synthetic data.')
    marker.touch()
    rng = np.random.default_rng(42)
    tickers = [f'DEMO{i:04}' for i in range(1, 81)]
    features = [f'feature_demo_{i:03}' for i in range(100)]
    (data / 'features.json').write_text(json.dumps({'feature_sets': {'small': features}}))
    for filename, start, weeks in (
        ('train_sample', '2017-01-06', 48),
        ('validation_sample', '2019-01-04', 14),
        ('agent_holdout', '2020-04-03', 12),
        ('live_sample', '2020-07-03', 1),
    ):
        frame = make_frame(rng, start, weeks, tickers, features)
        if filename == 'live_sample':
            frame = frame.drop(columns='target_jupiter_60')
        frame.to_parquet(data / (filename + '.parquet'), index=False)


if __name__ == '__main__':
    main()
