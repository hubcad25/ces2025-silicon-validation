"""Score a 1,500-row result CSV against the readable CES validation data."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from build import ROOT, QUESTIONS, GROUPS, COLUMNS, MISSING


def calculate(validation, results, options):
    for name, frame in [('validation', validation), ('results', results)]:
        if list(frame.columns) != COLUMNS:
            raise ValueError(f'{name}: expected the exact 13-column schema')
        if frame.cps25_ResponseId.isna().any() or not frame.cps25_ResponseId.is_unique:
            raise ValueError(f'{name}: missing or duplicate IDs')
        if set(zip(frame.Region, frame.Age)) != set(GROUPS):
            raise ValueError(f'{name}: incorrect groups')
    sizes = results.groupby(['Region', 'Age']).size()
    if len(results) != 1500 or not sizes.eq(100).all():
        raise ValueError('Results must contain exactly 100 rows in each of the 15 groups')
    weights = []
    cell_scores = []
    overall_scores = []
    for question in QUESTIONS:
        categories = list(options[question]) + ['Invalid output']
        valid_all = validation[validation[question].isin(options[question])]
        bad = validation[~validation[question].isin([*options[question], MISSING])]
        if len(bad):
            raise ValueError(f'{question}: unexpected human answer text')
        population_human = np.zeros(len(categories))
        population_model = np.zeros(len(categories))
        weighted_cell_tv = 0.0
        equal_cell_tv = 0.0
        for region, age in GROUPS:
            human = valid_all[(valid_all.Region == region) & (valid_all.Age == age)]
            synthetic = results[(results.Region == region) & (results.Age == age)][question]
            if human.empty:
                raise ValueError(f'{question}: no human answers in {region}, {age}')
            # Match each item's observed CES composition, including random-module assignment.
            share = len(human) / len(valid_all)
            p = human[question].value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float) / len(human)
            q = synthetic.where(synthetic.isin(options[question]), 'Invalid output').value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float) / 100
            tv = float(np.abs(p - q).sum() / 2)
            weights.append({'question': question, 'Region': region, 'Age': age, 'human_n': len(human), 'human_total_n': len(valid_all), 'group_share': share, 'weight_per_synthetic_row': share / 100})
            cell_scores.append({'question': question, 'Region': region, 'Age': age, 'human_n': len(human), 'synthetic_n': 100, 'invalid_n': int((~synthetic.isin(options[question])).sum()), 'TV': tv})
            population_human += share * p
            population_model += share * q
            weighted_cell_tv += share * tv
            equal_cell_tv += tv / 15
        overall_scores.append({'question': question, 'human_n': len(valid_all), 'TV_overall': float(np.abs(population_human - population_model).sum() / 2), 'TV_cells_weighted_mean': weighted_cell_tv, 'TV_cells_equal_mean': equal_cell_tv})
    return pd.DataFrame(cell_scores), pd.DataFrame(overall_scores), pd.DataFrame(weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    path = args.results.resolve()
    validation = pd.read_csv(ROOT / 'validation.csv', dtype=str, keep_default_na=False)
    results = pd.read_csv(path, dtype=str, keep_default_na=False)
    options = {item['variable']: list(item['options'].values()) for item in json.loads((ROOT / 'questions.json').read_text())}
    cells, overall, weights = calculate(validation, results, options)
    stem = path.stem.removeprefix('results_')
    cells.to_csv(ROOT / f'scores_{stem}_by_group.csv', index=False)
    overall.to_csv(ROOT / f'scores_{stem}_overall.csv', index=False)
    weights.to_csv(ROOT / 'group_weights.csv', index=False)
    baseline_path = ROOT / 'results_stupid_baseline.csv'
    reference_cells, reference, _ = calculate(validation, pd.read_csv(baseline_path, dtype=str, keep_default_na=False), options)
    comparison = overall[['question', 'TV_overall']].merge(reference[['question', 'TV_overall']], on='question', suffixes=('_model', '_baseline'))
    comparison['improvement'] = comparison.TV_overall_baseline - comparison.TV_overall_model
    comparison.to_csv(ROOT / f'scores_{stem}_comparison.csv', index=False)
    cell_comparison = cells.merge(reference_cells[['question', 'Region', 'Age', 'TV']], on=['question', 'Region', 'Age'], suffixes=('_model', '_baseline'))
    cell_comparison['improvement'] = cell_comparison.TV_baseline - cell_comparison.TV_model
    cell_comparison.to_csv(ROOT / f'scores_{stem}_group_comparison.csv', index=False)
    print(overall[['question', 'TV_overall']].to_string(index=False))
    print(f'Mean overall TV: {overall.TV_overall.mean():.6f}')
    print(f'Mean improvement over modal baseline: {comparison.improvement.mean():.6f}')
    print(f'Group-question cells improved: {(cell_comparison.improvement > 1e-12).sum()}/150')


if __name__ == '__main__':
    main()
