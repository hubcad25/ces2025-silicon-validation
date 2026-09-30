"""Build the readable CES benchmark and deterministic modal baseline."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'opinionbench-research/data/raw/CES2025/2025 Canadian Election Study v1.dta'
ITEMS = [
    ('cps25_spend_afford_h', 'How much should the federal government spend on affordable housing?'),
    ('cps25_spend_nation_c', 'How much should the federal government spend on a national childcare system?'),
    ('cps25_spend_defence', 'How much should the federal government spend on defence?'),
    ('cps25_spend_rec_indi', 'How much should the federal government spend on reconciliation with Indigenous Peoples?'),
    ('cps25_pos_life', 'Individuals who are terminally ill should be allowed to end their lives with the assistance of a doctor.'),
    ('cps25_pos_energy', 'The federal government should do more to help build oil pipelines.'),
    ('cps25_pos_jobs', 'When there is a conflict between protecting the environment and creating jobs, jobs should come first.'),
    ('cps25_imm', 'Do you think Canada should admit:'),
    ('cps25_demsat', 'On the whole, how satisfied are you with the way democracy works in Canada?'),
    ('cps25_own_fin_retro', 'Over the past year, has your financial situation:'),
]
QUESTIONS = [key for key, _ in ITEMS]
REGIONS = ['BC', 'AB+SK+MB+Terr', 'ON', 'QC', 'ATL']
AGES = ['18-34', '35-54', '55+']
GROUPS = [(region, age) for region in REGIONS for age in AGES]
COLUMNS = ['cps25_ResponseId', *QUESTIONS, 'Age', 'Region']
MISSING = 'Not asked / missing'


def main():
    raw = pd.read_stata(SOURCE, convert_categoricals=False)
    labels = pd.io.stata.StataReader(SOURCE).value_labels()
    out = pd.DataFrame({'cps25_ResponseId': raw['cps25_ResponseId']})
    metadata = []
    modes = {}
    for key, text in ITEMS:
        options = {int(code): re.sub(r'^\d+\.\s*', '', label).replace('Don’t', "Don't").replace('know/ Prefer', 'know/Prefer')
                   for code, label in labels[key].items()}
        unknown = set(raw[key].dropna().unique()) - set(options)
        if unknown:
            raise ValueError(f'{key}: unknown codes {unknown}')
        out[key] = raw[key].map(options).fillna(MISSING)
        counts = raw[key].value_counts()
        # Ties go to the smallest original CES code.
        modal_code = min(int(k) for k in counts[counts == counts.max()].index)
        modes[key] = options[modal_code]
        metadata.append({'variable': key, 'question': text, 'options': options, 'baseline_answer': modes[key]})
    out['Age'] = pd.cut(raw['cps25_age_in_years'], [18, 35, 55, np.inf], right=False, labels=AGES).astype('string')
    regions = {2: 'BC', 1: REGIONS[1], 12: REGIONS[1], 3: REGIONS[1], 6: REGIONS[1], 8: REGIONS[1], 13: REGIONS[1], 9: 'ON', 11: 'QC', 4: 'ATL', 5: 'ATL', 7: 'ATL', 10: 'ATL'}
    out['Region'] = raw.cps25_province.map(regions)
    if out[COLUMNS].isna().any().any() or not out.cps25_ResponseId.is_unique:
        raise ValueError('Missing grouping/identity fields or duplicate CES IDs')
    out = out[COLUMNS]
    assert out.shape == (20180, 13)
    out.to_csv(ROOT / 'validation.csv', index=False)
    baseline = []
    for group_number, (region, age) in enumerate(GROUPS, 1):
        for draw in range(1, 101):
            baseline.append({'cps25_ResponseId': f'mode_g{group_number:02d}_{draw:03d}', **modes, 'Age': age, 'Region': region})
    result = pd.DataFrame(baseline, columns=COLUMNS)
    assert result.shape == (1500, 13)
    result.to_csv(ROOT / 'results_stupid_baseline.csv', index=False)
    (ROOT / 'questions.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    (ROOT / 'manifest.json').write_text(json.dumps({
        'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'validation_rows': len(out), 'result_rows': len(result), 'columns': COLUMNS,
        'missing_value': MISSING, 'baseline': 'Unweighted overall CES modal answer per question; ties use smallest original code.'
    }, indent=2) + '\n')
    print(f'Created {len(out):,} validation rows and {len(result):,} baseline rows, each with 13 columns.')


if __name__ == '__main__':
    main()
