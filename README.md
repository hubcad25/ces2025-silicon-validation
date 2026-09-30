# CES 2025 silicon validation

Start with [the two-page plan](plan.pdf). Five regions × three ages = **15 groups**.

- `validation.csv`: 20,180 human respondents; 13 columns; original CES IDs, ten readable answers, Age and Region.
- `results_stupid_baseline.csv`: 1,500 example model-result rows; each question always gets its overall CES modal answer.
- `questions.json`: exact English questions, allowed labels, CES option codes and baseline answers.
- `group_weights.csv`: 150 question–group combinations, human counts and synthetic-row weights.
- `scores_stupid_baseline_*.csv`: baseline results at the overall and subgroup levels.
- `manifest.json`: source location and SHA-256 fingerprint.

Weights are **question-specific CES respondent shares, not official survey weights**. This includes the Territories, which have no official CES weights. `Not asked / missing` is excluded from human denominators; don't-know answers remain. Invalid model answers remain in the denominator as their own category.

## Reproduce

Python with pandas and numpy is required. From this directory:

```sh
../opinionbench/.venv/bin/python build.py
../opinionbench/.venv/bin/python score.py results_stupid_baseline.csv
quarto render plan.qmd --to pdf
```

For a new model, save `results_MODEL.csv` with exactly the same columns and 100 rows per group, using unique synthetic IDs and exact option text from `questions.json`. Then:

```sh
../opinionbench/.venv/bin/python score.py results_MODEL.csv
```

The scorer writes overall scores, subgroup scores and comparisons to the baseline. Positive `improvement` means lower TV than the baseline. Results do not claim respondent-level prediction: synthetic IDs have no human counterpart.

No model inference or fine-tuning has been run. The baseline selects its mode from the full evaluation data and is an informed diagnostic comparator. Record any training exposure to the validation questions/respondents before claiming generalization.

Respondent data and generated result CSVs are ignored by Git. The folder is not initialized as a separate Git repository.
