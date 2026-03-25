# PromptFE

The source code for the paper *PromptFE: Automated Feature Engineering by Prompting*.

PromptFE uses large language models to automatically suggest and evaluate engineered features for tabular machine learning datasets. It iteratively prompts an LLM to generate feature transformations, evaluates them using cross-validation, and selects the best-performing features.

## Requirements

- Python 3.8+
- OpenAI API key

Install dependencies:

```bash
pip install -r requirements.txt
```


## Setup

Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. **To create and evaluate features**, run:

```bash
python run.py
```

This prompts GPT to suggest feature transformations, evaluates them on the specified model and dataset, and saves results to the `results/` directory.

2. **To read and analyze results**, run:

```bash
python read.py
```

This loads saved results from `results/` and writes a performance summary to `log_result.txt`.

## Project Structure

```
PromptFE/
├── data/                       # Dataset files
│   ├── *.csv                   # Raw data
│   ├── *.json                  # Dataset metadata (task type, feature info)
│   └── *.txt                   # Feature descriptions
├── dataset_descriptions/       # Extended dataset descriptions
├── params/                     # Model hyperparameter configurations
│   └── {MODEL}-{DATASET}.txt
├── utils/
│   └── tools.py                # Logging and timing utilities
├── run.py                      # Main script: feature generation and evaluation
├── read.py                     # Results analysis script
├── env.py                      # Model training and evaluation environment
├── feat_tree.py                # Feature tree construction and RPN parsing
├── dataset.py                  # Dataset loading and preprocessing
├── search_space.py             # Feature operation definitions
├── metrics.py                  # Custom evaluation metrics
└── requirements.txt
```

## Example Datasets

| Dataset | Task |
|---|---|
| AIDS | Classification |
| Airfoil | Regression |
| Bikeshare | Regression |
| Credit Default | Classification |
| German Credit | Classification |
| Housing | Regression |
| Wine Quality (Red) | Regression |

## Supported Models

**Classification:** `LogisticRegression`, `RandomForestClassifier`, `XGBClassifier`, `LGBMClassifier`

**Regression:** `Lasso`, `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor`

## Feature Operations

PromptFE supports the following feature transformation types:

- **Unary:** `log`, `sqrt`, `reciprocal`, `min-max`
- **Binary:** `+`, `-`, `*`, `/`, `mod`

Features are represented as trees using canonical Reverse Polish Notation (cRPN) and evaluated iteratively to find the best-performing subset.

## Citation

If you use this code, please cite the paper:

```
PromptFE: Automated Feature Engineering by Prompting
```

A portion of the code is adapted from [DIFER](https://github.com/PasaLab/DIFER?utm_source=catalyzex.com).
