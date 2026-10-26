# Predicting Mortgage Default Early Using Loan Payment History and LSTM/GRU Models

MGSC 673 Individual Project – Alex Zhao

This project tests whether the first 12 months of a mortgage's payment history can predict serious delinquency (90+ days past due) in months 13 to 24. An LSTM that reads the monthly payment history together with the origination data is compared with a logistic regression, a feed-forward network and a simple credit score rule that only use the origination data.

On the test set, the LSTM has a ROC-AUC of 0.903 and catches 58.2% of the defaults in the 5% riskiest loans, compared with about 36% for the origination-only models and 25.7% for the credit score rule.

**Project page:** https://alexkyoya.github.io/mortgage-default-early-warning/

## Files

- `mortgage_default_early_warning.ipynb`: the full project (data preparation, EDA, models, experiments and analysis), with all outputs saved.
- `portfolio/index.html`: the project page, deployed to GitHub Pages.
- `models/`: the trained models and their training histories. The notebook loads these instead of training again.
- `requirements.txt`: the Python packages needed.

## Data

The data is not included because it can't be redistributed. It is free after registration:

1. Go to the Freddie Mac Single-Family Loan-Level Dataset page: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
2. Register and sign in to Clarity Data Intelligence, then open the data download section.
3. From the **Standard Dataset** (the full quarterly files, not the sample files), download:
   - `historical_data_2015Q1.zip`
   - `historical_data_2016Q1.zip`
   - `historical_data_2017Q1.zip`
4. Put the three zip files in `data/raw/` inside the project folder. They don't need to be unzipped.

I used Release 47 (July 2026), with performance data up to March 2026. Freddie Mac updates the dataset over time, so a newer release could give slightly different numbers.

## How to run

The notebook was run with Python 3.14 on a laptop CPU (no GPU needed).

```
python -m venv .venv
.venv\Scripts\activate          (on Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Then open `mortgage_default_early_warning.ipynb`, select the `.venv` kernel, restart the kernel and run all cells.

- The first run reads the raw files (about 26 million monthly rows for 2015 Q1 alone) and saves smaller parquet files in `data/processed/`. This takes a few minutes. Later runs skip this step.
- The trained models are loaded from `models/`, so the whole notebook runs in a few minutes. To train everything again, delete the `models/` folder. Training all the models and experiments takes almost two hours on a laptop CPU.
