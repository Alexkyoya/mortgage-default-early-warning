# Predicting Mortgage Default Early Using Loan Payment History and LSTM/GRU Models

MGSC 673 Individual Project – Alex Zhao

This project tests whether the first 12 months of a mortgage's payment history can predict serious delinquency (90+ days past due) in months 13 to 24. An LSTM that reads the monthly payment history together with the origination data is compared with a simple credit score rule, models that only use the origination data, and simpler models that get a summary of the payment history.

## Main results (test set)

- Adding the payment history raises the share of defaults caught in the 5% riskiest loans from about 36% (origination data only) to 57-58%. Picking the lowest credit scores catches 25.7%.
- Part of this comes from an easy case: loans already 30-89 days late at month 12 are only 0.5% of the loans but 26% of the defaults. For loans that are still current at month 12, the LSTM catches 44.7% of the defaults, compared with 35.0% for a logistic regression with origination data only.
- Most of the gain comes from the payment history itself, not from the LSTM. A logistic regression with three simple history summaries (months late, worst status, status at month 12) catches 57.0% of the defaults, close to the LSTM's 58.2%, and 43.8% vs. 44.7% on current loans. The LSTM is still better at ranking the very riskiest current loans (PR-AUC 0.065 vs. 0.035), but a lender would probably prefer the simpler model, which is easier to explain and validate.
- The result holds out of time: trained on 2015-2016 loans and tested on 2017 loans, the LSTM catches 54.2% of the defaults vs. 22.1% for the origination-only logistic regression.
- A sensitivity test found that LTV and CLTV are almost identical, so the model's response to LTV alone is not sensible; only their combined effect is.

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
