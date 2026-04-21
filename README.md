# Bootstrapping for A/B Testing

## Introduction

In digital advertising and marketing, decisions about which campaign, message, or creative performs better are rarely made by intuition alone. A/B testing — the practice of exposing different groups of users to different versions of an ad or experience and measuring the outcome — has become the standard method for turning data into decisions. Done well, it replaces guesswork with evidence and gives marketing teams a principled basis for recommending one approach over another.

The most common statistical tool for evaluating A/B test results is the two-sample t-test. It is fast, familiar, and available in virtually every statistical software package. For many applications it works well. But advertising and marketing data have characteristics that frequently push the t-test beyond the boundaries of what it was designed to handle. Conversion rates are often very low — fractions of a percent — making the underlying distribution heavily skewed rather than the bell-shaped normal curve the t-test assumes. Test groups are routinely unequal in size, sometimes dramatically so, with a control group drawn from hundreds of thousands of records and a treatment group drawn from far fewer. Individual users may appear across multiple touchpoints, introducing correlation between observations that the t-test treats as independent. Under these conditions, the t-test can produce p-values that are misleading — either overstating confidence in a result that isn't real, or failing to detect a difference that genuinely exists.

Bootstrapping offers a different approach. Rather than fitting the data to a theoretical distribution and computing a p-value from a formula, bootstrapping builds its own sampling distribution directly from the data at hand. It resamples each group thousands of times, computes the difference in conversion rates on each resample, and assembles those differences into an empirical distribution that reflects what the data actually look like — skewed, unbalanced, or otherwise. No normality assumption is required. No equal-sample-size requirement applies. The result is a p-value and a visual distribution that can be trusted even when the standard t-test cannot.

This program provides a ready-to-run bootstrapping tool for A/B testing of advertising campaign results. It was developed to make bootstrap analysis accessible to practitioners who work with real marketing data but may not have a background in writing statistical code from scratch. The initial framework draws on publicly available Python implementations of A/B bootstrap testing, extended and rebuilt here into a full Streamlit application with a browser-based interface, configurable inputs, publication-quality distribution plots, side-by-side comparison with the standard t-test, automatic Bonferroni correction for multi-group experiments, and structured CSV output. The goal is a tool that a data analyst can pick up, point at their data, and run — without modification.

The directions below show how to set this up in Python. Once set up, the user can run this from a terminal for production or sample testing.

---
## Why Bootstrapping?

The standard two-sample t-test rests on assumptions that real-world marketing data frequently violates:

**Normality** — the t-test assumes that the underlying data, or the sampling distribution of the mean, is approximately normal. Conversion rate data is binary (converted or not), and when conversion rates are very low — as is common in advertising — the distribution is heavily right-skewed. With rare events, the normal approximation breaks down.

**Equal or known variance** — Welch's t-test relaxes the equal-variance assumption somewhat, but the test still performs poorly when group variances differ substantially, which often happens when one group is far more engaged than another.

**Equal or balanced sample sizes** — the t-test's power and accuracy assumptions are best met with roughly equal group sizes. In practice, A/B test groups are frequently unbalanced: a control group may have hundreds of thousands of records while a test group has only tens of thousands. This imbalance inflates Type I error rates and distorts p-values.

**Independence of observations** — the t-test assumes each observation is independent. In advertising data, a single user may appear across multiple touchpoints, introducing correlation that the t-test ignores.

When any of these assumptions are violated, the t-test p-value can be misleading — either falsely declaring significance or missing a real effect.

### How Bootstrapping Works Differently

Bootstrapping is a non-parametric resampling method that makes no assumptions about the shape of the underlying distribution. Instead of relying on a theoretical formula, it builds an empirical sampling distribution directly from your data:

1. Take your observed Group A and Group B samples
2. Repeatedly resample each group *with replacement* (thousands of times)
3. For each resample, compute the difference in conversion rates between the groups
4. Build a distribution of those differences — this is the bootstrap sampling distribution
5. Compare that distribution against a null hypothesis distribution centered at zero to compute a p-value

Because the bootstrap derives its sampling distribution from the actual data rather than a theoretical model, it is robust to skewness, unequal sample sizes, and non-normal distributions. It also provides an intuitive visual: you can *see* whether the observed difference is plausibly explained by chance or not.

The bootstrap does not replace the t-test — it complements it. When both tests agree, your conclusion is on solid ground. When they disagree, that disagreement itself is informative and worth investigating.

---
## About This Program

This application gives you a ready-to-run bootstrapping tool built on top of Python and Streamlit. You do not need to write any code to use it — configure your experiment in the sidebar, click **Run Analysis**, and the program handles everything else.

### Key Features

**Browser-based interface** — runs locally via Streamlit with a sidebar configuration panel. No command line interaction needed after startup. Set your experiment name, data location, file name, column mappings, and group definitions all from the browser.

**Flexible data input** — accepts CSV or pipe-delimited TXT files, or generates synthetic binary test data if you want to explore the tool without a real dataset. Column names are fully configurable so the program adapts to your file rather than the other way around.

**Automatic path handling** — the program detects its own location and builds default data and output folder paths relative to it. Enter `data` and `output` in the folder fields and it resolves the full paths automatically. Full absolute paths are also accepted.

**Side-by-side test comparison** — runs both a standard t-test and a bootstrap test for every group pair, so you can directly compare results. When they agree, your conclusion is well-supported. When they differ, the bootstrap result is the more reliable guide for non-normal or unbalanced data.

**Bonferroni correction** — for experiments with more than two groups (A/B/C testing), the program automatically adjusts significance thresholds to control for multiple comparisons, reducing the risk of false positives.

**Distribution plots** — for every group pair, the program generates a histogram showing the bootstrap sampling distribution of the difference in conversion rates overlaid with the null hypothesis distribution. These plots make it visually clear whether the observed difference is statistically meaningful.

**Per-plot results summary** — immediately below each plot, a results table shows the t-statistic and p-value from the regular t-test alongside the mean difference and p-value from the bootstrap test, with significance stars (* / ** / ***) at the 95%, 99%, and 99.9% levels.

**Automatic output saving** — plots are saved as PNG files and results are saved as a CSV to your output folder. File paths are listed at the end of the run so you know exactly where everything landed.

**Large dataset handling** — for groups with more than 50,000 records, the bootstrap automatically caps each resample at 10,000 rows per iteration, keeping run times manageable without meaningfully affecting the results.

**Progress tracking** — a live progress bar in the browser shows bootstrap iteration progress (0–100%) for each test pair, along with an overall progress indicator across all pairs.

---
## Requirements

```bash
pip install streamlit pandas numpy matplotlib scipy
```

| Package | Purpose |
|---|---|
| `streamlit` | Browser-based UI |
| `pandas` | Data manipulation |
| `numpy` | Numerical calculations |
| `matplotlib` | Distribution plots |
| `scipy` | T-test and normal distribution |

---
## Environment Setup

These commands create an isolated Python environment for the project so that the required packages do not interfere with other Python installations on your machine. Run these once in a terminal. Change the path to match your own folder location.

```bash
# Create the project folder
md /home/tom/Python/Bootstrapping

# Navigate to the project folder
cd /home/tom/Python/Bootstrapping

# Create the virtual environment
python3 -m venv .venv

# Activate it (Mac/Linux)
source .venv/bin/activate

# Activate it (Windows)
# .venv\Scripts\activate

# Install required packages
pip install streamlit pandas numpy matplotlib scipy

# Save the environment so others can recreate it
pip freeze > requirements.txt
```

To deactivate the environment when you are done:

```bash
deactivate
```
---
## Running the App

Run this in a terminal. Change the path to the one you use.

```bash
cd /home/tom/Python/Bootstrapping
source .venv/bin/activate
streamlit run Bootstrapping_for_A-B_Testing.py
```
---
## Usage

### Sidebar Configuration

| Field | Description |
|---|---|
| Main Program Folder | Auto-detected from script location. Edit if needed. |
| Dataset Type | `File` to load CSV/TXT data, `Custom` to generate synthetic data |
| Experiment Name | Used in plot titles and output file names |
| Data Folder | Relative (e.g. `data`) or absolute path to your input file |
| Output Folder | Relative (e.g. `output`) or absolute path for plots and CSV |
| Input File Name | Name of your data file |
| Delimiter | Comma, pipe, or tab |
| ID / Group / Conversion Columns | Your file's column names for each role |
| Bootstrap Iterations (n) | Number of resamples — 10,000 is a good default |

---

## Data Choices

**Your Data** — you can use your own data. Simply enter the input file name and column names in the sidebar.

**Custom Data** — you can run this without any data file. The program will let you choose the sample sizes and conversion rates. The default is three groups.

**Kaggle Data** — you can download the Kaggle Marketing data using the link and inputs shown below.

### Custom Synthetic Data

Enter groups one per line in the format `GroupName, SampleSize, ConversionRate`:

```
A, 10000, 0.002
B, 10000, 0.0025
C, 10000, 0.001
```
---
## Sample Dataset

No data? No problem. This program works with the **Marketing A/B Testing** dataset from Kaggle:

https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing

Download `marketing_AB.csv` and place it in your `data/` folder. The sidebar is pre-configured with the correct column names for this dataset:

| Field | Value |
|---|---|
| ID Column | `user id` |
| Group Column | `test group` |
| Conversion Column | `converted` |

---
## Output Files

After a run, the output folder will contain:
- One PNG plot per group pair
- One CSV file with the full results table

The browser displays the full file paths at the end of the run.

---
## Significance Levels

| Stars | Meaning |
|---|---|
| `***` | p < Bonferroni-adjusted 99.9% threshold |
| `**` | p < Bonferroni-adjusted 99% threshold |
| `*` | p < Bonferroni-adjusted 95% threshold |
| *(blank)* | Not statistically significant |

---
## References

- Duong, B. T. (2021). *A/B Testing for Ad Campaign in Python*. Medium.
  https://baotramduong.medium.com/data-science-project-a-b-testing-for-ad-campaign-in-python-ffaca9170bc4

- DataTipz. *Hypothesis Testing with Bootstrapping in Python*.
  https://www.datatipz.com/blog/hypothesis-testing-with-bootstrapping-python

- Bootstrap Resampling — Wikipedia.
  https://en.wikipedia.org/wiki/Bootstrapping_(statistics)

- Bonferroni Correction — Wikipedia.
  https://en.wikipedia.org/wiki/Bonferroni_correction

- Kaggle Marketing A/B Testing Dataset.
  https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing

---
## Author

**Thomas K. Arnold**
Explorations in Science
arnoldtk@mail.uc.edu

## How to Cite
If you use this software in your research or work, please cite it as:

Arnold, T. K. (2026, April). *Bootstrapping for A/B Testing* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.19683408

---
## License

MIT License — Copyright (c) 2024, 2026 Thomas K. Arnold
Feel free to use, modify, and share with attribution.
