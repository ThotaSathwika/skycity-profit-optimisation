# Predictive Modelling & Profit Optimisation for Multi-Channel Restaurant Operations
SkyCity Auckland Restaurants & Bars · Unified Mentor project

Predicts monthly net profit from channel mix, commission and delivery-cost decisions, simulates what-if scenarios and recommends a profit-maximising mix.

## Run locally
```bash
pip install -r requirements.txt
python train_model.py   # trains + evaluates models  -> models/
python analysis.py      # EDA, scenarios, figures    -> figures/, models/results.json
streamlit run app.py    # dashboard
```

## Structure
| File | Purpose |
|---|---|
| `utils.py` | feature engineering, accounting engine, scenario + optimisation functions |
| `train_model.py` | Linear, Random Forest, Gradient Boosting, XGBoost; RMSE / MAE / R² / CV |
| `analysis.py` | EDA, sensitivity, commission and radius sweeps, optimisation, figures |
| `app.py` | Streamlit dashboard (profit dashboard, what-if sliders, sensitivity, optimiser) |
| `reports/` | Research paper and executive summary (.docx and .pdf) |

## Headline results (1,696 restaurants)
- Baseline predicted profit: **$7.87M / month**; 24% of restaurants lose money (all in the Full-service segment).
- +3 pp commission costs about **$1.43M / month** (−18%); Uber Eats +10 pp costs **$1.39M**.
- Median break-even commission is **34.7%**; for Full-service it is only **12.4%**.
- Best in-range channel mix lifts profit by **+48.7%**.
- Best models: XGBoost R² 0.994 (engineered features); linear on engineered features R² 1.000.

## Deploy (Streamlit Community Cloud)
Push this folder to GitHub → share.streamlit.io → New app → repo, branch `main`, file `app.py`.
