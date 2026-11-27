# Submission checklist (all four fields need https:// links)

1. **GitHub repository** - create a public repo (e.g. `skycity-profit-optimisation`), upload every file in this folder, copy the repo URL.( URL : https://github.com/ThotaSathwika/skycity-profit-optimisation)
2. **Research paper link** - upload `reports/Research_Paper.pdf` to Google Drive → Share → "Anyone with the link" → copy link. Link: (https://drive.google.com/file/d/1mq-kGeHPGHz7UV86LoNxMPib2hxdM0Uc/view?usp=sharing) (Add `Executive_Summary.pdf` the same way. Link:[https://drive.google.com/file/d/1Wkat5ALn0fv522BwiOCPy2Dw9t_jNgdW/view?usp=sharing)]
3. **Deployed app link** - share.streamlit.io → New app → pick the repo, branch `main`, main file `app.py` → Deploy → copy the `https://<name>.streamlit.app` URL. Link : (https://skycity-profit-optimiser.streamlit.app/)
4. **Feedback video link** - record 1-2 min (screen + voice), upload to YouTube as Unlisted or Drive with link sharing, copy the link. Link : (https://drive.google.com/file/d/1pnUm3eZvm_TudMq2UQb_kJWeAsZBTg4o/view?usp=sharing)

## Feedback video script (about 90 seconds)
- Intro: "I built a profit prediction and optimisation tool for SkyCity Auckland restaurants across four channels."
- What I did: cleaned data, found InStoreShare in the file differs from the real order share, engineered features such as commission × share, trained Linear, Random Forest, Gradient Boosting and XGBoost.
- Key learning: with engineered features even a linear model hits R² 1.000 because profit is an accounting identity; trees are needed when only raw inputs exist (R² 0.985 vs 0.892).
- Findings: +3 pp commission costs about $1.4M a month; break-even commission is 35% overall but 12% for full-service; best mix lifts profit about 49%.
- Demo: open the app, move the commission and Uber Eats sliders, show the optimisation tab.
- Challenges: path/data issues, deployment, and learning to state limits (fixed order volume).
- Close: what I would add next - time-series demand and customer-retention effects.
