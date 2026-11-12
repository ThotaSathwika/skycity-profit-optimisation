"""Train, evaluate and save the profit models. Run: python train_model.py"""
import json, joblib
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from utils import *

df = engineer(load_data())
q1, q3 = df.ProfitPerOrder.quantile([.25, .75]); iqr = q3 - q1
n_out = int(((df.ProfitPerOrder < q1 - 3 * iqr) | (df.ProfitPerOrder > q3 + 3 * iqr)).sum())


def pipe(model, cols, scale):
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATS), ("num", num, cols)])
    return Pipeline([("pre", pre), ("model", model)])


MODELS = {"Linear Regression": (lambda: LinearRegression(), True),
          "Random Forest": (lambda: RandomForestRegressor(300, min_samples_leaf=2, random_state=42, n_jobs=-1), False),
          "Gradient Boosting": (lambda: GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42), False),
          "XGBoost": (lambda: XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=42), False)}
SETS = {"Raw features": RAW_NUM, "Engineered features": ENG_NUM}

tr, te = train_test_split(df, test_size=0.2, random_state=42)
kf = KFold(5, shuffle=True, random_state=42)
rows, fitted = [], {}
for sname, cols in SETS.items():
    for mname, (mk, scale) in MODELS.items():
        p = pipe(mk(), cols, scale).fit(tr[cols + CATS], tr.TotalNetProfit)
        pr = p.predict(te[cols + CATS])
        cv = cross_val_score(pipe(mk(), cols, scale), tr[cols + CATS], tr.TotalNetProfit, cv=kf, scoring="r2").mean()
        rows.append(dict(FeatureSet=sname, Model=mname, RMSE=mean_squared_error(te.TotalNetProfit, pr) ** .5,
                         MAE=mean_absolute_error(te.TotalNetProfit, pr), R2=r2_score(te.TotalNetProfit, pr), CV_R2=cv))
        fitted[(sname, mname)] = p
res = pd.DataFrame(rows).round(4); res.to_csv("models/model_comparison.csv", index=False); print(res.to_string())

best = res[res.FeatureSet == "Engineered features"].sort_values("RMSE").iloc[0]
model = fitted[("Engineered features", best.Model)]
pred = model.predict(te[ENG_NUM + CATS])
resid_q90 = float(np.quantile(np.abs(te.TotalNetProfit - pred), 0.90))
joblib.dump(model, "models/profit_model.joblib")
xgb = fitted[("Engineered features", "XGBoost")]
xgb_q90 = float(np.quantile(np.abs(te.TotalNetProfit - xgb.predict(te[ENG_NUM + CATS])), 0.90))
joblib.dump(xgb, "models/xgb_model.joblib")

extra = {}
mk, scale = MODELS[best.Model]
for tgt in ["ProfitPerOrder"] + [c + "Margin" for c in CH]:
    m = pipe(mk(), ENG_NUM, scale).fit(tr[ENG_NUM + CATS], tr[tgt]); p = m.predict(te[ENG_NUM + CATS])
    extra[tgt] = dict(RMSE=float(mean_squared_error(te[tgt], p) ** .5), MAE=float(mean_absolute_error(te[tgt], p)),
                      R2=float(r2_score(te[tgt], p)))
    joblib.dump(m, f"models/{tgt}_model.joblib")
print(pd.DataFrame(extra).T.round(4))

gb = fitted[("Engineered features", "Gradient Boosting")]
names = gb.named_steps["pre"].get_feature_names_out()
pd.Series(gb.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False).head(12).to_csv("models/feature_importance.csv")

json.dump(dict(best_model=best.Model, resid_q90=resid_q90, xgb_q90=xgb_q90, n_outliers=n_out, extra_targets=extra,
               train_rows=len(tr), test_rows=len(te)), open("models/meta.json", "w"), indent=1)
print("best:", best.Model, "| outliers flagged:", n_out)
