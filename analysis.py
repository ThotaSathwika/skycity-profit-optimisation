"""EDA + scenario results + figures used in the paper. Run: python analysis.py"""
import json, joblib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

sns.set_theme(style="whitegrid", font_scale=0.9)
df = load_data(); model = joblib.load("models/profit_model.joblib")
base = df.copy(); E = engineer(df); R = {}
P = lambda f: predict(model, f).sum()
p0 = P(base); R["baseline_total"] = p0; R["baseline_avg"] = p0 / len(df)
R["actual_total"] = df.TotalNetProfit.sum()
R["loss_making_share"] = float((df.TotalNetProfit < 0).mean())


def save(name): plt.tight_layout(); plt.savefig(f"figures/{name}.png", dpi=160); plt.close()


fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
sns.histplot(df.TotalNetProfit, bins=40, ax=ax[0], color="#2a6f97"); ax[0].set_title("Monthly net profit per restaurant"); ax[0].set_xlabel("$")
order = df.groupby("Segment").TotalNetProfit.mean().sort_values().index
sns.barplot(df, x="Segment", y="TotalNetProfit", order=order, ax=ax[1], color="#61a5c2", errorbar=None)
ax[1].set_title("Mean net profit by segment"); ax[1].set_ylabel("$"); save("eda_profit")

m = df[[c + "Margin" for c in CH]].rename(columns=lambda c: c.replace("Margin", ""))
fig, ax = plt.subplots(figsize=(5.5, 3.3)); sns.boxplot(m, ax=ax, palette="Blues", showfliers=False)
ax.axhline(0, color="red", lw=.8); ax.set_title("Net margin by channel"); ax.set_ylabel("Net profit / revenue"); save("eda_margin")

cols = ["TotalNetProfit", "ProfitPerOrder", "in_sh", "UE_share", "SD_share", "CommissionRate", "DeliveryRadiusKM",
        "GrowthFactor", "MonthlyOrders", "AOV", "COGSRate", "OPEXRate"]
fig, ax = plt.subplots(figsize=(6.5, 5)); sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 6.5}, cbar=False)
ax.set_title("Correlation matrix"); save("eda_corr")

cmp = pd.read_csv("models/model_comparison.csv")
fig, ax = plt.subplots(figsize=(7, 3.2)); sns.barplot(cmp, x="Model", y="RMSE", hue="FeatureSet", ax=ax, palette=["#9ecae1", "#08519c"])
ax.set_yscale("log"); ax.set_title("Test RMSE by model and feature set (log scale)"); ax.set_ylabel("RMSE ($)"); save("model_rmse")

imp = pd.read_csv("models/feature_importance.csv", index_col=0).iloc[:, 0].sort_values()
fig, ax = plt.subplots(figsize=(5.5, 3.5)); imp.rename(lambda s: s.replace("num__", "")).plot.barh(ax=ax, color="#2a6f97")
ax.set_title("Gradient Boosting feature importance"); save("feature_importance")

psi = sensitivity_index(model, base); R["psi"] = psi.round(3).to_dict()
fig, ax = plt.subplots(figsize=(5.5, 3)); psi.sort_values().plot.barh(ax=ax, color=["#c0392b" if v < 0 else "#27ae60" for v in psi.sort_values()])
ax.set_title("Profit Sensitivity Index (% profit change per +1% shock)"); save("psi")

rows = []
for c in np.arange(0.20, 0.451, 0.01):
    s = scenario_frame(base, commission=c); a = account(s)
    rows.append((c, P(s), (a.UberEats < 0).mean(), (a.DoorDash < 0).mean()))
cs = pd.DataFrame(rows, columns=["Commission", "Profit", "UE_loss_share", "DD_loss_share"]); cs.to_csv("models/commission_sweep.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(cs.Commission * 100, cs.Profit / 1e3); ax[0].axvline(30, ls="--", c="grey"); ax[0].set_title("Portfolio profit vs commission"); ax[0].set_xlabel("Commission %"); ax[0].set_ylabel("$ thousand / month")
ax[1].plot(cs.Commission * 100, cs.UE_loss_share * 100); ax[1].axvline(30, ls="--", c="grey"); ax[1].set_title("Restaurants losing money on Uber Eats"); ax[1].set_xlabel("Commission %"); ax[1].set_ylabel("% of restaurants"); save("commission_sweep")
R["comm_slope_per_pp"] = float((cs.Profit.iloc[11] - cs.Profit.iloc[9]) / 2)

be = breakeven_commission(df)
R["breakeven_median"] = float(be.ZeroProfit.median()); R["sd_parity_median"] = float(be.SD_Parity.median())
R["ue_loss_now"] = float((df.CommissionRate > be.ZeroProfit).mean())
R["breakeven_by_segment"] = df.assign(b=be.ZeroProfit).groupby("Segment").b.median().round(3).to_dict()
R["safe_comm_by_segment"] = df.assign(b=be.ZeroProfit - 0.05).groupby("Segment").b.median().round(3).clip(lower=0).to_dict()

rs = [(r, P(scenario_frame(base, radius=r))) for r in range(3, 19)]
fig, ax = plt.subplots(figsize=(5.5, 3)); ax.plot(*zip(*rs)); ax.set_title("Profit vs delivery radius (cost only)"); ax.set_xlabel("Radius km"); ax.set_ylabel("$ / month"); save("radius")
R["sd_threshold_radius_km"] = float((df.CommissionRate * df.AOV / COST_PER_KM).median())

sc = {"Baseline": base,
      "UE +10pp (from Self-Delivery)": scenario_frame(base, mix=(base.UE_share + .10, base.DD_share, base.SD_share - .10)),
      "UE -10pp (to Self-Delivery)": scenario_frame(base, mix=(base.UE_share - .10, base.DD_share, base.SD_share + .10)),
      "Commission +3pp": scenario_frame(base, d_comm=.03), "Commission -3pp": scenario_frame(base, d_comm=-.03),
      "Radius +3km": scenario_frame(base, radius=(base.DeliveryRadiusKM + 3).clip(upper=21)),
      "In-store +5pp (from delivery)": scenario_frame(base, in_sh=base.in_sh + .05)}
tab = pd.Series({k: P(v) for k, v in sc.items()}); R["scenarios"] = tab.round(0).to_dict()
d = (tab - tab["Baseline"]).drop("Baseline") / 1e3
fig, ax = plt.subplots(figsize=(6, 3.3)); d.sort_values().plot.barh(ax=ax, color=["#c0392b" if v < 0 else "#27ae60" for v in d.sort_values()])
ax.set_title("Scenario impact on monthly profit ($ thousand)"); save("scenarios")

h = np.arange(0, 13); keep = [P(scenario_frame(base, months=t)) for t in h]
erode = [P(scenario_frame(base.assign(CommissionRate=base.CommissionRate + .03), months=t)) for t in h]
fig, ax = plt.subplots(figsize=(5.5, 3)); ax.plot(h, np.array(keep) / 1e3, label="Current commission"); ax.plot(h, np.array(erode) / 1e3, label="Commission +3pp")
ax.legend(); ax.set_xlabel("Months"); ax.set_ylabel("$ thousand"); ax.set_title("12-month profit path (observed growth factors)"); save("growth")
R["growth_12m"] = float(keep[-1] / keep[0] - 1); R["growth_12m_erode"] = float(erode[-1] / keep[0] - 1)

opt = optimise(model, base); b = opt.iloc[0]; R["opt_best"] = b.round(3).to_dict(); R["opt_uplift"] = float((b.Profit - p0) / abs(p0))
seg = {}
for sname, g in base.groupby("Segment"):
    o = optimise(model, g); bp = P(g)
    seg[sname] = dict(uplift=float((o.iloc[0].Profit - bp) / abs(bp)), base_profit=float(bp), best_profit=float(o.iloc[0].Profit),
                      **o.iloc[0].drop("Profit").round(2).to_dict())
R["opt_by_segment"] = seg
fs = base[base.Segment == "Full-service"]
R["fs_base"] = float(P(fs)); R["fs_no_agg"] = float(P(scenario_frame(fs, mix=(0.0, 0.0, 1.0)))); R["fs_n"] = int(len(fs))
R["inv_threshold_orders_F2000"] = float((2000 / (df.CommissionRate * df.AOV - df.DeliveryCostPerOrder)).median())
R["median_sd_orders"] = float(df.SelfDeliveryOrders.median())
a = account(base); sh = pd.Series({"InStore": E.in_sh.mean(), "UberEats": E.ue_sh.mean(), "DoorDash": E.dd_sh.mean(), "SelfDelivery": E.sd_sh.mean()})
R["efficiency_per_pp"] = (a[CH].mean() / (sh * 100)).round(1).to_dict()
R["mix_efficiency_score"] = float(p0 / b.Profit)

x = joblib.load("models/xgb_model.joblib"); R["xgb_baseline"] = float(predict(x, base).sum()); R["xgb_ue10"] = float(predict(x, sc["UE +10pp (from Self-Delivery)"]).sum())
json.dump(R, open("models/results.json", "w"), indent=1, default=float); print(json.dumps(R, indent=1, default=float))
