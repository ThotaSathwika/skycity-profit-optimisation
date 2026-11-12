"""SkyCity Auckland - Profit Prediction & Optimisation dashboard.  Run: streamlit run app.py"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import *

ROOT = Path(__file__).parent
st.set_page_config("SkyCity Profit Optimiser", "🍽️", layout="wide")

import plotly.io as pio
pio.templates.default = "plotly_dark"
INK, SURF, TXT, MUTED, GRID = "#130E24", "#1E1838", "#F5F1FF", "#A79FC9", "rgba(167,159,201,.16)"
CC = {"InStore": "#FFB627", "UberEats": "#2EE6A6", "DoorDash": "#FF4D6D", "SelfDelivery": "#7B8CFF"}
NAMES = {"InStore": "In-store", "UberEats": "Uber Eats", "DoorDash": "DoorDash", "SelfDelivery": "Self-delivery"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,800&family=Figtree:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Figtree', sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 85% -10%, rgba(123,140,255,.22), transparent 60%),
                     radial-gradient(900px 420px at -5% 0%, rgba(255,77,109,.16), transparent 55%), #130E24; }
h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Bricolage Grotesque', sans-serif !important; letter-spacing: -0.02em; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: #1A1433; border-right: 1px solid rgba(167,159,201,.14); }
[data-testid="stSidebar"] h1 { font-size: 1.5rem; }
.hero h1 { font-size: clamp(2rem, 3.8vw, 3.2rem); line-height: 1.02; font-weight: 800; margin: 0 0 .9rem 0; padding: 0; }
.hero p { color: #A79FC9; font-size: 1.08rem; max-width: 34rem; line-height: 1.55; margin: 0 0 1.4rem 0; }
.chips { margin-bottom: 1.2rem; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .7rem; max-width: 36rem; }
.chip { background: rgba(30,24,56,.85); border: 1px solid rgba(167,159,201,.18); border-radius: 14px; padding: .7rem .95rem; }
.chip b { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.05rem; font-weight: 800; }
.chip span { color: #A79FC9; font-size: .86rem; display: block; margin-top: .1rem; }
.dot { display: inline-block; width: .7rem; height: .7rem; border-radius: 50%; margin-right: .5rem; vertical-align: middle; }
[data-testid="stMetric"] { background: #1E1838; border: 1px solid rgba(167,159,201,.16); border-top: 3px solid #FFB627; border-radius: 14px; padding: .9rem 1rem; }
[data-testid="stMetricLabel"] p { color: #A79FC9; font-size: .85rem; }
[data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 800; }
[data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid rgba(167,159,201,.16); }
[data-baseweb="tab"] { border-radius: 999px 999px 0 0; padding: .55rem 1.1rem; color: #A79FC9; }
[data-baseweb="tab"][aria-selected="true"] { color: #130E24; background: #FFB627; font-weight: 600; }
[data-baseweb="tab-highlight"] { display: none; }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
@media (max-width: 700px) { .chips { grid-template-columns: 1fr; } }
</style>
""", unsafe_allow_html=True)


def themed(fig, h=380):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h, margin=dict(l=10, r=10, t=60, b=10),
                      font=dict(family="Figtree, sans-serif", color=TXT), title_font=dict(family="Bricolage Grotesque, sans-serif", size=20),
                      legend=dict(bgcolor="rgba(0,0,0,0)"), hoverlabel=dict(bgcolor=SURF, font_color=TXT))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


@st.cache_data
def get_data(): return load_data(str(ROOT / "data/skycity.csv"))


@st.cache_resource
def get_models():
    m = {"Linear (engineered features)": joblib.load(ROOT / "models/profit_model.joblib"),
         "XGBoost (engineered features)": joblib.load(ROOT / "models/xgb_model.joblib")}
    ch = {c: joblib.load(ROOT / f"models/{c}Margin_model.joblib") for c in CH}
    return m, ch, json.load(open(ROOT / "models/meta.json"))


df_all = get_data(); models, ch_models, meta = get_models()
band_q = {"Linear (engineered features)": meta["resid_q90"], "XGBoost (engineered features)": meta["xgb_q90"]}

# ---------------- sidebar
sb = st.sidebar
sb.title("🍽️ Scenario controls")
model_name = sb.selectbox("Prediction model", list(models)); model = models[model_name]
sb.subheader("Scope")
seg = sb.multiselect("Segment", sorted(df_all.Segment.unique()), default=sorted(df_all.Segment.unique()))
cui = sb.multiselect("Cuisine", sorted(df_all.CuisineType.unique()), default=sorted(df_all.CuisineType.unique()))
reg = sb.multiselect("Subregion", sorted(df_all.Subregion.unique()), default=sorted(df_all.Subregion.unique()))
base = df_all[df_all.Segment.isin(seg) & df_all.CuisineType.isin(cui) & df_all.Subregion.isin(reg)].reset_index(drop=True)
if base.empty: st.warning("No restaurants match the filters."); st.stop()

sb.subheader("Channel mix (what-if)")
d_in = sb.slider("In-store share change (pp)", -10, 20, 0) / 100
d_ue = sb.slider("Uber Eats share of delivery (pp)", -15, 25, 0) / 100
d_dd = sb.slider("DoorDash share of delivery (pp)", -10, 10, 0) / 100
sb.subheader("Costs")
d_comm = sb.slider("Commission rate change (pp)", -10.0, 15.0, 0.0, 0.5) / 100
d_km = sb.slider("Delivery radius change (km)", -5, 10, 0)
cost_mult = sb.slider("Delivery cost per order change (%)", -30, 50, 0) / 100
sb.subheader("Demand")
growth = sb.slider("Monthly growth factor (0 = use each restaurant's own)", 0.0, 1.08, 0.0, 0.01)
months = sb.slider("Projection horizon (months)", 1, 24, 12)


def shift_mix(s, du, dd_):
    if du == 0 and dd_ == 0: return s
    ue, dd = (s.UE_share + du).clip(0, 1), (s.DD_share + dd_).clip(0, 1)
    sd = (1 - ue - dd).clip(lower=0); t = ue + dd + sd
    s["UE_share"], s["DD_share"], s["SD_share"] = ue / t, dd / t, sd / t
    return s


def build(b):
    s = b.copy()
    if d_in: s["in_sh"] = (s.in_sh + d_in).clip(0.02, 0.95)
    s = shift_mix(s, d_ue, d_dd)
    s["CommissionRate"] = (s.CommissionRate + d_comm).clip(0, 0.9)
    new_km = (s.DeliveryRadiusKM + d_km).clip(1, 40)
    s["DeliveryCostPerOrder"] = s.DeliveryCostPerOrder * new_km / s.DeliveryRadiusKM * (1 + cost_mult)
    s["DeliveryRadiusKM"] = new_km
    if growth > 0: s["GrowthFactor"] = growth
    return s


scn = build(base); n = len(base)
p_base, p_scn = predict(model, base).sum(), predict(model, scn).sum()
band = band_q[model_name] * np.sqrt(n)
best = optimise(model, base).iloc[0]
be = breakeven_commission(scn)
psi = sensitivity_index(model, base)
es, eb = engineer(scn), engineer(base)


share = {"InStore": es.in_sh, "UberEats": es.ue_sh, "DoorDash": es.dd_sh, "SelfDelivery": es.sd_sh}
acc_h = account(scn)
orders_h = {c: (scn.MonthlyOrders * share[c]).sum() for c in CH}
shares_h = {c: orders_h[c] / scn.MonthlyOrders.sum() for c in CH}
per_order = {c: acc_h[c].sum() / max(orders_h[c], 1) for c in CH}
hl, hr = st.columns([6, 5], gap="large")
with hl:
    chips = "".join(f'<div class="chip"><b><i class="dot" style="background:{CC[c]}"></i>{NAMES[c]}</b>'
                    f'<span>{shares_h[c]:.0%} of orders · ${per_order[c]:,.2f} profit per order</span></div>' for c in CH)
    st.markdown(f'<div class="hero"><h1>Which orders actually make money?</h1>'
                f'<p>Slide commission, channel mix and delivery cost, and watch profit, risk and the best mix change for {n:,} Auckland restaurants.</p>'
                f'<div class="chips">{chips}</div></div>', unsafe_allow_html=True)
with hr:
    plate = go.Figure(go.Pie(labels=[NAMES[c] for c in CH], values=[shares_h[c] for c in CH], hole=0.74, sort=False, direction="clockwise",
                             marker=dict(colors=[CC[c] for c in CH], line=dict(color=INK, width=5)), textinfo="none",
                             hovertemplate="%{label}: %{percent}<extra></extra>"))
    plate.add_annotation(text=f"<b>${p_scn / 1e6:,.2f}M</b>", x=0.5, y=0.56, showarrow=False, font=dict(family="Bricolage Grotesque, sans-serif", size=44, color=TXT))
    plate.add_annotation(text="net profit per month", x=0.5, y=0.43, showarrow=False, font=dict(size=14, color=MUTED))
    dlt = p_scn - p_base
    plate.add_annotation(text=("no change vs baseline" if abs(dlt) < 1 else f"{dlt:+,.0f} vs baseline"), x=0.5, y=0.35, showarrow=False,
                         font=dict(size=14, color="#2EE6A6" if dlt >= 0 else "#FF4D6D"))
    plate.update_layout(showlegend=False, height=400, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Figtree, sans-serif"))
    st.plotly_chart(plate, width="stretch", config={"displayModeBar": False})

st.caption(f"SkyCity Auckland Restaurants & Bars · model: {model_name}")

k = st.columns(5)
k[0].metric("Predicted net profit / month", f"${p_scn:,.0f}", f"{p_scn - p_base:+,.0f} vs baseline")
k[1].metric("Net profit per order", f"${p_scn / scn.MonthlyOrders.sum():,.2f}", f"{(p_scn - p_base) / scn.MonthlyOrders.sum():+.2f}")
k[2].metric("Profit Sensitivity Index", f"{psi.abs().max():.2f}", f"top driver: {psi.index[0]}", delta_color="off",
            help="% change in profit for a +1% change in the driver. Higher = more vulnerable.")
k[3].metric("Break-even commission (median)", f"{be.ZeroProfit.median():.1%}", f"current avg {scn.CommissionRate.mean():.1%}", delta_color="off",
            help="Commission at which an aggregator order earns zero profit (1 - COGS - OPEX).")
k[4].metric("Optimisation uplift", f"{(best.Profit - p_scn) / abs(p_scn):+.1%}",
            f"mix efficiency {p_scn / best.Profit:.0%}" if best.Profit > 0 else "", delta_color="off",
            help="Gain of the best in-range mix vs the current scenario.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Profit dashboard", "🔀 Channel mix & what-if", "🌡️ Cost sensitivity", "🎯 Optimisation", "ℹ️ Model & data"])

with tab1:
    a, b = st.columns([3, 2])
    pj = pd.DataFrame([(t, predict(model, scn.assign(MonthlyOrders=scn.MonthlyOrders * scn.GrowthFactor ** t)).sum(),
                        predict(model, base.assign(MonthlyOrders=base.MonthlyOrders * base.GrowthFactor ** t)).sum())
                       for t in range(months + 1)], columns=["Month", "Scenario", "Baseline"])
    f = go.Figure()
    f.add_scatter(x=pj.Month, y=pj.Baseline, name="Baseline", line=dict(color="#A79FC9", dash="dash"))
    f.add_scatter(x=pj.Month, y=pj.Scenario + band, line=dict(width=0), showlegend=False)
    f.add_scatter(x=pj.Month, y=pj.Scenario - band, fill="tonexty", fillcolor="rgba(255,182,39,.20)", line=dict(width=0), name="90% band")
    f.add_scatter(x=pj.Month, y=pj.Scenario, name="Scenario", line=dict(color="#FFB627", width=4))
    f.update_layout(title="Projected monthly net profit (demand growth compounding)", xaxis_title="Months ahead", yaxis_title="$"); themed(f)
    a.plotly_chart(f, width="stretch")
    share = {"InStore": es.in_sh, "UberEats": es.ue_sh, "DoorDash": es.dd_sh, "SelfDelivery": es.sd_sh}
    rev = {c: scn.MonthlyOrders * share[c] * scn.AOV for c in CH}
    X = es[ENG_NUM + CATS]
    cdf = pd.DataFrame({"Channel": CH, "Net profit": [float((ch_models[c].predict(X) * rev[c]).sum()) for c in CH],
                        "Revenue": [rev[c].sum() for c in CH]})
    cdf["Margin"] = cdf["Net profit"] / cdf.Revenue
    cdf["Name"] = cdf.Channel.map(NAMES)
    fb = px.bar(cdf, x="Name", y="Net profit", color="Channel", color_discrete_map=CC, title="Profit by channel (margin on bar)", text=cdf.Margin.map("{:.1%}".format))
    fb.update_layout(showlegend=False, xaxis_title=None); fb.update_traces(textposition="outside", cliponaxis=False); b.plotly_chart(themed(fb), width="stretch")
    st.subheader("Risk indicators")
    acc, acc0 = account(scn), account(base)
    r = st.columns(4)
    r[0].metric("Restaurants losing money", f"{(acc.Total < 0).mean():.1%}", f"{(acc.Total < 0).mean() - (acc0.Total < 0).mean():+.1%} vs baseline", delta_color="inverse")
    r[1].metric("Aggregator-loss restaurants", f"{((acc.UberEats < 0) | (acc.DoorDash < 0)).mean():.1%}", help="Share where Uber Eats or DoorDash orders lose money")
    r[2].metric("Commission headroom", f"{(be.ZeroProfit - scn.CommissionRate).median() * 100:+.1f} pp", help="Median gap between break-even and current commission")
    r[3].metric("Profit at 90% band (low / high)", f"${p_scn - band:,.0f} / ${p_scn + band:,.0f}")
    if model_name.startswith("Linear"):
        st.caption("The linear model recovers the accounting identity almost exactly, so its band is very narrow; switch to XGBoost for a more conservative band.")

with tab2:
    c1, c2 = st.columns(2)
    mix = pd.DataFrame({"Channel": CH * 2, "Scenario": ["Baseline"] * 4 + ["What-if"] * 4,
                        "Share": [eb.in_sh.mean(), eb.ue_sh.mean(), eb.dd_sh.mean(), eb.sd_sh.mean(), es.in_sh.mean(), es.ue_sh.mean(), es.dd_sh.mean(), es.sd_sh.mean()]})
    mix["Channel"] = mix.Channel.map(NAMES)
    c1.plotly_chart(themed(px.bar(mix, x="Channel", y="Share", color="Scenario", barmode="group", title="Share of all orders",
                                  color_discrete_map={"Baseline": "#6E65A0", "What-if": "#FFB627"}).update_yaxes(tickformat=".0%")), width="stretch")
    sweep = pd.DataFrame([(c * 100, predict(model, scn.assign(CommissionRate=c)).sum()) for c in np.arange(0.15, 0.46, 0.01)], columns=["Commission %", "Profit"])
    c2.plotly_chart(themed(px.line(sweep, x="Commission %", y="Profit", title="Profit vs commission rate", color_discrete_sequence=["#FF4D6D"])), width="stretch")
    st.subheader("Preset scenarios")
    presets = {"Uber Eats +10pp (from self-delivery)": (0, .10, 0, 0, 0), "Uber Eats −10pp (to self-delivery)": (0, -.10, 0, 0, 0),
               "Commission +3pp": (0, 0, 0, .03, 0), "Commission −3pp": (0, 0, 0, -.03, 0), "Radius +3 km": (0, 0, 0, 0, 3), "In-store +5pp": (.05, 0, 0, 0, 0)}
    rows = []
    for name, (a_, u_, d_, c_, k_) in presets.items():
        s = base.copy(); s["in_sh"] = (s.in_sh + a_).clip(.03, .95); s = shift_mix(s, u_, d_)
        s["CommissionRate"] += c_; new_km = s.DeliveryRadiusKM + k_
        s["DeliveryCostPerOrder"] = s.DeliveryCostPerOrder * new_km / s.DeliveryRadiusKM; s["DeliveryRadiusKM"] = new_km
        p = predict(model, s).sum(); rows.append((name, p, p - p_base, (p - p_base) / abs(p_base)))
    pr = pd.DataFrame(rows, columns=["Scenario", "Profit / month", "Δ $", "Δ %"])
    st.dataframe(pr.style.format({"Profit / month": "${:,.0f}", "Δ $": "{:+,.0f}", "Δ %": "{:+.1%}"}), hide_index=True, width="stretch")
    st.info("Radius changes affect cost only: the data shows no link between radius and order volume, so no demand uplift is assumed.")

with tab3:
    c1, c2 = st.columns(2)
    pdf = psi.sort_values().rename("PSI").rename_axis("Driver").reset_index()
    pb = px.bar(pdf, x="PSI", y="Driver", orientation="h", title="Profit Sensitivity Index (elasticity)")
    pb.update_traces(marker_color=["#FF4D6D" if v < 0 else "#2EE6A6" for v in pdf.PSI]); c1.plotly_chart(themed(pb), width="stretch")
    grid = pd.DataFrame([(c * 100, k_, predict(model, scn.assign(CommissionRate=c, DeliveryRadiusKM=k_, DeliveryCostPerOrder=COST_PER_KM * k_)).sum())
                         for c in np.arange(0.20, 0.41, 0.025) for k_ in range(3, 19, 3)], columns=["Commission %", "Radius km", "Profit"])
    c2.plotly_chart(themed(px.density_heatmap(grid, x="Commission %", y="Radius km", z="Profit", histfunc="avg", color_continuous_scale=["#FF4D6D", "#2A2150", "#2EE6A6"], title="Profit: commission × delivery radius")), width="stretch")
    st.subheader("Break-even commission by segment")
    t = scn.assign(ZeroProfit=be.ZeroProfit, SD_Parity=be.SD_Parity).groupby("Segment").agg(
        Current=("CommissionRate", "mean"), BreakEven=("ZeroProfit", "median"), SD_Parity=("SD_Parity", "median"))
    min_m = st.slider("Minimum acceptable aggregator margin (pp)", 0, 15, 5) / 100
    t["Safe ceiling"] = (t.BreakEven - min_m).clip(lower=0)
    st.dataframe(t.style.format("{:.1%}"), width="stretch")
    st.caption("Break-even = 1 − COGS − OPEX. Safe ceiling keeps the aggregator margin above your minimum. SD parity = delivery cost ÷ AOV, the commission at which Uber Eats equals self-delivery per order.")

with tab4:
    shift = st.slider("Max change in in-store share vs today (pp)", 0, 20, 10) / 100
    o = optimise(model, base, max_in_shift=shift); b0 = o.iloc[0]
    cur = [eb.in_sh.mean(), eb.ue_sh.mean(), eb.dd_sh.mean(), eb.sd_sh.mean()]
    opt_tot = [b0.InStore, (1 - b0.InStore) * b0.UE, (1 - b0.InStore) * b0.DD, (1 - b0.InStore) * b0.SD]
    st.success(f"Best in-range mix lifts predicted profit from \\${p_base:,.0f} to \\${b0.Profit:,.0f} per month ({(b0.Profit - p_base) / abs(p_base):+.1%}).")
    cmp_ = pd.DataFrame({"Channel": [NAMES[c] for c in CH] * 2, "Mix": ["Current"] * 4 + ["Optimised"] * 4, "Share": cur + opt_tot})
    c1, c2 = st.columns(2)
    c1.plotly_chart(themed(px.bar(cmp_, x="Channel", y="Share", color="Mix", barmode="group", title="Current vs profit-maximising mix",
                                  color_discrete_map={"Current": "#6E65A0", "Optimised": "#2EE6A6"}).update_yaxes(tickformat=".0%")), width="stretch")
    c2.dataframe(o.head(10).style.format({"InStore": "{:.0%}", "UE": "{:.0%}", "DD": "{:.0%}", "SD": "{:.0%}", "Profit": "${:,.0f}"}), hide_index=True, width="stretch")
    st.caption("Search space = shares observed in the data (Uber Eats 35–60%, DoorDash 20–30%, self-delivery 15–45% of delivery orders). Assumes order volume and self-delivery cost per order stay fixed; validate capacity before acting.")
    st.subheader("Self-delivery investment threshold")
    F = st.number_input("Extra monthly fixed cost of a delivery fleet ($)", 0, 50000, 2000, 500)
    per = scn.CommissionRate * scn.AOV - scn.DeliveryCostPerOrder
    st.metric("Self-delivery orders / month needed to break even", f"{(F / per.clip(lower=0.01)).median():,.0f}",
              f"saves ${per.median():.2f} per order vs Uber Eats (median)", delta_color="off")
    st.caption(f"Median self-delivery volume today: {base.SelfDeliveryOrders.median():,.0f} orders/month.")

with tab5:
    cmp = pd.read_csv(ROOT / "models/model_comparison.csv")
    st.subheader("Model comparison (20% hold-out, total monthly net profit)")
    st.dataframe(cmp.style.format({"RMSE": "{:,.1f}", "MAE": "{:,.1f}", "R2": "{:.4f}", "CV_R2": "{:.4f}"}), hide_index=True, width="stretch")
    st.markdown("Engineered features (revenue, contribution margin, commission × share, delivery cost × share) let a linear model recover profit almost exactly; "
                "tree models matter when only raw decision variables are available (R² 0.98+ vs 0.89 for linear).")
    st.subheader("Data notes")
    st.markdown("- `in_sh` = in-store orders ÷ all orders (the supplied `InStoreShare` column differs by ~5 pp on average).\n"
                "- `UE_share`, `DD_share`, `SD_share` are shares of **delivery** orders; total-order shares = (1 − in_sh) × share.\n"
                "- `DeliveryCostPerOrder` ≈ 0.295 × `DeliveryRadiusKM` in the data.")
    st.dataframe(base.head(200), width="stretch")
