"""Shared helpers: data loading, feature engineering, scenario + optimisation engine."""
import itertools
import numpy as np
import pandas as pd

DATA_PATH = "data/skycity.csv"
CATS = ["CuisineType", "Segment", "Subregion"]
CH = ["InStore", "UberEats", "DoorDash", "SelfDelivery"]
COST_PER_KM = 0.295  # DeliveryCostPerOrder is almost exactly 0.295 x radius (corr = 1.00)

RAW_NUM = ["in_sh", "UE_share", "DD_share", "SD_share", "CommissionRate", "DeliveryCostPerOrder",
           "DeliveryRadiusKM", "GrowthFactor", "MonthlyOrders", "AOV", "COGSRate", "OPEXRate"]
ENG_NUM = RAW_NUM + ["ue_sh", "dd_sh", "sd_sh", "Revenue", "ContributionMargin", "GrossContribution",
                     "AggCommissionCost", "UE_CommissionCost", "SD_DeliveryCost", "DeliveryCostRatio",
                     "Comm_x_UEshare", "DelCost_x_SDshare", "GrowthAdjOrders", "CostToRevenue"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["TotalNetProfit"] = df[[c + "NetProfit" for c in CH]].sum(axis=1)
    df["TotalRevenue"] = df[[c + "Revenue" for c in CH]].sum(axis=1)
    df["ProfitPerOrder"] = df.TotalNetProfit / df.MonthlyOrders
    for c in CH:
        df[c + "Margin"] = df[c + "NetProfit"] / df[c + "Revenue"]
    # realised in-store share of ALL orders (the supplied InStoreShare column deviates ~5pp from this)
    df["in_sh"] = df.InStoreOrders / df.MonthlyOrders
    return df


def engineer(df):
    """Add model-ready features."""
    d = df.copy()
    d["ue_sh"], d["dd_sh"], d["sd_sh"] = [(1 - d.in_sh) * d[c] for c in ["UE_share", "DD_share", "SD_share"]]
    d["Revenue"] = d.MonthlyOrders * d.AOV
    d["CostToRevenue"] = d.COGSRate + d.OPEXRate
    d["ContributionMargin"] = 1 - d.CostToRevenue
    d["GrossContribution"] = d.Revenue * d.ContributionMargin
    d["UE_CommissionCost"] = d.Revenue * d.ue_sh * d.CommissionRate
    d["AggCommissionCost"] = d.Revenue * (d.ue_sh + d.dd_sh) * d.CommissionRate
    d["SD_DeliveryCost"] = d.MonthlyOrders * d.sd_sh * d.DeliveryCostPerOrder
    d["DeliveryCostRatio"] = d.SD_DeliveryCost / d.Revenue
    d["Comm_x_UEshare"] = d.CommissionRate * d.UE_share
    d["DelCost_x_SDshare"] = d.DeliveryCostPerOrder * d.SD_share
    d["GrowthAdjOrders"] = d.MonthlyOrders * d.GrowthFactor
    return d


def account(df):
    """Deterministic accounting engine (same formulas as the dataset)."""
    d = engineer(df)
    o, cm = d.MonthlyOrders, d.ContributionMargin
    out = pd.DataFrame(index=d.index)
    out["InStore"] = o * d.in_sh * d.AOV * cm
    out["UberEats"] = o * d.ue_sh * d.AOV * (cm - d.CommissionRate)
    out["DoorDash"] = o * d.dd_sh * d.AOV * (cm - d.CommissionRate)
    out["SelfDelivery"] = o * d.sd_sh * d.AOV * cm - o * d.sd_sh * d.DeliveryCostPerOrder
    out["Total"] = out.sum(axis=1)
    return out


def scenario_frame(base, in_sh=None, mix=None, commission=None, radius=None, delivery_cost=None,
                   growth=None, months=0, d_comm=0.0):
    """Apply what-if changes. `mix` = (UE, DD, SD) shares of delivery orders."""
    s = base.copy()
    if in_sh is not None: s["in_sh"] = in_sh
    if mix is not None:
        s["UE_share"], s["DD_share"], s["SD_share"] = mix
    if commission is not None: s["CommissionRate"] = commission
    if d_comm: s["CommissionRate"] = s.CommissionRate + d_comm
    if radius is not None:
        s["DeliveryRadiusKM"] = radius
        s["DeliveryCostPerOrder"] = COST_PER_KM * radius
    if delivery_cost is not None: s["DeliveryCostPerOrder"] = delivery_cost
    if growth is not None: s["GrowthFactor"] = growth
    if months: s["MonthlyOrders"] = s.MonthlyOrders * s.GrowthFactor ** months
    return s


def predict(model, frame):
    return model.predict(engineer(frame)[ENG_NUM + CATS])


def optimise(model, base, max_in_shift=0.10, step=0.05):
    """Grid-search the mix that maximises predicted profit inside observed data ranges
    (UE 35-60%, DD 20-30%, SD 15-45% of delivery orders)."""
    lo, hi = max(0.03, base.in_sh.mean() - max_in_shift), min(0.55, base.in_sh.mean() + max_in_shift)
    ins = np.arange(round(lo / step) * step, hi + 1e-9, step)
    res = []
    for ue, sd in itertools.product(np.arange(0.35, 0.601, step), np.arange(0.15, 0.451, step)):
        dd = round(1 - ue - sd, 4)
        if 0.20 - 1e-9 <= dd <= 0.30 + 1e-9:
            for i in ins:
                s = scenario_frame(base, in_sh=i, mix=(ue, dd, sd))
                res.append((i, ue, dd, sd, predict(model, s).sum()))
    r = pd.DataFrame(res, columns=["InStore", "UE", "DD", "SD", "Profit"]).sort_values("Profit", ascending=False)
    return r.reset_index(drop=True)


def breakeven_commission(df):
    """ZeroProfit: commission where an aggregator order earns 0. SD_Parity: commission where Uber = self-delivery per order."""
    return pd.DataFrame({"ZeroProfit": (1 - df.COGSRate - df.OPEXRate).clip(lower=0),
                         "SD_Parity": df.DeliveryCostPerOrder / df.AOV})


def sensitivity_index(model, base, shock=0.10):
    """Profit Sensitivity Index: % change in total predicted profit per +1% relative shock."""
    p0 = predict(model, base).sum()
    cases = {"Commission rate": scenario_frame(base, commission=base.CommissionRate * (1 + shock)),
             "Delivery cost / order": scenario_frame(base, delivery_cost=base.DeliveryCostPerOrder * (1 + shock)),
             "Uber Eats share": scenario_frame(base, mix=(base.UE_share * (1 + shock), base.DD_share,
                                                           base.SD_share - base.UE_share * shock)),
             "COGS rate": base.assign(COGSRate=base.COGSRate * (1 + shock)),
             "OPEX rate": base.assign(OPEXRate=base.OPEXRate * (1 + shock)),
             "Order volume": base.assign(MonthlyOrders=base.MonthlyOrders * (1 + shock))}
    out = {k: (predict(model, s).sum() - p0) / abs(p0) / shock for k, s in cases.items()}
    return pd.Series(out).sort_values(key=abs, ascending=False)
