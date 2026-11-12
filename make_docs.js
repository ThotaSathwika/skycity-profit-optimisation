const fs = require("fs");
const d = require("docx");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType, LevelFormat, BorderStyle } = d;
const R = JSON.parse(fs.readFileSync("models/results.json"));
const cmp = fs.readFileSync("models/model_comparison.csv", "utf8").trim().split("\n").slice(1).map(l => l.split(","));
const M = x => "$" + (x / 1e6).toFixed(2) + "M", K = x => "$" + Math.round(x / 1e3).toLocaleString() + "K", pc = (x, n = 1) => (x * 100).toFixed(n) + "%";
const W = 9360;
const run = x => typeof x === "string" ? new TextRun({ text: x }) : x;
const p = (t, o = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, alignment: o.align,
  children: [].concat(t).map(x => typeof x === "string" ? new TextRun({ text: x, bold: o.bold, italics: o.it, size: o.size }) : x) });
const h1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bl = t => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60 }, children: [].concat(t).map(run) });
const b = t => new TextRun({ text: t, bold: true });
const img = (f, w, h, cap) => [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 40 },
  children: [new ImageRun({ type: "png", data: fs.readFileSync(`figures/${f}.png`), transformation: { width: w, height: h }, altText: { title: cap, description: cap, name: f } })] }),
  p(cap, { align: AlignmentType.CENTER, it: true, size: 18 })];
const bd = { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" }, bds = { top: bd, bottom: bd, left: bd, right: bd };
function table(head, rows, widths) {
  const cell = (t, i, hd) => new TableCell({ borders: bds, width: { size: widths[i], type: WidthType.DXA }, margins: { top: 50, bottom: 50, left: 80, right: 80 },
    shading: hd ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined, children: [new Paragraph({ children: [new TextRun({ text: String(t), bold: hd, size: 18 })] })] });
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ children: head.map((t, i) => cell(t, i, true)) }), ...rows.map(r => new TableRow({ children: r.map((t, i) => cell(t, i, false)) }))] });
}
const S = R.scenarios, base = S["Baseline"];
const dl = k => `${S[k] >= base ? "+" : "−"}${K(Math.abs(S[k] - base))} (${S[k] >= base ? "+" : "−"}${pc(Math.abs(S[k] - base) / base)})`;
const lossC = base - S["Commission +3pp"];
const styles = { default: { document: { run: { font: "Calibri", size: 22 } } }, paragraphStyles: [
  { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, color: "1F4E79" }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
  { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 25, bold: true, color: "2E75B6" }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } }] };
const numbering = { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] };
const page = { page: { size: { width: 12240, height: 15840 }, margin: { top: 1260, right: 1440, bottom: 1260, left: 1440 } } };

const eng = cmp.filter(r => r[0] === "Engineered features"), raw = cmp.filter(r => r[0] === "Raw features");
const mrow = r => [r[1], Number(r[2]).toLocaleString(undefined, { maximumFractionDigits: 1 }), Number(r[3]).toLocaleString(undefined, { maximumFractionDigits: 1 }), Number(r[4]).toFixed(4), Number(r[5]).toFixed(4)];
const xgbDrop = R.xgb_baseline - R.xgb_ue10;

const paper = [
  p("Predictive Modelling and Profit Optimisation for Multi-Channel Restaurant Operations", { align: AlignmentType.CENTER, bold: true, size: 40 }),
  p("A Case Study of SkyCity Auckland Restaurants & Bars", { align: AlignmentType.CENTER, size: 26 }),
  p("Project for Unified Mentor", { align: AlignmentType.CENTER, it: true }),
  h1("Abstract"),
  p(`Restaurants sell through four channels: in-store, Uber Eats, DoorDash and self-delivery. Each has a different margin structure, yet channel decisions are rarely tested before they are made. This paper builds a predictive and prescriptive framework on 1,696 SkyCity Auckland restaurant records. XGBoost and Gradient Boosting predict monthly net profit with R² of 0.994 on hold-out data using engineered features, and a linear model with the same features reproduces the profit accounting almost exactly (R² = 1.000). A scenario engine and a grid-search optimiser, delivered through a Streamlit dashboard, show that a 3 percentage-point (pp) commission rise removes ${M(lossC)} of monthly profit (−${pc(lossC / base, 0)}), that the median break-even commission is ${pc(R.breakeven_median)} but only ${pc(R.breakeven_by_segment["Full-service"])} for Full-service restaurants, and that the best channel mix inside the observed data range lifts predicted profit by ${pc(R.opt_uplift)}.`),
  p([b("Keywords: "), "profit prediction, channel mix, delivery aggregators, scenario simulation, Streamlit, XGBoost."]),
  h1("1. Introduction"),
  p("Delivery aggregators drive order volume but charge commissions of 27–33% in this data. Self-delivery avoids commission but adds a logistics cost per order. Static reports describe what happened; they cannot answer questions such as what happens if Uber Eats share rises by 10 pp, when self-delivery becomes the better option, or how sensitive profit is to commission and delivery radius. This project provides (i) predictive models linking channel mix and costs to profit, (ii) a sensitivity analysis, (iii) a scenario simulator and (iv) a prescriptive optimiser."),
  h1("2. Data and Exploratory Analysis"),
  p("The dataset has 1,696 restaurants and 30 columns, with no missing values or duplicate IDs. It covers 8 cuisines, 4 segments (Cafe, QSR, Full-service, Ghost Kitchen) and 4 Auckland subregions. Channel profit follows the data's own accounting: in-store = revenue × (1 − COGS − OPEX); aggregator = revenue × (1 − COGS − OPEX − commission); self-delivery = revenue × (1 − COGS − OPEX) − delivery cost."),
  h2("2.1 Data observations"),
  bl("InStoreShare in the file differs from in-store orders ÷ total orders by about 5 pp on average, so the realised share (in_sh) is used. UE_share, DD_share and SD_share are shares of delivery orders and sum to 1."),
  bl("DeliveryCostPerOrder is almost exactly 0.295 × DeliveryRadiusKM (correlation 1.00). Radius has no measurable link with order volume, so radius scenarios change cost only."),
  bl(`Total monthly profit averages $4,638 per restaurant (${M(R.actual_total)} in total). ${pc(R.loss_making_share)} of restaurants (408) lose money, and every one of them is in the Full-service segment, where about 90% of restaurants are loss-making because COGS plus OPEX consume about 87.5% of revenue.`),
  bl("No outliers were found at 3 × IQR on profit per order. The loss-making cases are genuine, so no rows were removed."),
  ...img("eda_profit", 600, 220, "Figure 1. Profit distribution and mean profit by segment."),
  ...img("eda_margin", 400, 240, "Figure 2. Net margin by channel: aggregator margins fall below zero for many restaurants."),
  h1("3. Methodology"),
  h2("3.1 Feature engineering"),
  p("Features built from the decision variables and cost structure: revenue (orders × AOV); contribution margin (1 − COGS − OPEX); commission cost (revenue × aggregator share × commission); self-delivery cost (orders × SD share × cost per order); delivery-cost-to-revenue ratio; the interaction terms CommissionRate × UE_share and DeliveryCostPerOrder × SD_share; and growth-adjusted orders (orders × GrowthFactor). Cuisine, segment and subregion are one-hot encoded."),
  h2("3.2 Models and evaluation"),
  p("Linear Regression (interpretable baseline), Random Forest, Gradient Boosting and XGBoost were trained on an 80/20 split (random_state = 42) with 5-fold cross-validation on the training set. Each model was run on raw decision variables and on the engineered set. Metrics are RMSE, MAE and R². Profit per order and the four channel margins were modelled separately with the linear model (R² 0.990 for profit per order; 1.000 for in-store, Uber Eats and DoorDash margins; 0.999 for self-delivery)."),
  h1("4. Results"),
  h2("4.1 Model performance"),
  table(["Model (engineered features)", "RMSE ($)", "MAE ($)", "R²", "CV R²"], eng.map(mrow), [3360, 1500, 1500, 1500, 1500]),
  p(""),
  table(["Model (raw features only)", "RMSE ($)", "MAE ($)", "R²", "CV R²"], raw.map(mrow), [3360, 1500, 1500, 1500, 1500]),
  p(""),
  p(`Two findings stand out. First, profit is an accounting identity of revenue, margins, commissions and delivery costs, so a linear model with engineered interaction features recovers it almost exactly (RMSE about $2). Second, when only raw decision variables are available, tree ensembles are far better than linear regression (XGBoost R² 0.985 vs 0.892), because profit depends on products of shares, rates and volume. XGBoost is kept in the dashboard as an independent cross-check; its baseline prediction is within 0.2% of the linear model.`),
  ...img("model_rmse", 520, 240, "Figure 3. Test RMSE by model and feature set."),
  ...img("feature_importance", 400, 255, "Figure 4. Top Gradient Boosting feature importances."),
  h2("4.2 Cost sensitivity"),
  p("The Profit Sensitivity Index (PSI) is the percentage change in portfolio profit for a 1% relative increase in a driver:"),
  table(["Driver", "PSI"], Object.entries(R.psi).map(([k, v]) => [k, v.toFixed(2)]), [6000, 3360]),
  p(""),
  p(`Operating and cost-of-goods ratios dominate, but among the levers a restaurant can negotiate or change, commission rate is the most damaging (${R.psi["Commission rate"].toFixed(2)}): each 1% relative rise in commission removes about 1.8% of profit. Each commission point costs about ${K(-R.comm_slope_per_pp)} per month across the portfolio. Delivery cost per order is a minor lever (${R.psi["Delivery cost / order"].toFixed(2)}).`),
  ...img("psi", 400, 220, "Figure 5. Profit Sensitivity Index."),
  ...img("commission_sweep", 600, 215, "Figure 6. Profit and share of Uber Eats loss-making restaurants against commission rate."),
  h2("4.3 Scenario simulation"),
  table(["Scenario", "Monthly profit", "Change"], Object.keys(S).map(k => [k, M(S[k]), k === "Baseline" ? "—" : dl(k)]), [4200, 2200, 2960]),
  p(""),
  p(`Shifting 10 pp of delivery orders from self-delivery to Uber Eats costs ${K(base - S["UE +10pp (from Self-Delivery)"])} a month; the reverse shift gains the same. The XGBoost cross-check gives a smaller loss (${K(xgbDrop)}) because tree models flatten at the edge of the training data, so the linear estimate is the more reliable one for large shifts. A +3 km radius reduces profit by ${K(base - S["Radius +3km"])} (cost only, no demand gain assumed). In the demand-versus-margin test, growth compounding at each restaurant's own factor raises 12-month profit by ${pc(R.growth_12m, 0)}; with commission +3 pp the gain falls to ${pc(R.growth_12m_erode, 0)}, so margin erosion removes more than half of the growth benefit.`),
  ...img("scenarios", 450, 250, "Figure 7. Scenario impact on monthly profit."),
  ...img("growth", 400, 220, "Figure 8. 12-month profit path with and without a 3 pp commission rise."),
  h2("4.4 Break-even commission and safe operating range"),
  p(`An aggregator order earns zero profit when commission equals 1 − COGS − OPEX. The median break-even is ${pc(R.breakeven_median)}, against actual rates of 27–33%. ${pc(R.ue_loss_now)} of restaurants currently lose money on aggregator orders. By segment the median break-even is Cafe ${pc(R.breakeven_by_segment.Cafe)}, QSR ${pc(R.breakeven_by_segment.QSR)}, Ghost Kitchen ${pc(R.breakeven_by_segment["Ghost Kitchen"])} and Full-service ${pc(R.breakeven_by_segment["Full-service"])}. Keeping a 5 pp margin of safety gives recommended commission ceilings of about ${pc(R.safe_comm_by_segment.Cafe, 0)} (Cafe), ${pc(R.safe_comm_by_segment.QSR, 0)} (QSR), ${pc(R.safe_comm_by_segment["Ghost Kitchen"], 0)} (Ghost Kitchen) and ${pc(R.safe_comm_by_segment["Full-service"], 1)} (Full-service). The commission at which Uber Eats equals self-delivery per order is only ${pc(R.sd_parity_median)} (delivery cost ÷ AOV), so self-delivery earns more per order than an aggregator at any radius in the data (3–18 km); the parity radius would be about ${R.sd_threshold_radius_km.toFixed(0)} km.`),
  h2("4.5 Prescriptive optimisation"),
  p(`A grid search over mixes observed in the data (Uber Eats 35–60%, DoorDash 20–30%, self-delivery 15–45% of delivery orders; in-store within ±10 pp of today) finds that the best mix is ${pc(R.opt_best.InStore, 0)} in-store, with delivery split ${pc(R.opt_best.UE, 0)} Uber Eats, ${pc(R.opt_best.DD, 0)} DoorDash and ${pc(R.opt_best.SD, 0)} self-delivery. Predicted profit rises from ${M(R.baseline_total)} to ${M(R.opt_best.Profit)} (${pc(R.opt_uplift)}). Profit per 1 pp of order share (Channel Mix Efficiency Score) is $${R.efficiency_per_pp.InStore} for in-store, $${R.efficiency_per_pp.SelfDelivery} for self-delivery, $${R.efficiency_per_pp.UberEats} for Uber Eats and $${R.efficiency_per_pp.DoorDash} for DoorDash, per restaurant per month. The current mix reaches ${pc(R.mix_efficiency_score, 0)} of the optimum.`),
  table(["Segment", "Baseline profit", "Optimised profit", "Uplift"], Object.entries(R.opt_by_segment).map(([k, v]) => [k, M(v.base_profit), M(v.best_profit), pc(v.uplift, 0)]), [2800, 2200, 2200, 2160]),
  p(""),
  p(`For the ${R.fs_n} Full-service restaurants even the best in-range mix stays negative. Removing aggregator orders altogether (outside the observed range, so an extrapolation) would move them from ${M(R.fs_base)} to ${M(R.fs_no_agg)} a month. For a fixed logistics cost of $2,000 a month, a restaurant needs about ${Math.round(R.inv_threshold_orders_F2000)} self-delivery orders a month to break even; the median today is ${Math.round(R.median_sd_orders)}.`),
  ...img("radius", 400, 220, "Figure 9. Portfolio profit against delivery radius (cost effect only)."),
  h1("5. Recommendations"),
  bl([b("Cap commissions by segment. "), "Negotiate Cafe and QSR rates to about 31% or below, Ghost Kitchens to about 40%, and treat the break-even rate as a hard walk-away point."]),
  bl([b("Review Full-service aggregator listings. "), "At a 12% break-even, aggregator orders lose money in this segment; pilot de-listing or a delivery-only menu with higher prices."]),
  bl([b("Grow self-delivery where volume exists. "), "It earns roughly $8 more per order than Uber Eats; invest where monthly self-delivery orders exceed the break-even threshold."]),
  bl([b("Protect in-store share. "), `Each extra 5 pp of in-store orders adds about ${M(S["In-store +5pp (from delivery)"] - base)} a month in the model.`]),
  bl([b("Use the dashboard before changing channel terms. "), "Test the change under both models and read the risk indicators."]),
  h1("6. Limitations"),
  bl("The data is one cross-section; GrowthFactor is used as a given trend, not forecast from a time series."),
  bl("Scenarios hold order volume fixed when shares change. A real shift away from aggregators may lose customers who find the restaurant only on those apps, and self-delivery needs capacity at the same cost per order."),
  bl("Radius shows no demand effect in the data, so radius expansion is shown as cost only."),
  bl("Linear predictions rely on the accounting identity and extrapolate well; tree models do not extrapolate beyond the training range."),
  h1("7. Conclusion"),
  p(`Profit in this business is driven by cost structure and by how many orders pass through high-commission channels. The framework turns historical records into a forecasting and decision tool: it predicts profit accurately, ranks the cost levers, gives negotiation benchmarks and recommends a mix that is about ${pc(R.opt_uplift, 0)} more profitable within observed ranges. Future work should add time-series demand data, customer-retention effects of channel changes and capacity limits for self-delivery.`),
  h1("Appendix: Reproducibility"),
  p("Code: utils.py (features, accounting engine, optimiser), train_model.py, analysis.py and app.py. Run python train_model.py, python analysis.py, then streamlit run app.py. Random seed 42 throughout."),
];

const ex = [
  p("Executive Summary", { bold: true, size: 40 }),
  p("Profit optimisation for multi-channel restaurant operations · SkyCity Auckland Restaurants & Bars", { it: true }),
  h1("Purpose"),
  p("Restaurants use in-store dining, Uber Eats, DoorDash and their own delivery. This study used 1,696 restaurant records to test how channel mix and delivery costs change profit, so that decisions can be checked before they are made."),
  h1("Key findings"),
  bl(`The restaurants earn about ${M(R.baseline_total)} a month in total. ${pc(R.loss_making_share, 0)} lose money, and all of them are full-service restaurants.`),
  bl(`Commission is the biggest lever the businesses can influence. A rise of 3 percentage points removes about ${M(lossC)} a month (−${pc(lossC / base, 0)}); a fall of 3 points adds about the same.`),
  bl(`On average a restaurant stops making money on an aggregator order at a commission of ${pc(R.breakeven_median, 0)}. For full-service restaurants the figure is only ${pc(R.breakeven_by_segment["Full-service"], 0)}, far below the 27–33% charged.`),
  bl("Self-delivery earns about $8 more per order than Uber Eats because its logistics cost (about $3 per order) is far below the commission (about $11.50). Orders taken in-store earn the most."),
  bl(`Shifting 10 points of delivery orders to Uber Eats costs about ${M(base - S["UE +10pp (from Self-Delivery)"])} a month. The best mix within today's observed ranges raises profit by ${pc(R.opt_uplift, 0)}.`),
  h1("Recommendations"),
  bl("Set commission ceilings by segment (about 31% for cafes and quick-service, 40% for ghost kitchens) and treat break-even as the walk-away point."),
  bl("Review aggregator use for full-service restaurants, where aggregator orders currently lose money."),
  bl(`Expand self-delivery where a restaurant handles about ${Math.round(R.inv_threshold_orders_F2000)} or more delivery orders a month (assuming $2,000 a month in fleet costs).`),
  bl(`Protect in-store dining: each extra 5 points of in-store orders adds roughly ${M(S["In-store +5pp (from delivery)"] - base)} a month.`),
  h1("Caution"),
  p("Results assume order volumes stay the same when channels change. Pilot changes with a small group first, and use the dashboard to test scenarios. The data is a single snapshot and does not show how customers react to leaving a platform."),
  h1("Tools delivered"),
  p("A research paper, a Streamlit dashboard with what-if sliders, risk indicators and an optimisation panel, and the supporting code."),
];
(async () => {
  for (const [n, c] of [["reports/Research_Paper.docx", paper], ["reports/Executive_Summary.docx", ex]])
    fs.writeFileSync(n, await Packer.toBuffer(new Document({ styles, numbering, sections: [{ properties: page, children: c }] })));
  console.log("written");
})();
