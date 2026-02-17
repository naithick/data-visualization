"""Marketing Budget Optimization Engine — Mac M4 optimized.

Runs headless (Agg backend), caches trained model to disk via joblib,
subsamples training for speed. Re-run = instant load from cache.

  pip install -r requirements.txt
  python optimizer_mac.py
"""

import warnings, os, time, hashlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

sns.set_theme(context="talk", style="whitegrid", palette="muted")
PAL = sns.color_palette("muted")
FIG = (14, 7)

# cache paths
# cache paths
CACHE = "../models"
os.makedirs(CACHE, exist_ok=True)
MDL_PATH = os.path.join(CACHE, "gbr_pipeline.joblib")
LE_PATH = os.path.join(CACHE, "label_encoders.joblib")
META_PATH = os.path.join(CACHE, "meta.joblib")

# speed knobs — tune for your M4
TRAIN_CAP = 50_000   # subsample rows for training (None = all)
CV_K = 3

# ---------------------------------------------------------------------------
# 0. Load
# ---------------------------------------------------------------------------
print("=" * 60)
print("  MARKETING BUDGET OPTIMIZER  (Mac M4)")
print("=" * 60)

DATA = "../data/raw/marketing_campaign_dataset.csv"
raw = pd.read_csv(DATA)
print(f"\nLoaded {raw.shape[0]:,} rows x {raw.shape[1]} cols")

# ---------------------------------------------------------------------------
# 1. Clean
# ---------------------------------------------------------------------------
print("\n-- Step 1: Data Cleaning")

df = raw.copy()

df["Acquisition_Cost"] = (
    df["Acquisition_Cost"].astype(str)
    .str.replace(r"[^\d.]", "", regex=True)
    .replace("", np.nan).astype(float)
)

df["Duration"] = (
    df["Duration"].astype(str)
    .str.replace(r"[^\d.]", "", regex=True)
    .replace("", np.nan).astype(float)
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Quarter"] = df["Date"].dt.quarter
df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

n0 = len(df)
df = df[(df["Acquisition_Cost"] > 0) & (df["Duration"] > 0)].copy()
df.dropna(subset=["ROI", "Conversion_Rate", "Clicks", "Impressions"], inplace=True)
print(f"  Dropped {n0 - len(df):,} bad rows -> {len(df):,} remain")

# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------
print("\n-- Step 2: Feature Engineering")

df["CPA"] = df["Acquisition_Cost"] / (df["Clicks"] * df["Conversion_Rate"]).replace(0, np.nan)
df["ROAS"] = df["ROI"] + 1
df["Engagement_Yield"] = df["Engagement_Score"] / df["Acquisition_Cost"]
df["Interaction_Velocity"] = df["Clicks"] / df["Duration"]
df["CTR"] = df["Clicks"] / df["Impressions"].replace(0, np.nan)
df["Cost_Per_Impression"] = df["Acquisition_Cost"] / df["Impressions"].replace(0, np.nan)
df["log_Acquisition_Cost"] = np.log1p(df["Acquisition_Cost"])
df["log_Impressions"] = np.log1p(df["Impressions"])

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

eng_cols = [
    "CPA", "ROAS", "Engagement_Yield", "Interaction_Velocity",
    "CTR", "Cost_Per_Impression", "log_Acquisition_Cost", "log_Impressions",
]
print(f"  {len(eng_cols)} features created")

summary = df[["ROI", "CPA", "ROAS", "Engagement_Yield", "Interaction_Velocity"]].describe().T
print(summary[["mean", "50%", "std"]].round(3).to_string())

# ---------------------------------------------------------------------------
# 3. K-Means Segmentation
# ---------------------------------------------------------------------------
print("\n-- Step 3: Segmentation (K=4)")

clust_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("km", KMeans(n_clusters=4, n_init=10, random_state=42)),
])
df["Cluster"] = clust_pipe.fit_predict(df[["ROI", "Conversion_Rate", "Acquisition_Cost"]])

cstats = df.groupby("Cluster")[["ROI", "Acquisition_Cost"]].mean()
roi_m, cost_m = cstats["ROI"].median(), cstats["Acquisition_Cost"].median()

name_map, used = {}, set()
for idx, r in cstats.iterrows():
    hi_roi = r["ROI"] >= roi_m
    hi_cost = r["Acquisition_Cost"] >= cost_m
    if hi_roi and hi_cost:
        n = "Whales"
    elif not hi_roi and hi_cost:
        n = "Money Pits"
    elif hi_roi and not hi_cost:
        n = "Hidden Gems"
    else:
        n = "Standard"
    if n in used:
        n = f"Cluster_{idx}"
    name_map[idx] = n
    used.add(n)

df["Segment"] = df["Cluster"].map(name_map)

seg_summary = (
    df.groupby("Segment")
    .agg(Count=("ROI", "size"), Avg_ROI=("ROI", "mean"),
         Avg_Cost=("Acquisition_Cost", "mean"),
         Avg_ConvRate=("Conversion_Rate", "mean"), Avg_ROAS=("ROAS", "mean"))
    .round(3)
)
print(seg_summary.to_string())

# ---------------------------------------------------------------------------
# 4. GBR Model (with caching)
# ---------------------------------------------------------------------------
print("\n-- Step 4: GBR Model")

cat_cols = ["Campaign_Type", "Channel_Used", "Target_Audience",
            "Customer_Segment", "Location", "Language"]
le_dict = {}
for c in cat_cols:
    le = LabelEncoder()
    df[f"{c}_enc"] = le.fit_transform(df[c].astype(str))
    le_dict[c] = le

feature_cols = [
    "Acquisition_Cost", "Duration", "Clicks", "Impressions",
    "Engagement_Score", "Conversion_Rate",
    "CPA", "ROAS", "Engagement_Yield", "Interaction_Velocity",
    "CTR", "Cost_Per_Impression",
    "log_Acquisition_Cost", "log_Impressions",
    "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos",
] + [f"{c}_enc" for c in cat_cols]

X = df[feature_cols]
y = df["ROI"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fingerprint the data so we know when to retrain
data_hash = hashlib.md5(
    pd.util.hash_pandas_object(df[feature_cols].head(500)).values.tobytes()
).hexdigest()[:12]

cache_valid = (
    os.path.exists(MDL_PATH)
    and os.path.exists(META_PATH)
    and joblib.load(META_PATH).get("hash") == data_hash
)

if cache_valid:
    print("  Loading cached model (skip training)")
    model_pipe = joblib.load(MDL_PATH)
    le_dict = joblib.load(LE_PATH)
    meta = joblib.load(META_PATH)
    feature_cols = meta["features"]
    cat_cols = meta["cats"]

    y_pred = model_pipe.predict(X_test)
    print(f"  MAE = {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"  R2  = {r2_score(y_test, y_pred):.4f}")
else:
    # subsample for speed
    if TRAIN_CAP and len(X_train) > TRAIN_CAP:
        ix = np.random.RandomState(42).choice(len(X_train), TRAIN_CAP, replace=False)
        Xf, yf = X_train.iloc[ix], y_train.iloc[ix]
        print(f"  Subsampled to {TRAIN_CAP:,} rows")
    else:
        Xf, yf = X_train, y_train

    model_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42)),
    ])

    t0 = time.time()
    print("  Training (300 trees) ...")
    model_pipe.fit(Xf, yf)
    print(f"  Done in {time.time()-t0:.1f}s")

    y_pred = model_pipe.predict(X_test)
    print(f"  MAE = {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"  R2  = {r2_score(y_test, y_pred):.4f}")

    cv = cross_val_score(model_pipe, Xf, yf, cv=CV_K, scoring="r2", n_jobs=-1)
    print(f"  {CV_K}-fold CV R2 = {cv.mean():.4f} +/- {cv.std():.4f}")

    # save
    joblib.dump(model_pipe, MDL_PATH, compress=3)
    joblib.dump(le_dict, LE_PATH, compress=3)
    joblib.dump({"features": feature_cols, "cats": cat_cols, "hash": data_hash}, META_PATH)
    print(f"  Saved -> {MDL_PATH} ({os.path.getsize(MDL_PATH)/1e6:.1f} MB)")

# importance
gbr = model_pipe.named_steps["gbr"]
imp = pd.Series(gbr.feature_importances_, index=feature_cols)
top10 = imp.sort_values(ascending=False).head(10)
print("\n  Top 10 ROI drivers:")
for i, (f, v) in enumerate(top10.items(), 1):
    print(f"    {i:2d}. {f:30s} {v:.4f}")

# ---------------------------------------------------------------------------
# 5. Visualizations (saved to disk, no GUI)
# ---------------------------------------------------------------------------
print("\n-- Step 5: Saving charts")

seg_pal = {"Whales": "#2ecc71", "Hidden Gems": "#3498db",
           "Money Pits": "#e74c3c", "Standard": "#95a5a6"}
for s in df["Segment"].unique():
    seg_pal.setdefault(s, "#bdc3c7")

sample = df.sample(min(10_000, len(df)), random_state=42)


def save(fig, name):
    out_path = os.path.join("../results/figures", name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# 1 importance
fig, ax = plt.subplots(figsize=FIG)
top10.sort_values().plot.barh(ax=ax, color=PAL[0], edgecolor="black")
ax.set_xlabel("Importance"); ax.set_title("Top 10 ROI Drivers")
plt.tight_layout(); save(fig, "fig01_importance.png")

# 2 segments
fig, ax = plt.subplots(figsize=FIG)
sns.scatterplot(data=sample, x="Acquisition_Cost", y="ROI",
                hue="Segment", palette=seg_pal, alpha=.5, s=25, ax=ax)
ax.set_title("Segments: ROI vs Cost")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.tight_layout(); save(fig, "fig02_segments.png")

# 3 saturation
fig, ax = plt.subplots(figsize=FIG)
ax.scatter(sample["Acquisition_Cost"], sample["ROI"], alpha=.12, s=8, c=PAL[0])
xv = sample["Acquisition_Cost"].values.reshape(-1, 1)
poly = Pipeline([("p", PolynomialFeatures(3, include_bias=False)), ("lr", LinearRegression())])
poly.fit(xv, sample["ROI"].values)
xr = np.linspace(xv.min(), xv.max(), 300).reshape(-1, 1)
yr = poly.predict(xr)
ax.plot(xr, yr, color="#e74c3c", lw=3, label="Poly-3 trend")
peak = np.argmax(yr)
ax.axvline(xr[peak], color="#f39c12", ls="--", lw=2,
           label=f"Dim. returns ~ ${xr[peak][0]:,.0f}")
ax.scatter([xr[peak]], [yr[peak]], color="#f39c12", s=200, marker="*",
           edgecolors="black", zorder=5)
ax.set_title("Spend Saturation"); ax.set_xlabel("Cost ($)"); ax.set_ylabel("ROI")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(); plt.tight_layout(); save(fig, "fig03_saturation.png")

# 4 channel boxplot
fig, ax = plt.subplots(figsize=FIG)
co = df.groupby("Channel_Used")["ROI"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="Channel_Used", y="ROI", order=co,
            palette="viridis", ax=ax, showfliers=False)
ax.set_title("ROI by Channel"); plt.tight_layout(); save(fig, "fig04_channel.png")

# 5 ROI heatmap
fig, ax = plt.subplots(figsize=FIG)
pvt = df.pivot_table("ROI", index="Campaign_Type", columns="Target_Audience", aggfunc="mean")
sns.heatmap(pvt, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=.5, ax=ax)
ax.set_title("ROI: Type x Audience"); plt.tight_layout(); save(fig, "fig05_roi_heatmap.png")

# 6 segment profiles
fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 7))
sm = seg_summary[["Avg_ROI", "Avg_ConvRate"]].reset_index().melt(
    id_vars="Segment", var_name="Metric", value_name="Value")
sns.barplot(data=sm, x="Segment", y="Value", hue="Metric", palette="Set2", ax=a1)
a1.set_title("Segment Comparison")
sc = df["Segment"].value_counts()
a2.pie(sc, labels=sc.index, autopct="%1.1f%%",
       colors=[seg_pal.get(s, "#bdc3c7") for s in sc.index],
       startangle=140, textprops={"fontsize": 12})
a2.set_title("Segment Split"); plt.tight_layout(); save(fig, "fig06_seg_profiles.png")

# 7 monthly trend
monthly = (df.groupby(df["Date"].dt.to_period("M"))
           .agg(roi=("ROI", "mean"), spend=("Acquisition_Cost", "sum"))
           .reset_index())
monthly["Date"] = monthly["Date"].dt.to_timestamp()
fig, a1 = plt.subplots(figsize=FIG); a2 = a1.twinx()
a1.plot(monthly["Date"], monthly["roi"], color=PAL[0], marker="o", lw=2, label="ROI")
a2.bar(monthly["Date"], monthly["spend"], alpha=.25, color=PAL[1], width=20, label="Spend")
a1.set_ylabel("Avg ROI"); a2.set_ylabel("Total Spend")
a2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
a1.set_title("Monthly ROI vs Spend")
h1, l1 = a1.get_legend_handles_labels()
h2, l2 = a2.get_legend_handles_labels()
a1.legend(h1+h2, l1+l2); plt.tight_layout(); save(fig, "fig07_monthly.png")

# 8 correlation
cc = ["ROI", "CPA", "ROAS", "Engagement_Yield", "Interaction_Velocity",
      "CTR", "Cost_Per_Impression", "Acquisition_Cost", "Clicks",
      "Impressions", "Engagement_Score", "Conversion_Rate", "Duration"]
fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones((len(cc), len(cc)), dtype=bool))
sns.heatmap(df[cc].corr(), mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=.5, ax=ax)
ax.set_title("Feature Correlations"); plt.tight_layout(); save(fig, "fig08_corr.png")

# 9a duration
fig, ax = plt.subplots(figsize=FIG)
ds = df.groupby("Duration").agg(roi=("ROI", "mean"), cr=("Conversion_Rate", "mean")).reset_index().sort_values("Duration")
ax2 = ax.twinx()
ax.bar(ds["Duration"].astype(str), ds["roi"], color=PAL[0], alpha=.75, edgecolor="black", label="ROI")
ax2.plot(ds["Duration"].astype(str), ds["cr"], color="#e74c3c", marker="D", lw=2.5, ms=9, label="Conv Rate")
ax.set_xlabel("Duration (days)"); ax.set_ylabel("ROI"); ax2.set_ylabel("Conv Rate")
ax.set_title("Duration Effectiveness")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2); plt.tight_layout(); save(fig, "fig09a_duration.png")

# 9b top companies
fig, ax = plt.subplots(figsize=FIG)
cr = (df.groupby("Company")["ROI"].agg(["mean", "count"])
      .rename(columns={"mean": "roi", "count": "n"})
      .sort_values("roi", ascending=False).head(15))
ax.barh(cr.index[::-1], cr["roi"][::-1],
        color=sns.color_palette("YlGnBu_r", 15)[::-1], edgecolor="black", lw=.6)
for i, (v, n) in enumerate(zip(cr["roi"][::-1], cr["n"][::-1])):
    ax.text(v+.03, i, f"{v:.2f} (n={n})", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Avg ROI"); ax.set_title("Top 15 Companies")
plt.tight_layout(); save(fig, "fig09b_companies.png")

# 9c engagement density
fig, ax = plt.subplots(figsize=FIG)
sk = df.sample(min(15_000, len(df)), random_state=42)
sns.kdeplot(x=sk["Engagement_Score"], y=sk["ROI"], cmap="YlGnBu",
            fill=True, levels=15, thresh=.05, ax=ax)
sns.scatterplot(x=sk["Engagement_Score"], y=sk["ROI"], alpha=.06,
                s=4, color="black", ax=ax, legend=False)
ax.set_title("Engagement vs ROI"); plt.tight_layout(); save(fig, "fig09c_engage.png")

# 9d efficiency frontier
fig, ax = plt.subplots(figsize=FIG)
se = df.groupby("Customer_Segment").agg(
    roi=("ROI", "mean"), cost=("Acquisition_Cost", "mean"), n=("ROI", "size")).reset_index()
sc = ax.scatter(se["cost"], se["roi"], s=se["n"]/se["n"].max()*800+100,
                c=se["roi"], cmap="RdYlGn", edgecolors="black", lw=1.2, zorder=3)
for _, r in se.iterrows():
    ax.annotate(r["Customer_Segment"], (r["cost"], r["roi"]),
                xytext=(10, 8), textcoords="offset points", fontsize=11, fontweight="bold")
ax.set_xlabel("Avg Cost"); ax.set_ylabel("Avg ROI")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.set_title("Efficiency Frontier (bubble = count)")
plt.colorbar(sc, ax=ax, label="ROI", shrink=.8)
plt.tight_layout(); save(fig, "fig09d_frontier.png")

# 9e conv rate heatmap
fig, ax = plt.subplots(figsize=FIG)
pc = df.pivot_table("Conversion_Rate", index="Channel_Used", columns="Target_Audience", aggfunc="mean")
sns.heatmap(pc, annot=True, fmt=".3f", cmap="BuPu", linewidths=.8, ax=ax)
ax.set_title("Conv Rate: Channel x Audience")
plt.tight_layout(); save(fig, "fig09e_conv.png")

plt.close("all")

# ---------------------------------------------------------------------------
# 6. Insights
# ---------------------------------------------------------------------------
print("\n-- Step 6: Insights")

best_ch = df.groupby("Channel_Used")["ROI"].mean()
best_tp = df.groupby("Campaign_Type")["ROI"].mean()
best_au = df.groupby("Target_Audience")["ROI"].mean()
best_cs = df.groupby("Customer_Segment")["ROI"].mean()

topq = df[df["ROI"] >= df["ROI"].quantile(.75)]
slo, shi = topq["Acquisition_Cost"].quantile(.25), topq["Acquisition_Cost"].quantile(.75)

hg = df[df["Segment"] == "Hidden Gems"] if "Hidden Gems" in df["Segment"].values else df.head(0)
mp = df[df["Segment"] == "Money Pits"] if "Money Pits" in df["Segment"].values else df.head(0)

print(f"""
  CHANNEL     : {best_ch.idxmax()} (ROI {best_ch.max():.2f})
  CAMPAIGN    : {best_tp.idxmax()} (ROI {best_tp.max():.2f})
  AUDIENCE    : {best_au.idxmax()} (ROI {best_au.max():.2f})
  SEGMENT     : {best_cs.idxmax()}
  SWEET SPOT  : ${slo:,.0f} - ${shi:,.0f}
  DIM RETURNS : ~${xr[peak][0]:,.0f}
  HIDDEN GEMS : {len(hg):,} (scale up)
  MONEY PITS  : {len(mp):,} (audit/kill)
  TOP DRIVERS : {', '.join(top10.index[:5])}
""")

out = (["Campaign_ID", "Company", "Campaign_Type", "Channel_Used",
        "Target_Audience", "Customer_Segment", "Location",
        "Duration", "Acquisition_Cost", "Clicks", "Impressions",
        "Conversion_Rate", "ROI", "Engagement_Score"] + eng_cols + ["Segment"])
df[out].to_csv("../data/processed/enriched_campaigns.csv", index=False)
seg_summary.to_csv("../data/processed/segment_summary.csv")
print("  Exported CSVs")

# ---------------------------------------------------------------------------
# 7. Campaign Simulator
# ---------------------------------------------------------------------------
print("\n-- Step 7: Simulator")


def _safe_encode(le, val):
    try:
        return le.transform([val])[0]
    except ValueError:
        return le.transform([pd.Series(le.classes_).mode()[0]])[0]


def predict_optimal_strategy(budget, audience, duration=30):
    channels = list(le_dict["Channel_Used"].classes_)
    types = list(le_dict["Campaign_Type"].classes_)

    grid = pd.DataFrame([{"Channel_Used": ch, "Campaign_Type": ct}
                         for ch in channels for ct in types])

    med = df[["Clicks", "Impressions", "Engagement_Score", "Conversion_Rate"]].median()
    grid["Acquisition_Cost"] = budget
    grid["Duration"] = duration
    grid["Clicks"] = med["Clicks"]
    grid["Impressions"] = med["Impressions"]
    grid["Engagement_Score"] = med["Engagement_Score"]
    grid["Conversion_Rate"] = med["Conversion_Rate"]
    grid["Target_Audience"] = audience
    grid["Customer_Segment"] = df["Customer_Segment"].mode()[0]
    grid["Location"] = df["Location"].mode()[0]
    grid["Language"] = df["Language"].mode()[0]

    denom = (grid["Clicks"] * grid["Conversion_Rate"]).replace(0, np.nan)
    grid["CPA"] = grid["Acquisition_Cost"] / denom
    grid["ROAS"] = 0
    grid["Engagement_Yield"] = grid["Engagement_Score"] / grid["Acquisition_Cost"]
    grid["Interaction_Velocity"] = grid["Clicks"] / grid["Duration"]
    grid["CTR"] = grid["Clicks"] / grid["Impressions"].replace(0, np.nan)
    grid["Cost_Per_Impression"] = grid["Acquisition_Cost"] / grid["Impressions"].replace(0, np.nan)
    grid["log_Acquisition_Cost"] = np.log1p(grid["Acquisition_Cost"])
    grid["log_Impressions"] = np.log1p(grid["Impressions"])

    now = pd.Timestamp.now()
    grid["Month_sin"] = np.sin(2 * np.pi * now.month / 12)
    grid["Month_cos"] = np.cos(2 * np.pi * now.month / 12)
    grid["DayOfWeek_sin"] = np.sin(2 * np.pi * now.dayofweek / 7)
    grid["DayOfWeek_cos"] = np.cos(2 * np.pi * now.dayofweek / 7)

    for c in cat_cols:
        grid[f"{c}_enc"] = grid[c].apply(lambda v, _le=le_dict[c]: _safe_encode(_le, str(v)))

    grid.replace([np.inf, -np.inf], np.nan, inplace=True)
    grid.fillna(df[feature_cols].median(), inplace=True)

    grid["Predicted_ROI"] = model_pipe.predict(grid[feature_cols])
    grid.sort_values("Predicted_ROI", ascending=False, inplace=True)

    best = grid.iloc[0]
    avg_rest = grid.iloc[1:]["Predicted_ROI"].mean()
    gain = best["Predicted_ROI"] - avg_rest

    # heatmap
    pvt = grid.pivot_table("Predicted_ROI", index="Campaign_Type", columns="Channel_Used")
    fig, ax = plt.subplots(figsize=FIG)
    sns.heatmap(pvt, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=1.2, linecolor="white",
                cbar_kws={"label": "Predicted ROI"}, ax=ax)
    ax.set_title(f"ROI Matrix | ${budget:,.0f} | {audience} | {duration}d",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(); save(fig, f"sim_{budget}_{audience.replace(' ','_')}.png")

    # top-5
    t5 = grid.head(5).copy()
    t5["Label"] = t5["Campaign_Type"] + " x " + t5["Channel_Used"]
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(t5["Label"][::-1], t5["Predicted_ROI"][::-1],
                   color=sns.color_palette("YlGnBu_r", 5), edgecolor="black", lw=.7)
    for b, v in zip(bars, t5["Predicted_ROI"][::-1]):
        ax.text(b.get_width()+.03, b.get_y()+b.get_height()/2,
                f"{v:.2f}", va="center", fontsize=12, fontweight="bold")
    ax.axvline(avg_rest, color="#e74c3c", ls="--", lw=2, label=f"Avg = {avg_rest:.2f}")
    ax.set_title("Top 5 Strategies", fontweight="bold")
    ax.legend(); plt.tight_layout()
    save(fig, f"sim_top5_{budget}.png")

    print(f"  BEST: {best['Campaign_Type']} x {best['Channel_Used']}"
          f" -> ROI {best['Predicted_ROI']:.2f}"
          f" (+{gain:.2f}, {gain/max(avg_rest,.01)*100:.1f}%)")

    ranked = grid[["Campaign_Type", "Channel_Used", "Predicted_ROI"]].reset_index(drop=True)
    ranked.index += 1; ranked.index.name = "Rank"
    print(ranked.to_string())
    return grid


scenarios = [
    (10_000, "Men 25-34", 30),
    (5_000, "Women 35-44", 45),
    (20_000, "All Ages", 60),
]

results = {}
for bud, aud, dur in scenarios:
    tag = f"${bud/1e3:.0f}K/{aud}/{dur}d"
    print(f"\n{'='*60}\n  {tag}\n{'='*60}")
    results[tag] = predict_optimal_strategy(bud, aud, dur)

# cross-scenario
fig, ax = plt.subplots(figsize=(14, 6))
xp = np.arange(len(results))
rois = [g.iloc[0]["Predicted_ROI"] for g in results.values()]
lbls = [f"{g.iloc[0]['Campaign_Type']}\n{g.iloc[0]['Channel_Used']}" for g in results.values()]
bars = ax.bar(xp, rois, color=sns.color_palette("YlGnBu", len(results)),
              edgecolor="black", lw=.8, width=.55)
for b, v, l in zip(bars, rois, lbls):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+.05,
            f"ROI {v:.2f}\n{l}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(xp); ax.set_xticklabels(list(results.keys()), fontsize=12)
ax.set_ylabel("Best ROI"); ax.set_title("Cross-Scenario Comparison", fontweight="bold")
plt.tight_layout(); save(fig, "fig_scenarios.png")

# ---------------------------------------------------------------------------
# 8. Interactive Campaign Advisor
# ---------------------------------------------------------------------------
print("\n-- Step 8: Campaign Advisor")


def campaign_advisor(query):
    """Input a company name, customer segment, or target audience.
    Returns historical analysis + AI-predicted optimal campaign strategy."""

    query = query.strip()
    match_col, match_val = None, None

    for col, vals in [
        ("Company", df["Company"].unique()),
        ("Customer_Segment", df["Customer_Segment"].unique()),
        ("Target_Audience", df["Target_Audience"].unique()),
        ("Location", df["Location"].unique()),
        ("Channel_Used", df["Channel_Used"].unique()),
        ("Campaign_Type", df["Campaign_Type"].unique()),
    ]:
        for v in vals:
            if query.lower() in v.lower() or v.lower() in query.lower():
                match_col, match_val = col, v
                break
        if match_col:
            break

    if not match_col:
        print(f"\n  No match for '{query}'.")
        print(f"    Companies : {', '.join(df['Company'].unique())}")
        print(f"    Segments  : {', '.join(df['Customer_Segment'].unique())}")
        print(f"    Audiences : {', '.join(df['Target_Audience'].unique())}")
        return None

    print(f"\n  Matched: {match_col} = '{match_val}'")
    subset = df[df[match_col] == match_val]
    dataset_avg_roi = df["ROI"].mean()

    print(f"\n  --- Historical ({len(subset):,} campaigns) ---")
    print(f"  Avg ROI        : {subset['ROI'].mean():.3f}  (dataset avg: {dataset_avg_roi:.3f})")
    print(f"  Avg Conv Rate  : {subset['Conversion_Rate'].mean():.4f}")
    print(f"  Avg Cost       : ${subset['Acquisition_Cost'].mean():,.0f}")
    print(f"  Best Duration  : {subset.groupby('Duration')['ROI'].mean().idxmax():.0f} days")

    combo = (subset.groupby(["Campaign_Type", "Channel_Used"])["ROI"]
             .agg(["mean", "count"])
             .rename(columns={"mean": "avg_roi", "count": "n"})
             .sort_values("avg_roi", ascending=False))
    combo = combo[combo["n"] >= 5]
    print(f"\n  --- Best Historical Combos ---")
    for i, (idx, row) in enumerate(combo.head(5).iterrows()):
        print(f"    {i+1}. {idx[0]:15s} x {idx[1]:12s}  ROI={row['avg_roi']:.3f}  (n={row['n']:.0f})")

    # AI prediction
    med_budget = subset["Acquisition_Cost"].median()
    best_dur = int(subset.groupby("Duration")["ROI"].mean().idxmax())
    aud = match_val if match_col == "Target_Audience" else (
        subset["Target_Audience"].mode()[0] if len(subset) > 0 else df["Target_Audience"].mode()[0])

    channels = list(le_dict["Channel_Used"].classes_)
    types = list(le_dict["Campaign_Type"].classes_)
    grid = pd.DataFrame([{"Channel_Used": ch, "Campaign_Type": ct}
                         for ch in channels for ct in types])

    med = df[["Clicks", "Impressions", "Engagement_Score", "Conversion_Rate"]].median()
    grid["Acquisition_Cost"] = med_budget
    grid["Duration"] = best_dur
    grid["Clicks"] = med["Clicks"]
    grid["Impressions"] = med["Impressions"]
    grid["Engagement_Score"] = med["Engagement_Score"]
    grid["Conversion_Rate"] = med["Conversion_Rate"]
    grid["Target_Audience"] = aud
    grid["Customer_Segment"] = subset["Customer_Segment"].mode()[0] if len(subset) > 0 else df["Customer_Segment"].mode()[0]
    grid["Location"] = subset["Location"].mode()[0] if len(subset) > 0 else df["Location"].mode()[0]
    grid["Language"] = subset["Language"].mode()[0] if len(subset) > 0 else df["Language"].mode()[0]

    denom = (grid["Clicks"] * grid["Conversion_Rate"]).replace(0, np.nan)
    grid["CPA"] = grid["Acquisition_Cost"] / denom
    grid["ROAS"] = 0
    grid["Engagement_Yield"] = grid["Engagement_Score"] / grid["Acquisition_Cost"]
    grid["Interaction_Velocity"] = grid["Clicks"] / grid["Duration"]
    grid["CTR"] = grid["Clicks"] / grid["Impressions"].replace(0, np.nan)
    grid["Cost_Per_Impression"] = grid["Acquisition_Cost"] / grid["Impressions"].replace(0, np.nan)
    grid["log_Acquisition_Cost"] = np.log1p(grid["Acquisition_Cost"])
    grid["log_Impressions"] = np.log1p(grid["Impressions"])
    now = pd.Timestamp.now()
    grid["Month_sin"] = np.sin(2 * np.pi * now.month / 12)
    grid["Month_cos"] = np.cos(2 * np.pi * now.month / 12)
    grid["DayOfWeek_sin"] = np.sin(2 * np.pi * now.dayofweek / 7)
    grid["DayOfWeek_cos"] = np.cos(2 * np.pi * now.dayofweek / 7)
    for c in cat_cols:
        grid[f"{c}_enc"] = grid[c].apply(lambda v, _le=le_dict[c]: _safe_encode(_le, str(v)))
    grid.replace([np.inf, -np.inf], np.nan, inplace=True)
    grid.fillna(df[feature_cols].median(), inplace=True)

    grid["Predicted_ROI"] = model_pipe.predict(grid[feature_cols])
    grid.sort_values("Predicted_ROI", ascending=False, inplace=True)
    best = grid.iloc[0]

    print(f"\n  --- AI Recommendation ---")
    print(f"    Type       : {best['Campaign_Type']}")
    print(f"    Channel    : {best['Channel_Used']}")
    print(f"    Budget     : ${med_budget:,.0f}")
    print(f"    Duration   : {best_dur} days")
    print(f"    Pred. ROI  : {best['Predicted_ROI']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    pvt = grid.pivot_table("Predicted_ROI", index="Campaign_Type", columns="Channel_Used")
    sns.heatmap(pvt, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=1, ax=axes[0])
    axes[0].set_title(f"Predicted ROI\n{match_col}: {match_val}", fontsize=13)

    t5 = grid.head(5).copy()
    t5["Label"] = t5["Campaign_Type"] + "\n" + t5["Channel_Used"]
    axes[1].barh(t5["Label"][::-1], t5["Predicted_ROI"][::-1],
                 color=sns.color_palette("YlGnBu_r", 5), edgecolor="black")
    for i, v in enumerate(t5["Predicted_ROI"][::-1]):
        axes[1].text(v + 0.02, i, f"{v:.2f}", va="center", fontweight="bold")
    axes[1].axvline(dataset_avg_roi, color="#e74c3c", ls="--", lw=2,
                    label=f"Avg = {dataset_avg_roi:.2f}")
    axes[1].set_title("Top 5 Strategies", fontsize=13); axes[1].legend()

    ch_roi = subset.groupby("Channel_Used")["ROI"].mean().sort_values(ascending=True)
    ch_roi.plot.barh(ax=axes[2], color=PAL[2], edgecolor="black")
    axes[2].axvline(dataset_avg_roi, color="#e74c3c", ls="--", lw=2, label="Dataset avg")
    axes[2].set_title(f"Historical ROI\n{match_val}", fontsize=13); axes[2].legend()

    plt.suptitle(f"Campaign Advisor: {match_val}", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, f"advisor_{match_val.replace(' ', '_')}.png")
    return grid


for q in ["TechCorp", "Fashionistas", "Women 35-44"]:
    print(f"\n{'='*60}\n  QUERY: '{q}'\n{'='*60}")
    campaign_advisor(q)


# ---------------------------------------------------------------------------
# 9. Anomaly Detection
# ---------------------------------------------------------------------------
print("\n-- Step 9: Anomaly Detection")

df["Predicted_ROI"] = model_pipe.predict(df[feature_cols])
df["ROI_Gap"] = df["ROI"] - df["Predicted_ROI"]
df["ROI_Gap_Pct"] = (df["ROI_Gap"] / df["Predicted_ROI"].replace(0, np.nan)) * 100
gap_mean = df["ROI_Gap"].mean()
gap_std = df["ROI_Gap"].std()
df["Gap_Z"] = (df["ROI_Gap"] - gap_mean) / gap_std

df["Anomaly"] = "Normal"
df.loc[df["Gap_Z"] > 2, "Anomaly"] = "Overperformer"
df.loc[df["Gap_Z"] < -2, "Anomaly"] = "Underperformer"

anom_counts = df["Anomaly"].value_counts()
print(f"  Normal         : {anom_counts.get('Normal', 0):,}")
print(f"  Overperformers : {anom_counts.get('Overperformer', 0):,}")
print(f"  Underperformers: {anom_counts.get('Underperformer', 0):,}")

over = df[df["Anomaly"] == "Overperformer"].nlargest(10, "ROI_Gap")
under = df[df["Anomaly"] == "Underperformer"].nsmallest(10, "ROI_Gap")
print("\n  Top Overperformers:")
for _, r in over.iterrows():
    print(f"    {r['Company']:22s} | {r['Campaign_Type']:12s} | ROI={r['ROI']:.2f} vs pred={r['Predicted_ROI']:.2f}")
print("\n  Top Underperformers:")
for _, r in under.iterrows():
    print(f"    {r['Company']:22s} | {r['Campaign_Type']:12s} | ROI={r['ROI']:.2f} vs pred={r['Predicted_ROI']:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
anom_pal = {"Normal": "#95a5a6", "Overperformer": "#2ecc71", "Underperformer": "#e74c3c"}
anom_sample = df.sample(min(15_000, len(df)), random_state=42)
for label, color in anom_pal.items():
    mask = anom_sample["Anomaly"] == label
    axes[0].scatter(anom_sample.loc[mask, "Predicted_ROI"],
                    anom_sample.loc[mask, "ROI"], c=color, s=8, alpha=0.4, label=label)
lims = [df["ROI"].min() - 0.5, df["ROI"].max() + 0.5]
axes[0].plot(lims, lims, "k--", lw=1.5, alpha=0.5, label="Perfect")
axes[0].set_xlabel("Predicted ROI"); axes[0].set_ylabel("Actual ROI")
axes[0].set_title("Predicted vs Actual"); axes[0].legend(fontsize=10)

axes[1].hist(df["ROI_Gap"], bins=80, color=PAL[0], edgecolor="black", alpha=0.7)
axes[1].axvline(gap_mean + 2*gap_std, color="#2ecc71", ls="--", lw=2, label="Over thresh")
axes[1].axvline(gap_mean - 2*gap_std, color="#e74c3c", ls="--", lw=2, label="Under thresh")
axes[1].set_xlabel("ROI Gap"); axes[1].set_title("Gap Distribution"); axes[1].legend(fontsize=10)

anom_rate = (df.groupby("Company")["Anomaly"]
             .apply(lambda x: (x != "Normal").mean() * 100).sort_values(ascending=True))
anom_rate.plot.barh(ax=axes[2], color=PAL[3], edgecolor="black")
axes[2].set_xlabel("Anomaly Rate (%)"); axes[2].set_title("Anomaly Rate by Company")
for i, v in enumerate(anom_rate):
    axes[2].text(v + 0.1, i, f"{v:.1f}%", va="center", fontweight="bold")

plt.suptitle("Campaign Anomaly Detection", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout(); save(fig, "fig10_anomalies.png")

anomaly_df = df[df["Anomaly"] != "Normal"][
    ["Campaign_ID", "Company", "Campaign_Type", "Channel_Used",
     "Target_Audience", "ROI", "Predicted_ROI", "ROI_Gap", "Anomaly"]
].sort_values("ROI_Gap")
anomaly_df.to_csv("../data/processed/anomaly_campaigns.csv", index=False)
print(f"  Exported {len(anomaly_df):,} anomalies -> ../data/processed/anomaly_campaigns.csv")


# ---------------------------------------------------------------------------
# 10. Smart Budget Allocator
# ---------------------------------------------------------------------------
print("\n-- Step 10: Budget Allocator")


def allocate_budget(total_budget, audience="All Ages", duration=30, n_steps=10):
    channels = list(le_dict["Channel_Used"].classes_)
    n_ch = len(channels)
    from itertools import product as iterproduct

    steps = list(range(0, n_steps + 1))
    allocs = []
    for combo in iterproduct(steps, repeat=n_ch):
        if sum(combo) == n_steps:
            allocs.append([c / n_steps * total_budget for c in combo])

    print(f"  Testing {len(allocs):,} allocations ...")
    best_total_roi = -np.inf
    best_alloc = None

    for alloc in allocs:
        total_roi = 0
        for ch, budget in zip(channels, alloc):
            if budget < 100:
                continue
            types = list(le_dict["Campaign_Type"].classes_)
            max_roi = 0
            for ct in types:
                row = {}
                med = df[["Clicks", "Impressions", "Engagement_Score", "Conversion_Rate"]].median()
                row["Acquisition_Cost"] = budget
                row["Duration"] = duration
                row["Clicks"] = med["Clicks"]
                row["Impressions"] = med["Impressions"]
                row["Engagement_Score"] = med["Engagement_Score"]
                row["Conversion_Rate"] = med["Conversion_Rate"]
                row["Target_Audience"] = audience
                row["Customer_Segment"] = df["Customer_Segment"].mode()[0]
                row["Location"] = df["Location"].mode()[0]
                row["Language"] = df["Language"].mode()[0]
                row["Campaign_Type"] = ct
                row["Channel_Used"] = ch
                denom = row["Clicks"] * row["Conversion_Rate"]
                row["CPA"] = row["Acquisition_Cost"] / denom if denom > 0 else 0
                row["ROAS"] = 0
                row["Engagement_Yield"] = row["Engagement_Score"] / row["Acquisition_Cost"]
                row["Interaction_Velocity"] = row["Clicks"] / row["Duration"]
                row["CTR"] = row["Clicks"] / row["Impressions"] if row["Impressions"] > 0 else 0
                row["Cost_Per_Impression"] = row["Acquisition_Cost"] / row["Impressions"] if row["Impressions"] > 0 else 0
                row["log_Acquisition_Cost"] = np.log1p(row["Acquisition_Cost"])
                row["log_Impressions"] = np.log1p(row["Impressions"])
                now = pd.Timestamp.now()
                row["Month_sin"] = np.sin(2 * np.pi * now.month / 12)
                row["Month_cos"] = np.cos(2 * np.pi * now.month / 12)
                row["DayOfWeek_sin"] = np.sin(2 * np.pi * now.dayofweek / 7)
                row["DayOfWeek_cos"] = np.cos(2 * np.pi * now.dayofweek / 7)
                for c in cat_cols:
                    row[f"{c}_enc"] = _safe_encode(le_dict[c], str(row[c]))
                row_df = pd.DataFrame([row])
                row_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                row_df.fillna(df[feature_cols].median(), inplace=True)
                pred = model_pipe.predict(row_df[feature_cols])[0]
                if pred > max_roi:
                    max_roi = pred
            total_roi += max_roi * (budget / total_budget)
        if total_roi > best_total_roi:
            best_total_roi = total_roi
            best_alloc = alloc

    print(f"\n  Optimal for ${total_budget:,}:")
    alloc_data = []
    for ch, amt in zip(channels, best_alloc):
        pct = amt / total_budget * 100
        if amt >= 100:
            print(f"    {ch:15s}: ${amt:>10,.0f}  ({pct:5.1f}%)")
            alloc_data.append({"Channel": ch, "Budget": amt, "Pct": pct})
    print(f"  Weighted ROI: {best_total_roi:.3f}")

    alloc_df = pd.DataFrame(alloc_data)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors_pie = sns.color_palette("Set2", len(alloc_df))
    axes[0].pie(alloc_df["Budget"], labels=alloc_df["Channel"],
                autopct=lambda p: f"${p/100*total_budget:,.0f}\n({p:.1f}%)",
                colors=colors_pie, startangle=140, textprops={"fontsize": 11})
    axes[0].set_title(f"Allocation: ${total_budget:,.0f}", fontsize=14)

    bars = axes[1].bar(alloc_df["Channel"], alloc_df["Budget"],
                       color=colors_pie, edgecolor="black", lw=0.8)
    for b, pct in zip(bars, alloc_df["Pct"]):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + total_budget*0.01,
                     f"${b.get_height():,.0f}\n({pct:.1f}%)",
                     ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Budget ($)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    axes[1].set_title(f"Channel Split (ROI: {best_total_roi:.3f})", fontsize=14)

    plt.suptitle("Smart Budget Allocator", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout(); save(fig, f"fig11_budget_{total_budget:.0f}.png")
    return alloc_df, best_total_roi


for budget in [25_000, 50_000, 100_000]:
    print(f"\n{'='*60}\n  ALLOCATING ${budget:,}\n{'='*60}")
    allocate_budget(budget)

print("\n" + "=" * 60)
print("  DONE. Charts saved. Model cached in ./model_weights/")
print("  Functions available: campaign_advisor('TechCorp'), allocate_budget(50000)")
print("=" * 60)
