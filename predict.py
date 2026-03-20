"""
IoT Environmental Predictor
----------------------------
Predicts indoor temperature from sensor readings:
  DHT11  -> Temperature, Humidity
  LDR    -> Light intensity (Lux)
  MH-Z19 -> CO2 (ppm)

Dataset: UCI Occupancy Detection (Candanedo, 2016)
  Place datatraining.txt, datatest.txt, datatest2.txt alongside this file.
  Download: https://archive.ics.uci.edu/dataset/357/occupancy+detection

Run:  python predict.py
Outputs: assets/fig1_eda.png
         assets/fig2_model_eval.png
         assets/fig3_comparison.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs('assets', exist_ok=True)


# ── Style ─────────────────────────────────────────────────────────────────────
# Consistent dark palette used across all three figures

BG    = '#0d1117'
PANEL = '#161b22'
GRID  = '#21262d'
GREEN = '#39d353'
AMBER = '#f0a500'
BLUE  = '#58a6ff'
WHITE = '#e6edf3'
DIM   = '#4a6070'
MID   = '#7a9bb0'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    PANEL,
    'axes.edgecolor':    GRID,
    'axes.labelcolor':   MID,
    'xtick.color':       DIM,
    'ytick.color':       DIM,
    'text.color':        WHITE,
    'grid.color':        GRID,
    'grid.linewidth':    0.5,
    'font.family':       'monospace',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# ── Load ──────────────────────────────────────────────────────────────────────

train = pd.read_csv('datatraining.txt', parse_dates=['date'], index_col=0)
test1 = pd.read_csv('datatest.txt',     parse_dates=['date'], index_col=0)
test2 = pd.read_csv('datatest2.txt',    parse_dates=['date'], index_col=0)

print(f"Loaded  train={len(train)}  test1={len(test1)}  test2={len(test2)}")


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean(df):
    # Remove readings outside physical sensor limits
    return df[
        df['Temperature'].between(5, 45) &
        df['Humidity'].between(0, 100) &
        (df['Light'] >= 0) &
        df['CO2'].between(300, 5000)
    ]

train, test1, test2 = clean(train), clean(test1), clean(test2)


# ── Features ──────────────────────────────────────────────────────────────────

def add_features(df):
    df = df.copy()
    h = df['date'].dt.hour

    # Encode hour cyclically so 23:00 and 00:00 are numerically close
    df['hour_sin'] = np.sin(2 * np.pi * h / 24)
    df['hour_cos'] = np.cos(2 * np.pi * h / 24)

    df['day_of_week'] = df['date'].dt.dayofweek

    # Relative light: how bright vs recent 60-min max (mirrors LDR behavior)
    df['light_ratio'] = df['Light'] / (df['Light'].rolling(60, min_periods=1).max() + 1e-6)

    return df

train, test1, test2 = add_features(train), add_features(test1), add_features(test2)

FEATURES = ['Humidity', 'Light', 'CO2', 'HumidityRatio',
            'hour_sin', 'hour_cos', 'day_of_week', 'light_ratio']
TARGET = 'Temperature'

X_train, y_train = train[FEATURES], train[TARGET]
X_t1,    y_t1    = test1[FEATURES], test1[TARGET]
X_t2,    y_t2    = test2[FEATURES], test2[TARGET]

# Scale for linear regression — tree models don't need it
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_t1_sc    = scaler.transform(X_t1)
X_t2_sc    = scaler.transform(X_t2)


# ── Train ─────────────────────────────────────────────────────────────────────

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                                    learning_rate=0.08, random_state=42),
}

results  = {}
preds_t1 = {}
preds_t2 = {}

print(f"\n{'Model':<22} {'Test1 R2':>8}  {'Test1 RMSE':>10}  {'Test2 R2':>8}  {'Test2 RMSE':>10}")
print("-" * 65)

for name, model in models.items():
    sc = name == 'Linear Regression'
    model.fit(X_train_sc if sc else X_train, y_train)

    p1 = model.predict(X_t1_sc if sc else X_t1)
    p2 = model.predict(X_t2_sc if sc else X_t2)

    r2_1  = r2_score(y_t1, p1);  rmse_1 = mean_squared_error(y_t1, p1) ** 0.5
    r2_2  = r2_score(y_t2, p2);  rmse_2 = mean_squared_error(y_t2, p2) ** 0.5

    results[name]  = dict(r2_t1=round(r2_1, 4), rmse_t1=round(rmse_1, 4),
                          r2_t2=round(r2_2, 4), rmse_t2=round(rmse_2, 4))
    preds_t1[name] = p1
    preds_t2[name] = p2

    print(f"{name:<22} {r2_1:>8.4f}  {rmse_1:>10.4f}  {r2_2:>8.4f}  {rmse_2:>10.4f}")


# ── Feature importance ────────────────────────────────────────────────────────

rf  = models['Random Forest']
imp = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])
print("\nFeature importances (Random Forest):")
for feat, score in imp:
    print(f"  {feat:<15} {score:.4f}")


# ── Figure 1: EDA ─────────────────────────────────────────────────────────────
# Temperature timeline across all splits + correlation heatmap + scatter

fig = plt.figure(figsize=(14, 9), facecolor=BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

# Temperature timeline — all three splits on same axis
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(train['date'], train['Temperature'], color=AMBER, linewidth=0.6, label='Training (Feb 4-10)')
ax0.plot(test1['date'], test1['Temperature'], color=BLUE,  linewidth=0.8, label='Test 1 (Feb 2-4)')
ax0.plot(test2['date'], test2['Temperature'], color=GREEN, linewidth=0.6, label='Test 2 (Feb 11-18)')
ax0.set_ylabel('Temperature (C)', fontsize=9)
ax0.set_title('Temperature across all three dataset splits', fontsize=10, color=MID, pad=6)
ax0.legend(fontsize=8, framealpha=0.2)
ax0.grid(True, alpha=0.4)
ax0.tick_params(axis='x', labelrotation=20, labelsize=7)

# Correlation heatmap (training set)
ax1 = fig.add_subplot(gs[1, 0])
corr = train[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy']].corr()
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, ax=ax1,
            linewidths=0.5, linecolor=GRID,
            annot_kws={'size': 8, 'color': WHITE},
            cbar_kws={'shrink': 0.8})
ax1.set_title('Correlation matrix (training)', fontsize=9, color=MID, pad=4)
ax1.tick_params(labelsize=7)

# Light vs Temperature coloured by occupancy
ax2 = fig.add_subplot(gs[1, 1])
ax2.set_facecolor(PANEL)
unocc = train[train['Occupancy'] == 0]
occ   = train[train['Occupancy'] == 1]
ax2.scatter(unocc['Light'], unocc['Temperature'], s=2, color=DIM,   alpha=0.4, label='Unoccupied')
ax2.scatter(occ['Light'],   occ['Temperature'],   s=2, color=GREEN, alpha=0.4, label='Occupied')
ax2.set_xlabel('Light (Lux)', fontsize=9)
ax2.set_ylabel('Temperature (C)', fontsize=9)
ax2.set_title('Light vs Temperature by occupancy', fontsize=9, color=MID, pad=4)
ax2.legend(fontsize=7, framealpha=0.2)
ax2.grid(True, alpha=0.3)
for sp in ax2.spines.values():
    sp.set_edgecolor(GRID)

plt.savefig('assets/fig1_eda.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("\nSaved: assets/fig1_eda.png")


# ── Figure 2: Actual vs Predicted — Test Set 1 ───────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=BG)
fig.suptitle('Actual vs Predicted Temperature - Test Set 1', color=WHITE,
             fontsize=11, fontfamily='monospace')

colors = [BLUE, GREEN, AMBER]

for ax, name, color in zip(axes, models.keys(), colors):
    ax.set_facecolor(PANEL)
    ax.scatter(y_t1, preds_t1[name], s=3, alpha=0.35, color=color, rasterized=True)
    lo, hi = y_t1.min(), y_t1.max()
    ax.plot([lo, hi], [lo, hi], '--', color=WHITE, linewidth=0.8, alpha=0.4)
    ax.set_title(f"{name}\nR2={results[name]['r2_t1']}  RMSE={results[name]['rmse_t1']} C",
                 color=color, fontsize=9, fontfamily='monospace')
    ax.set_xlabel('Actual (C)', fontsize=8)
    ax.set_ylabel('Predicted (C)', fontsize=8)
    ax.tick_params(labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

plt.tight_layout()
plt.savefig('assets/fig2_model_eval.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: assets/fig2_model_eval.png")


# ── Figure 3: Feature importance + R2 comparison across both test sets ────────

fig, (ax_fi, ax_r2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle('Feature Importance & Cross-Split Comparison',
             color=WHITE, fontsize=11, fontfamily='monospace')

# Feature importance horizontal bars
feat_names = [f for f, _ in imp]
feat_vals  = [v for _, v in imp]
bar_colors = [GREEN if i == 0 else BLUE if i == 1 else DIM for i in range(len(feat_names))]
ax_fi.set_facecolor(PANEL)
ax_fi.barh(feat_names, feat_vals, color=bar_colors, edgecolor=GRID, linewidth=0.4)
ax_fi.set_xlabel('Importance', fontsize=9)
ax_fi.set_title('Random Forest - Feature Importances', fontsize=9, color=MID, pad=6)
ax_fi.invert_yaxis()
ax_fi.grid(True, axis='x', alpha=0.3)
for sp in ax_fi.spines.values():
    sp.set_edgecolor(GRID)
for i, (name, val) in enumerate(zip(feat_names, feat_vals)):
    ax_fi.text(val + 0.003, i, f'{val:.3f}', va='center', fontsize=7, color=MID)

# R2 grouped bars — same model, two different test windows
# This directly visualises distribution shift
model_names = list(results.keys())
short       = ['Linear\nReg', 'Random\nForest', 'Grad.\nBoost']
r2_t1_vals  = [results[m]['r2_t1'] for m in model_names]
r2_t2_vals  = [results[m]['r2_t2'] for m in model_names]
x = np.arange(len(model_names))
w = 0.35

ax_r2.set_facecolor(PANEL)
ax_r2.bar(x - w/2, r2_t1_vals, w, label='Test 1 (same week)',      color=BLUE,  alpha=0.85)
ax_r2.bar(x + w/2, r2_t2_vals, w, label='Test 2 (following week)', color=GREEN, alpha=0.85)
ax_r2.axhline(0, color=DIM, linewidth=0.8, linestyle='--')
ax_r2.set_xticks(x)
ax_r2.set_xticklabels(short, fontsize=9)
ax_r2.set_ylabel('R2', fontsize=10)
ax_r2.set_title('R2 Score - Both Test Sets', fontsize=9, color=MID, pad=6)
ax_r2.legend(fontsize=8, framealpha=0.2)
ax_r2.set_ylim(-1.1, 1.1)
ax_r2.grid(True, axis='y', alpha=0.3)
for sp in ax_r2.spines.values():
    sp.set_edgecolor(GRID)

plt.tight_layout()
plt.savefig('assets/fig3_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: assets/fig3_comparison.png")
