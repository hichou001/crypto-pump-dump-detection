import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("final_dataset_15k.csv")

# =========================
# BASIC INFO
# =========================

print(df.head())

print(df.shape)

print(df.info())

# =========================
# LABEL DISTRIBUTION
# =========================

print(df['is_pump'].value_counts())

# =========================
# CORRELATION MATRIX
# =========================

plt.figure(figsize=(16,10))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    cmap='coolwarm'
)

plt.title("Correlation Matrix")

plt.show()

# =========================
# PUMP VS NORMAL
# =========================

pump_df = df[df['is_pump'] == 1]

normal_df = df[df['is_pump'] == 0]

# =========================
# RSI DISTRIBUTION
# =========================

plt.figure(figsize=(10,5))

plt.hist(
    normal_df['rsi'],
    bins=50,
    alpha=0.5,
    label='Normal'
)

plt.hist(
    pump_df['rsi'],
    bins=50,
    alpha=0.5,
    label='Pump'
)

plt.legend()

plt.title("RSI Distribution")

plt.show()

# =========================
# VOLUME RATIO DISTRIBUTION
# =========================

plt.figure(figsize=(10,5))

plt.hist(
    normal_df['vol_ratio'],
    bins=50,
    alpha=0.5,
    label='Normal'
)

plt.hist(
    pump_df['vol_ratio'],
    bins=50,
    alpha=0.5,
    label='Pump'
)

plt.legend()

plt.title("Volume Ratio Distribution")

plt.show()

# =========================
# PRICE ZSCORE
# =========================

plt.figure(figsize=(10,5))

plt.hist(
    normal_df['price_zscore'],
    bins=50,
    alpha=0.5,
    label='Normal'
)

plt.hist(
    pump_df['price_zscore'],
    bins=50,
    alpha=0.5,
    label='Pump'
)

plt.legend()

plt.title("Price Z-Score Distribution")

plt.show()