import pandas as pd

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("pump_dump_dataset.csv")

print(df.columns)

# =========================
# CLEAN DATA
# =========================

df = df.dropna()

df = df.drop_duplicates()

# =========================
# SORT TIME
# =========================

df['timestamp'] = pd.to_datetime(df['timestamp'])

if 'symbol' in df.columns:
    df = df.sort_values(['symbol', 'timestamp'])
else:
    df = df.sort_values('timestamp')

# =========================
# CREATE FUTURE RETURN
# =========================

if 'symbol' in df.columns:

    df['future_return_5m'] = (
        df.groupby('symbol')['close']
        .shift(-5) - df['close']
    ) / df['close']

else:

    df['future_return_5m'] = (
        df['close'].shift(-5) - df['close']
    ) / df['close']

# =========================
# CREATE PUMP LABEL
# =========================

df['is_pump'] = (
    df['future_return_5m'] > 0.05
).astype(int)

# =========================
# CHECK LABEL DISTRIBUTION
# =========================

print("\nLABEL DISTRIBUTION")

print(df['is_pump'].value_counts())

# =========================
# SPLIT PUMP / NORMAL
# =========================

pump_df = df[df['is_pump'] == 1]

normal_df = df[df['is_pump'] == 0]

# =========================
# SAMPLE NORMAL DATA
# =========================

normal_sample = normal_df.sample(
    n=14320,
    random_state=42
)

# =========================
# COMBINE FINAL DATA
# =========================

final_df = pd.concat([
    pump_df,
    normal_sample
])

# =========================
# SHUFFLE
# =========================

final_df = final_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# =========================
# FINAL CHECK
# =========================

print("\nFINAL DATASET")

print(final_df['is_pump'].value_counts())

print(final_df.shape)

# =========================
# SAVE
# =========================

final_df.to_csv(
    "final_dataset_10k.csv",
    index=False
)

print("\nSaved: final_dataset_10k.csv")