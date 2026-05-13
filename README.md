# crypto-pump-dump-detection
# Crypto Pump & Dump Detection

Machine Learning project for detecting abnormal crypto market signals and early Pump & Dump activities.

## Overview
This project aims to detect abnormal crypto market behavior and provide early warnings for potential Pump & Dump events.

## Dataset
- Binance data collected via CCXT
- 1-minute timeframe
- Over 3 million raw rows
- Final processed dataset: ~15,000 rows

## Features
- RSI
- MACD
- Bollinger Bands
- Volatility
- Volume Ratio
- Future Return

## Models
- Random Forest
- XGBoost
- LightGBM
- SVM
- ANN

## Results
XGBoost achieved the best ROC-AUC performance for Pump & Dump detection.

## Project Structure
- Data Collection
- Data Preprocessing
- Feature Engineering
- Model Training
- Evaluation
