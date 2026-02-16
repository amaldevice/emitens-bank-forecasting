# Macroeconomic Data Sources

This directory contains macroeconomic indicators for Indonesia covering the period **2004-2024** to be integrated into the financial distress prediction model.

## 📊 Available Data

### 1. Inflation Rate (CPI Annual % Change)
- **File:** `API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_23195/API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_23195.csv`
- **Source:** [World Bank - International Monetary Fund, International Financial Statistics](https://data.worldbank.org/)
- **Indicator Code:** FP.CPI.TOTL.ZG
- **Coverage:** 2004-2024 (21 years)
- **Description:** Consumer Price Index annual percentage change, measuring inflation rate
- **Sample Data:**
  - 2004: 6.76%
  - 2008: 6.41% (Financial crisis)
  - 2020: 3.03% (COVID-19 pandemic)
  - 2024: 3.67%

### 2. BI Rate (Policy Interest Rate)
- **File:** `indonesia_bi_rate_fred.csv`
- **Source:** [FRED - Federal Reserve Economic Data (OECD)](https://fred.stlouisfed.org/series/IRSTCB01IDM156N)
- **Series ID:** IRSTCB01IDM156N
- **Coverage:** 2004-2023 (monthly data)
- **Description:** Bank Indonesia policy rate (7-Day Reverse Repo Rate since 2016, BI Rate before that)
- **Note:** 2024 data needs to be manually added (6.00% based on end-of-year rate)
- **Sample Data:**
  - 2004: 7.86%
  - 2008: 9.50% (Peak during financial crisis)
  - 2020: 4.00% (COVID-19 stimulus)
  - 2023: 6.00%

### 3. Exchange Rate (USD/IDR)
- **File:** `API_IDN_PA.NUS.FCRF_en_csv_v2_111434.csv`
- **Source:** [World Bank - International Monetary Fund, International Financial Statistics](https://data.worldbank.org/)
- **Indicator Code:** PA.NUS.FCRF
- **Coverage:** 2004-2024 (21 years)
- **Description:** Official exchange rate (Local Currency Units per US Dollar, period average)
- **Sample Data:**
  - 2004: 8,577.13 IDR/USD
  - 2008: 9,141.00 IDR/USD (Financial crisis)
  - 2020: 14,147.67 IDR/USD (COVID-19 impact)
  - 2024: 15,236.88 IDR/USD

### 4. IHSG / JCI (Jakarta Composite Index)
- **Files:**
  - `ihsg_daily_2004_2024.csv` - Daily OHLCV data (5,101 trading days)
  - `ihsg_annual_2004_2024.csv` - Annual aggregated metrics
- **Source:** [Yahoo Finance](https://finance.yahoo.com/quote/%5EJKSE/)
- **Ticker:** ^JKSE
- **Coverage:** 2004-2024 (21 years)
- **Description:** Indonesia's main stock market index tracking all listed companies on Jakarta Stock Exchange
- **Annual Metrics Include:**
  - Open: Opening index at start of year
  - High: Highest index during the year
  - Low: Lowest index during the year
  - Close: Closing index at end of year
  - Volume: Total trading volume
  - Avg_Close: Average closing index
  - Yearly_Return: Year-over-year percentage return
- **Sample Data:**
  - 2004: 1,000.19 (close)
  - 2008: 1,355.36 (close, -50.64% return - financial crisis)
  - 2020: 5,979.07 (close, -5.09% return - COVID-19)
  - 2024: 7,079.90 (close, -2.65% return)

### 5. GDP Growth Rate
- **File:** Already integrated in `preprocessed_metrics_data.csv`
- **Source:** Previously obtained
- **Coverage:** 2004-2024
- **Description:** Indonesia's annual GDP growth rate

## 🔧 Data Processing Scripts

### `get_ihsg_data.py`
Script to download IHSG historical data from Yahoo Finance using `yfinance` library.

**Usage:**
```bash
python get_ihsg_data.py
```

**Output:**
- Daily OHLCV data: `ihsg_daily_2004_2024.csv`
- Annual metrics: `ihsg_annual_2004_2024.csv`

### `add_macro_data.py` (to be created)
Script to integrate all macroeconomic data into `preprocessed_metrics_data.csv`.

**Will merge by year:**
- Inflation rate
- BI Rate (annual average from monthly data)
- Exchange rate (annual average)
- IHSG metrics (Close, Avg_Close, Yearly_Return)

## 📝 Notes

1. **Data Frequency:** All macro indicators are aggregated to annual frequency to match the main dataset
2. **Missing Data:** BI Rate 2024 needs manual addition (6.00%)
3. **Data Quality:** All sources are official/reputable:
   - World Bank (IMF data)
   - FRED (OECD data)
   - Yahoo Finance (market data)
4. **Time Period:** Consistent 2004-2024 coverage across all indicators (21 years)

## 🌐 Direct Download Links

- **World Bank Inflation:** `https://api.worldbank.org/v2/en/indicator/FP.CPI.TOTL.ZG?downloadformat=csv`
- **World Bank Exchange Rate:** `https://api.worldbank.org/v2/en/country/IDN/indicator/PA.NUS.FCRF?downloadformat=csv`
- **FRED BI Rate:** `https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRSTCB01IDM156N`
- **Yahoo Finance IHSG:** Use `yfinance` library with ticker `^JKSE`

## 📚 References

- [World Bank Open Data](https://data.worldbank.org/)
- [FRED Economic Data](https://fred.stlouisfed.org/)
- [Bank Indonesia](https://www.bi.go.id/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Indonesia Stock Exchange](https://www.idx.co.id/)

---

**Last Updated:** October 19, 2025
