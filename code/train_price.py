import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

print("🏦 LSTM Closing Price Prediction Pipeline")
print("=" * 50)

# Input bank name
BANK_NAME = input("Enter bank name (e.g., BBCA): ").strip().upper()

# Input forecast days
FORECAST_DAYS = int(input("Enter number of days to forecast (e.g., 30, 60, 90): ").strip())

# Ask for hyperparameter optimization
USE_OPTIMIZATION = input("Use metaheuristic optimization? (y/n, default=n): ").strip().lower()
if USE_OPTIMIZATION == '':
    USE_OPTIMIZATION = 'n'

print(f"\n🔄 Processing {BANK_NAME} data...")
print(f"📅 Forecast period: {FORECAST_DAYS} days")

# Create output folder for this bank
OUTPUT_FOLDER = f"results_revised/{BANK_NAME.upper()}"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"📁 Output folder: {OUTPUT_FOLDER}")

# Load CSV file
csv_path = f"data/{BANK_NAME}.csv"

if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
    print(f"✅ Successfully loaded {BANK_NAME} data: {data.shape}")
else:
    print(f"❌ Error: File {csv_path} not found")
    exit()

# =============================================================================
# STEP 1: FEATURE SELECTION & DATA PREPROCESSING
# =============================================================================

print("\n" + "="*50)
print("STEP 1: FEATURE SELECTION & PREPROCESSING")
print("="*50)

# Define target variable
target_col = 'Closing Price'

# Selected features (based on correlation analysis)
feature_cols = [
    'Ask Price', 'Bid Price', 'High Price', 'Low Price',
    'Moving Average 20 Day',
    'Price to Sales Ratio', 'Price to Book Ratio', 'Price Earnings Ratio',
    'Dividend Indicated Yld - Gross', 'Price Change 1 Day Percent'
]

print(f"Target variable: {target_col}")
print(f"Selected features ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

# Verify columns exist
missing_cols = [col for col in feature_cols + [target_col] if col not in data.columns]
if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
    exit()

print(f"✅ All columns available!")

# Parse dates and set index
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date').sort_index()

print(f"Date range: {data.index.min()} to {data.index.max()}")
print(f"Data shape after date processing: {data.shape}")

# Handle missing values
missing_counts = data.isnull().sum()
if missing_counts.sum() > 0:
    print(f"⚠️  Missing values found, applying forward fill...")
    data = data.fillna(method='ffill').dropna()
    print(f"Data shape after handling missing values: {data.shape}")
else:
    print(f"✅ No missing values found!")

# Feature selection
selected_data = data[feature_cols + [target_col]].copy()
print(f"Selected data shape: {selected_data.shape}")

print(f"✅ Step 1 completed: Feature selection and preprocessing done!")

# =============================================================================
# STEP 2: DATA SCALING & SPLITTING
# =============================================================================

print("\n" + "="*50)
print("STEP 2: DATA SCALING & SPLITTING")
print("="*50)

# Separate features and target
X = selected_data[feature_cols].values
y = selected_data[target_col].values

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Scale features and target
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

X_scaled = feature_scaler.fit_transform(X)
y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

print(f"Features scaled - Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
print(f"Target scaled - Mean: {y_scaled.mean():.4f}, Std: {y_scaled.std():.4f}")

# Data splitting by date (6-2-2 format)
def split_dataset_by_date(df, start_date, end_date):
    """Split dataset based on date range"""
    return df[(df.index >= start_date) & (df.index < end_date)]

# Create scaled dataset
df_scaled = selected_data.copy()
df_scaled[feature_cols] = X_scaled
df_scaled[target_col] = y_scaled

# Split data by date ranges
train_set = split_dataset_by_date(df_scaled, '2014-01-02', '2020-01-02')  # 6 years
val_set = split_dataset_by_date(df_scaled, '2020-01-02', '2022-01-02')    # 2 years
test_set = split_dataset_by_date(df_scaled, '2022-01-02', '2024-12-31')   # 2 years

print(f"\nDataset splits:")
print(f"Train: {train_set.shape} | {train_set.index.min()} to {train_set.index.max()}")
print(f"Val:   {val_set.shape} | {val_set.index.min()} to {val_set.index.max()}")
print(f"Test:  {test_set.shape} | {test_set.index.min()} to {test_set.index.max()}")
print(f"Total: {train_set.shape[0] + val_set.shape[0] + test_set.shape[0]} samples")

print(f"✅ Step 2 completed: Data scaling and splitting done!")

# =============================================================================
# STEP 3: LSTM SEQUENCE GENERATION
# =============================================================================

print("\n" + "="*50)
print("STEP 3: LSTM SEQUENCE GENERATION")
print("="*50)

def create_sequences(data, feature_cols, target_col, time_steps=60):
    """
    Create sequences for LSTM input
    Args:
        data: DataFrame with scaled data
        feature_cols: list of feature column names
        target_col: target column name
        time_steps: number of time steps for sequence
    Returns:
        X: array of sequences (samples, time_steps, features)
        y: array of targets (samples,)
    """
    X, y = [], []

    features = data[feature_cols].values
    targets = data[target_col].values

    for i in range(time_steps, len(data)):
        X.append(features[i-time_steps:i])
        y.append(targets[i])

    return np.array(X), np.array(y)

# Set time steps (60 days = ~3 months trading data)
TIME_STEPS = 60

print(f"Time steps for sequences: {TIME_STEPS} days")
print("Creating LSTM sequences...")

# Create sequences for each split
X_train, y_train = create_sequences(train_set, feature_cols, target_col, TIME_STEPS)
X_val, y_val = create_sequences(val_set, feature_cols, target_col, TIME_STEPS)
X_test, y_test = create_sequences(test_set, feature_cols, target_col, TIME_STEPS)

print(f"\nSequence shapes:")
print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape} | y_val: {y_val.shape}")
print(f"X_test:  {X_test.shape} | y_test: {y_test.shape}")

print(f"Total sequences: {len(X_train) + len(X_val) + len(X_test)}")
print(f"Input shape for LSTM: (batch_size, {TIME_STEPS}, {len(feature_cols)})")

print(f"✅ Step 3 completed: LSTM sequences generated!")

# =============================================================================
# STEP 4: METAHEURISTIC HYPERPARAMETER OPTIMIZATION (Optional)
# =============================================================================

if USE_OPTIMIZATION == 'y':
    print("\n" + "="*50)
    print("STEP 4: FAST GENETIC ALGORITHM OPTIMIZATION")
    print("="*50)

    import random
    from sklearn.metrics import mean_squared_error

    # Define hyperparameter search space
    PARAM_SPACE = {
        'lstm_units': [16, 32, 64],
        'learning_rate': [0.0001, 0.0005, 0.001],
        'batch_size': [16, 32, 64],
        'dropout_rate': [0.2, 0.3, 0.4]
    }

    def create_individual():
        """Create random individual (hyperparameter set)"""
        return {
            'lstm_units': random.choice(PARAM_SPACE['lstm_units']),
            'learning_rate': random.choice(PARAM_SPACE['learning_rate']),
            'batch_size': random.choice(PARAM_SPACE['batch_size']),
            'dropout_rate': random.choice(PARAM_SPACE['dropout_rate'])
        }

    def build_model_with_params(input_shape, params):
        """Build LSTM model with specific hyperparameters"""
        model = Sequential()
        model.add(LSTM(units=params['lstm_units'], return_sequences=False,
                      input_shape=input_shape, name='lstm_1'))
        model.add(Dropout(params['dropout_rate'], name='dropout_1'))
        model.add(Dense(16, activation='relu', name='dense_1'))
        model.add(Dropout(0.2, name='dropout_2'))
        model.add(Dense(1, activation='linear', name='output'))

        model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                     loss='mae', metrics=['mse', 'mae'])
        return model

    def evaluate_individual(params):
        """Fast evaluation with reduced epochs"""
        try:
            model = build_model_with_params((TIME_STEPS, len(feature_cols)), params)

            # Fast training (5 epochs only for speed)
            history = model.fit(X_train, y_train,
                              batch_size=params['batch_size'],
                              epochs=5,  # Very fast evaluation
                              validation_data=(X_val, y_val),
                              verbose=0)

            # Return validation loss (lower is better)
            val_loss = min(history.history['val_loss'])
            del model  # Free memory
            return val_loss
        except:
            return 999.0  # Penalty for failed configs

    def genetic_algorithm(pop_size=8, generations=3):
        """Fast GA with small population and few generations"""
        print(f"🧬 Running Fast GA: {pop_size} individuals × {generations} generations")
        print("⚡ Using 5 epochs per evaluation for speed")

        # Initialize population
        population = [create_individual() for _ in range(pop_size)]
        best_individual = None
        best_fitness = float('inf')

        for gen in range(generations):
            print(f"\nGeneration {gen+1}/{generations}")

            # Evaluate population
            fitness_scores = []
            for i, individual in enumerate(population):
                fitness = evaluate_individual(individual)
                fitness_scores.append(fitness)
                print(f"  Individual {i+1}: loss={fitness:.4f} | {individual}")

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()

            # Selection (tournament selection)
            new_population = []
            for _ in range(pop_size):
                # Tournament selection
                idx1, idx2 = random.sample(range(pop_size), 2)
                if fitness_scores[idx1] < fitness_scores[idx2]:
                    parent = population[idx1].copy()
                else:
                    parent = population[idx2].copy()

                # Mutation (20% chance)
                if random.random() < 0.2:
                    param_to_mutate = random.choice(list(PARAM_SPACE.keys()))
                    parent[param_to_mutate] = random.choice(PARAM_SPACE[param_to_mutate])

                new_population.append(parent)

            population = new_population

        print(f"\n🏆 Best hyperparameters found:")
        print(f"   LSTM units: {best_individual['lstm_units']}")
        print(f"   Learning rate: {best_individual['learning_rate']}")
        print(f"   Batch size: {best_individual['batch_size']}")
        print(f"   Dropout rate: {best_individual['dropout_rate']}")
        print(f"   Best validation loss: {best_fitness:.4f}")

        return best_individual

    # Run optimization
    optimal_params = genetic_algorithm()
    print(f"✅ Step 4 completed: Fast GA optimization done!")

else:
    # Default parameters
    optimal_params = {
        'lstm_units': 32,
        'learning_rate': 0.001,
        'batch_size': 32,
        'dropout_rate': 0.3
    }
    print(f"\n⏭️  Skipping optimization, using default hyperparameters")

# =============================================================================
# STEP 5: BUILD OPTIMIZED LSTM MODEL ARCHITECTURE
# =============================================================================

print("\n" + "="*50)
print("STEP 5: BUILD OPTIMIZED LSTM MODEL")
print("="*50)

def build_lstm_model(input_shape):
    """
    Build simplified LSTM model for closing price prediction
    Args:
        input_shape: tuple (time_steps, features)
    Returns:
        compiled Keras model
    """
    model = Sequential()

    # Optimized LSTM layer
    model.add(LSTM(units=optimal_params['lstm_units'], return_sequences=False,
                  input_shape=input_shape, name='lstm_1'))
    model.add(Dropout(optimal_params['dropout_rate'], name='dropout_1'))

    # Single Dense layer
    model.add(Dense(16, activation='relu', name='dense_1'))
    model.add(Dropout(0.2, name='dropout_2'))

    # Output layer (linear for regression)
    model.add(Dense(1, activation='linear', name='output'))

    return model

# Build model
input_shape = (TIME_STEPS, len(feature_cols))
model = build_lstm_model(input_shape)

print("Model architecture:")
model.summary()

# Compile model
print(f"\nCompiling model...")

model.compile(
    optimizer=Adam(learning_rate=optimal_params['learning_rate']),
    loss='mae',
    metrics=['mse', 'mae']
)

print(f"✅ Model compiled successfully!")
print(f"Optimizer: Adam (lr={optimal_params['learning_rate']})")
print(f"Loss: MSE (standard regression loss)")
print(f"Metrics: MAE")
print(f"LSTM units: {optimal_params['lstm_units']}")
print(f"Dropout rate: {optimal_params['dropout_rate']}")
print(f"Batch size: {optimal_params['batch_size']}")
print(f"Total parameters: {model.count_params():,}")

print(f"✅ Step 5 completed: Optimized LSTM model built!")

# =============================================================================
# STEP 6: MODEL TRAINING WITH OPTIMIZED PARAMETERS
# =============================================================================

print("\n" + "="*50)
print("STEP 6: MODEL TRAINING WITH OPTIMIZED PARAMETERS")
print("="*50)

# Setup callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=1e-6,
    verbose=1
)

# Training parameters
EPOCHS = 50
BATCH_SIZE = optimal_params['batch_size']

print(f"Training parameters:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Train sequences: {len(X_train)}")
print(f"  Validation sequences: {len(X_val)}")
print(f"  Early stopping patience: 20")
print(f"  Learning rate reduction patience: 8")

print(f"\n🚀 Starting training...")

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=optimal_params['batch_size'],
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print(f"\n✅ Training completed!")
epochs_trained = len(history.history['loss'])
print(f"Total epochs trained: {epochs_trained}")
print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
print(f"Final training loss: {history.history['loss'][-1]:.6f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")

print(f"✅ Step 5 completed: Model training finished!")

# =============================================================================
# STEP 6: MODEL EVALUATION & FORECASTING
# =============================================================================

print("\n" + "="*50)
print("STEP 6: MODEL EVALUATION & FORECASTING")
print("="*50)

# Evaluate on test set
print("Evaluating model on test set...")
y_test_pred_scaled = model.predict(X_test, verbose=0)

# Inverse transform to original scale
y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled)
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# Flatten arrays
y_test_pred = y_test_pred.flatten()
y_test_actual = y_test_actual.flatten()

# Calculate metrics
test_mse = mean_squared_error(y_test_actual, y_test_pred)
test_mae = mean_absolute_error(y_test_actual, y_test_pred)
test_rmse = np.sqrt(test_mse)

# Calculate normalized MSE
mean_actual_price = np.mean(y_test_actual)
normalized_mse = test_mse / (mean_actual_price ** 2)
normalized_rmse = test_rmse / mean_actual_price

print(f"\n📊 Test Set Performance:")
print(f"  MSE:  {test_mse:.2f}")
print(f"  RMSE: {test_rmse:.2f}")
print(f"  MAE:  {test_mae:.2f}")
print(f"  Normalized MSE: {normalized_mse:.4f} ({normalized_mse*100:.2f}%)")
print(f"  Normalized RMSE: {normalized_rmse:.4f} ({normalized_rmse*100:.2f}%)")

# Calculate percentage accuracy
mape = np.mean(np.abs((y_test_actual - y_test_pred) / (y_test_actual + 1e-8))) * 100
print(f"  MAPE: {mape:.2f}%")

# Sample predictions with percentage analysis
print(f"\n📈 Sample Predictions (Last 10 days):")
print(f"{'Actual':<10} {'Predicted':<10} {'Error':<8} {'Daily%':<8}")
print("-" * 38)
for i in range(-10, 0):
    actual = y_test_actual[i]
    pred = y_test_pred[i]
    error = abs(actual - pred)

    # Calculate daily percentage change for actual prices
    if i == -10:
        actual_daily_pct = 0.0  # First day has no previous day
    else:
        prev_actual = y_test_actual[i-1]
        actual_daily_pct = ((actual - prev_actual) / prev_actual) * 100

    print(f"{actual:<10.2f} {pred:<10.2f} {error:<8.2f} {actual_daily_pct:<7.2f}%")

# Calculate percentage changes for test predictions
print(f"\n📊 Test Period Percentage Analysis:")
print("-" * 40)

# Daily percentage changes for actual vs predicted
actual_daily_changes = []
pred_daily_changes = []

for i in range(1, len(y_test_actual)):
    # Actual daily change
    actual_change = ((y_test_actual[i] - y_test_actual[i-1]) / y_test_actual[i-1]) * 100
    actual_daily_changes.append(actual_change)

    # Predicted daily change
    pred_change = ((y_test_pred[i] - y_test_pred[i-1]) / y_test_pred[i-1]) * 100
    pred_daily_changes.append(pred_change)

actual_daily_changes = np.array(actual_daily_changes)
pred_daily_changes = np.array(pred_daily_changes)

print(f"Daily Change Stats:")
print(f"  Actual: avg={actual_daily_changes.mean():.3f}%, std={actual_daily_changes.std():.3f}%")
print(f"  Predicted: avg={pred_daily_changes.mean():.3f}%, std={pred_daily_changes.std():.3f}%")

# Cumulative percentage change from start to end
test_start_price = y_test_actual[0]
test_end_price = y_test_actual[-1]
actual_cumulative = ((test_end_price - test_start_price) / test_start_price) * 100

pred_start_price = y_test_pred[0]
pred_end_price = y_test_pred[-1]
pred_cumulative = ((pred_end_price - pred_start_price) / pred_start_price) * 100

print(f"\nCumulative Change (Test Period):")
print(f"  Actual: {actual_cumulative:.2f}% (from {test_start_price:.2f} to {test_end_price:.2f})")
print(f"  Predicted: {pred_cumulative:.2f}% (from {pred_start_price:.2f} to {pred_end_price:.2f})")
print(f"  Difference: {abs(actual_cumulative - pred_cumulative):.2f}%")

# 30-day forecasting
print(f"\n🔮 {FORECAST_DAYS}-Day Forecast:")
last_sequence = X_test[-1:] # Shape: (1, TIME_STEPS, features)

def forecast_future(model, last_sequence, scaler_features, scaler_target, days=30):
    """
    Forecast future closing prices using the trained model
    """
    forecasts = []
    current_sequence = last_sequence.copy() # Shape: (1, TIME_STEPS, features)

    for day in range(days):
        # Make prediction (scaled)
        pred_scaled = model.predict(current_sequence, verbose=0)[0][0]

        # Convert prediction back to original scale
        pred_original = scaler_target.inverse_transform([[pred_scaled]])[0][0]
        forecasts.append(pred_original)

        # Update sequence for next prediction (simulate next day features)
        # Use last day's features as base and update price-related features
        next_day_features = current_sequence[0, -1].copy()

        # Update price-related features based on predicted price (scaled)
        # Feature order: 'Ask Price', 'Bid Price', 'High Price', 'Low Price',
        #               'Moving Average 20 Day', 'VWAP', ...

        # Update price features to evolve with prediction
        next_day_features[0] = pred_scaled * 1.001  # Ask Price (slightly higher)
        next_day_features[1] = pred_scaled * 0.999  # Bid Price (slightly lower)
        next_day_features[2] = pred_scaled * 1.003  # High Price (potential high)
        next_day_features[3] = pred_scaled * 0.997  # Low Price (potential low)
        next_day_features[4] = pred_scaled          # Moving Average 20 Day (close to predicted)
        next_day_features[5] = pred_scaled          # VWAP (close to predicted)

        # Keep other features (ratios, volume, etc.) relatively unchanged
        # They evolve more slowly than price features

        # Roll the sequence: remove first day, add simulated next day
        new_sequence = np.roll(current_sequence[0], -1, axis=0)
        new_sequence[-1] = next_day_features
        current_sequence = new_sequence.reshape(1, TIME_STEPS, -1)

        if (day + 1) % 10 == 0:
            print(f"  Day {day+1}: {pred_original:.2f}")

    return np.array(forecasts)

# Generate 30-day forecast
# Use the user-defined forecast days
forecasts = forecast_future(model, last_sequence, feature_scaler, target_scaler, FORECAST_DAYS)

print(f"\n📊 Forecast Summary:")
print(f"  Average predicted price: {forecasts.mean():.2f}")
print(f"  Min predicted price: {forecasts.min():.2f}")
print(f"  Max predicted price: {forecasts.max():.2f}")
print(f"  Price volatility (std): {forecasts.std():.2f}")

# Calculate percentage changes for forecast
print(f"\n📈 Forecast Percentage Analysis:")
print("-" * 40)

# Daily percentage changes in forecast
forecast_daily_changes = []
for i in range(1, len(forecasts)):
    daily_change = ((forecasts[i] - forecasts[i-1]) / forecasts[i-1]) * 100
    forecast_daily_changes.append(daily_change)

forecast_daily_changes = np.array(forecast_daily_changes)

print(f"Daily Change Stats (Forecast):")
print(f"  Average: {forecast_daily_changes.mean():.3f}%")
print(f"  Std: {forecast_daily_changes.std():.3f}%")
print(f"  Max gain: {forecast_daily_changes.max():.2f}%")
print(f"  Max loss: {forecast_daily_changes.min():.2f}%")

# Cumulative change from last actual price to end of forecast
last_actual_price = y_test_actual[-1]  # Last known actual price
forecast_end_price = forecasts[-1]
forecast_cumulative = ((forecast_end_price - last_actual_price) / last_actual_price) * 100

print(f"\nCumulative Change (Forecast Period):")
print(f"  From last actual: {forecast_cumulative:.2f}% (from {last_actual_price:.2f} to {forecast_end_price:.2f})")

# Show first few and last few forecast changes
print(f"\n📅 Daily Changes in Forecast:")
print(f"{'Day':<4} {'Price':<8} {'Change%':<8}")
print("-" * 22)
print(f"0    {last_actual_price:<8.2f} --      ")  # Starting point
for i in range(min(5, len(forecasts))):
    if i == 0:
        change = ((forecasts[i] - last_actual_price) / last_actual_price) * 100
    else:
        change = forecast_daily_changes[i-1]
    print(f"{i+1:<4} {forecasts[i]:<8.2f} {change:<7.2f}%")

if len(forecasts) > 5:
    print("...")
    for i in range(max(5, len(forecasts)-3), len(forecasts)):
        change = forecast_daily_changes[i-1]
        print(f"{i+1:<4} {forecasts[i]:<8.2f} {change:<7.2f}%")

# Generate future dates
from datetime import datetime, timedelta
import pandas as pd

last_date = pd.to_datetime('2024-12-30')
future_dates = []
current_date = last_date + timedelta(days=1)

while len(future_dates) < FORECAST_DAYS:
    # Skip weekends (Saturday=5, Sunday=6)
    if current_date.weekday() < 5:
        future_dates.append(current_date)
    current_date += timedelta(days=1)

# Comprehensive visualization
print(f"\n📊 Creating comprehensive visualizations...")

# 1. Training History
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

epochs = range(len(history.history['loss']))
ax1.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
ax1.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
ax1.set_title(f'{BANK_NAME} Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MSE Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, history.history['mae'], 'b-', label='Training MAE')
ax2.plot(epochs, history.history['val_mae'], 'r-', label='Validation MAE')
ax2.set_title(f'{BANK_NAME} Training and Validation MAE')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MAE')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Test Set: Actual vs Predicted
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Plot 1: Time series comparison (all test data)
sample_size = len(y_test_actual)
start_idx = 0

actual_sample = y_test_actual[start_idx:]
pred_sample = y_test_pred[start_idx:]
days = range(len(actual_sample))

ax1.plot(days, actual_sample, 'b-', label='Actual', linewidth=2, alpha=0.8)
ax1.plot(days, pred_sample, 'r--', label='Predicted', linewidth=2, alpha=0.8)
ax1.fill_between(days, actual_sample, pred_sample, alpha=0.2, color='gray')

ax1.set_title(f'{BANK_NAME} Test Set: Actual vs Predicted Closing Prices (All {sample_size} Days)')
ax1.set_xlabel('Days')
ax1.set_ylabel('Closing Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add performance metrics
r2_score = 1 - np.sum((y_test_actual - y_test_pred) ** 2) / np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
correlation = np.corrcoef(y_test_actual, y_test_pred)[0, 1]

ax1.text(0.02, 0.98, f'RMSE: {test_rmse:.0f}\nMAE: {test_mae:.0f}\nMAPE: {mape:.1f}%\nR²: {r2_score:.3f}',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 2: Scatter plot for correlation analysis
ax2.scatter(y_test_actual, y_test_pred, alpha=0.6, s=20, color='blue')
ax2.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()],
         'r--', linewidth=2, label='Perfect Prediction')

ax2.set_title(f'{BANK_NAME} Prediction Accuracy: Actual vs Predicted Scatter Plot')
ax2.set_xlabel('Actual Closing Price')
ax2.set_ylabel('Predicted Closing Price')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/test_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Complete timeline: Train/Val/Test + 30-day forecast
fig, ax = plt.subplots(1, 1, figsize=(20, 8))

# Prepare data for full timeline
train_dates = train_set.index
val_dates = val_set.index
test_dates_full = test_set.index  # Full test dates for timeline
test_dates_pred = test_set.index[TIME_STEPS:]  # Dates for predictions (with sequence offset)

train_prices = target_scaler.inverse_transform(train_set[target_col].values.reshape(-1, 1)).flatten()
val_prices = target_scaler.inverse_transform(val_set[target_col].values.reshape(-1, 1)).flatten()
test_actual_prices_full = target_scaler.inverse_transform(test_set[target_col].values.reshape(-1, 1)).flatten()
test_actual_prices = y_test_actual  # Only for predicted portion

# Plot historical data
ax.plot(train_dates, train_prices, 'b-', label='Train', alpha=0.7, linewidth=1)
ax.plot(val_dates, val_prices, 'orange', label='Validation', alpha=0.7, linewidth=1)
ax.plot(test_dates_full, test_actual_prices_full, 'g-', label='Test (Actual)', alpha=0.8, linewidth=1.5)
ax.plot(test_dates_pred, y_test_pred, 'r--', label='Test (Predicted)', alpha=0.8, linewidth=1.5)

# Plot 30-day forecast
ax.plot(future_dates, forecasts, color='red', linestyle='--', label=f'{FORECAST_DAYS}-Day Forecast', linewidth=2)

# Add vertical lines for splits
ax.axvline(x=pd.to_datetime('2020-01-02'), color='gray', linestyle='--', alpha=0.5, label='Train/Val Split')
ax.axvline(x=pd.to_datetime('2022-01-02'), color='gray', linestyle='--', alpha=0.5, label='Val/Test Split')
ax.axvline(x=pd.to_datetime('2024-12-30'), color='red', linestyle='--', alpha=0.7, label='Forecast Start')

ax.set_title(f'{BANK_NAME} Complete Timeline: Historical Data + {FORECAST_DAYS}-Day Forecast', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Closing Price', fontsize=12)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Rotate x-axis labels
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/complete_timeline.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Visualizations saved to {OUTPUT_FOLDER}/:")
print(f"  - training_history.png")
print(f"  - test_predictions.png")
print(f"  - complete_timeline.png")

# =============================================================================
# EXPLAINABLE AI (XAI) - PERMUTATION FEATURE IMPORTANCE
# =============================================================================

print("\n" + "="*50)
print("EXPLAINABLE AI (XAI) ANALYSIS")
print("="*50)

print("Computing Permutation Feature Importance...")
print("This analyzes which features are most important for accurate predictions.")

# Custom scoring function for LSTM regression
def lstm_scoring_function(model, X, y):
    """Custom scoring function for LSTM model evaluation"""
    try:
        predictions = model.predict(X, verbose=0)
        mse = np.mean((y - predictions.flatten()) ** 2)
        return -mse  # Return negative MSE (higher is better)
    except Exception as e:
        print(f"Error in scoring function: {e}")
        return -np.inf

# Use subset for efficiency
test_subset_size = min(200, len(X_test))
X_test_subset = X_test[:test_subset_size]
y_test_subset = y_test[:test_subset_size]

print(f"Using {test_subset_size} test samples for efficiency...")

# Calculate baseline score
baseline_score = lstm_scoring_function(model, X_test_subset, y_test_subset)
print(f"Baseline model score (negative MSE): {baseline_score:.6f}")

# Compute feature importance by permuting each feature
print(f"\nComputing feature importance by permuting each feature...")

feature_importances = []
feature_scores = []

for i, feature_name in enumerate(feature_cols):
    print(f"Processing feature {i+1}/{len(feature_cols)}: {feature_name}")

    # Create permuted version of test data
    X_permuted = X_test_subset.copy()

    # Permute feature i across all time steps and samples
    np.random.seed(42)  # For reproducibility
    original_feature = X_permuted[:, :, i].copy()
    permuted_feature = np.random.permutation(original_feature.flatten()).reshape(original_feature.shape)
    X_permuted[:, :, i] = permuted_feature

    # Calculate score with permuted feature
    permuted_score = lstm_scoring_function(model, X_permuted, y_test_subset)

    # Importance = decrease in performance
    importance = baseline_score - permuted_score
    feature_importances.append(importance)
    feature_scores.append(permuted_score)

    print(f"  Importance: {importance:.6f}")

print(f"\nPermutation Feature Importance computation completed!")

# Create feature importance dataframe
import pandas as pd
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importances,
    'Permuted_Score': feature_scores
}).sort_values('Importance', ascending=False)

print(f"\n📊 TOP 10 MOST IMPORTANT FEATURES:")
print("="*60)
print(f"{'Rank':<4} {'Feature':<35} {'Importance':<12}")
print("-" * 60)
for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
    score_drop = (row['Importance'] / abs(baseline_score)) * 100 if baseline_score != 0 else 0
    print(f"{i+1:2d}.  {row['Feature']:<35} {row['Importance']:<12.6f}")

# Feature importance visualization
print(f"\n📊 Creating XAI feature importance plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Left plot: Feature importance bar chart
y_pos = np.arange(len(feature_cols))
colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(feature_cols)))

bars = ax1.barh(y_pos, feature_importance_df['Importance'],
                alpha=0.8, color=colors, edgecolor='black', linewidth=0.5)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(feature_importance_df['Feature'])
ax1.set_xlabel('Permutation Importance (Performance Drop)', fontsize=12)
ax1.set_title(f'{BANK_NAME} LSTM Model - Feature Importance Rankings', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (bar, importance) in enumerate(zip(bars, feature_importance_df['Importance'])):
    if importance > 0.001:  # Only show significant values
        ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')

# Right plot: Top 5 features detailed
top_5_features = feature_importance_df.head(5)
score_drops = (top_5_features['Importance'] / abs(baseline_score)) * 100

bars2 = ax2.bar(range(len(top_5_features)), score_drops,
               alpha=0.7, color='lightcoral', edgecolor='darkred')

ax2.set_xticks(range(len(top_5_features)))
ax2.set_xticklabels(top_5_features['Feature'], rotation=45, ha='right')
ax2.set_ylabel('Performance Drop (%)', fontsize=11)
ax2.set_title(f'Top 5 Most Important Features', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add percentage values
for bar, pct in zip(bars2, score_drops):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/xai_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# XAI Summary and Business Insights
print(f"\n🔍 XAI INSIGHTS SUMMARY:")
print("="*50)

top_3_features = feature_importance_df.head(3)
print(f"🥇 Most Important Feature: {top_3_features.iloc[0]['Feature']}")
print(f"   Impact: {(top_3_features.iloc[0]['Importance'] / abs(baseline_score)) * 100:.2f}% performance drop when permuted")

print(f"\n🏆 Top 3 Features Driving Predictions:")
for i in range(3):
    feature = top_3_features.iloc[i]['Feature']
    impact = (top_3_features.iloc[i]['Importance'] / abs(baseline_score)) * 100
    print(f"   {i+1}. {feature} (Impact: {impact:.2f}%)")

# Feature category analysis
price_features = [f for f in feature_cols if any(keyword in f for keyword in ['Price', 'VWAP', 'Moving Average'])]
ratio_features = [f for f in feature_cols if 'Ratio' in f or 'Yld' in f]
volume_features = [f for f in feature_cols if 'Volume' in f or 'Float' in f or 'Change' in f]

price_importance = feature_importance_df[feature_importance_df['Feature'].isin(price_features)]['Importance'].mean()
ratio_importance = feature_importance_df[feature_importance_df['Feature'].isin(ratio_features)]['Importance'].mean()
volume_importance = feature_importance_df[feature_importance_df['Feature'].isin(volume_features)]['Importance'].mean()

print(f"\n📈 Feature Category Analysis:")
print(f"   Price-based features ({len(price_features)} features): avg importance {price_importance:.6f}")
print(f"   Financial ratios ({len(ratio_features)} features): avg importance {ratio_importance:.6f}")
print(f"   Volume/Market activity ({len(volume_features)} features): avg importance {volume_importance:.6f}")

# Determine most important category
categories = [('Price-based', price_importance), ('Financial ratios', ratio_importance), ('Volume/Market', volume_importance)]
most_important_category = max(categories, key=lambda x: x[1])
print(f"   👑 Most important category: {most_important_category[0]} features")

print(f"\n💡 Business Insights:")
top_feature = feature_importance_df.iloc[0]['Feature']
if any(keyword in top_feature for keyword in ['Price', 'VWAP', 'Moving Average']):
    print(f"   📊 Price-based indicators drive predictions most strongly")
elif 'Ratio' in top_feature:
    print(f"   📈 Financial ratios are key drivers of closing price changes")
elif any(keyword in top_feature for keyword in ['Volume', 'Float']):
    print(f"   🔄 Trading volume and liquidity factors are most important")

print(f"\n✅ XAI Analysis completed!")
print(f"   - Feature importance plot: xai_feature_importance.png")
print(f"   - Model interpretability: {len(feature_cols)} features analyzed")
print(f"   - Key driver: {top_feature}")

# =============================================================================
# EXCEL REPORT GENERATION
# =============================================================================

print(f"\n📊 Generating comprehensive Excel report...")

# Prepare data for Excel export
excel_file = f'{OUTPUT_FOLDER}/{BANK_NAME}_analysis_report.xlsx'

# Create comprehensive datasets with ALL data and logs
with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:

    # 1. Model Performance Summary
    performance_data = {
        'Metric': ['MSE', 'RMSE', 'MAE', 'Normalized MSE', 'Normalized RMSE', 'MAPE (%)', 'Training Epochs', 'Model Parameters'],
        'Value': [f'{test_mse:.6f}', f'{test_rmse:.6f}', f'{test_mae:.6f}', f'{normalized_mse:.6f}', f'{normalized_rmse:.6f}', f'{mape:.6f}',
                 len(history.history['loss']), model.count_params()],
        'Description': ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error',
                       'Normalized MSE (MSE/mean_price²)', 'Normalized RMSE (RMSE/mean_price)', 'Mean Absolute Percentage Error', 'Number of training epochs', 'Total model parameters']
    }
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_excel(writer, sheet_name='Model_Performance', index=False)

    # 2. Hyperparameters Used
    hyperparams_data = {
        'Parameter': ['LSTM Units', 'Learning Rate', 'Batch Size', 'Dropout Rate', 'Forecast Days', 'LSTM Layers', 'Max Epochs'],
        'Value': [optimal_params['lstm_units'], optimal_params['learning_rate'],
                 optimal_params['batch_size'], optimal_params['dropout_rate'], FORECAST_DAYS, 1, EPOCHS],
        'Description': ['Number of LSTM units', 'Adam optimizer learning rate', 'Training batch size',
                       'Dropout regularization rate', 'Number of forecast days', 'Number of LSTM layers', 'Maximum number of training epochs']
    }
    hyperparams_df = pd.DataFrame(hyperparams_data)
    hyperparams_df.to_excel(writer, sheet_name='Hyperparameters', index=False)

    # 3. Feature Importance (XAI)
    sorted_df = feature_importance_df.sort_values('Importance', ascending=False)
    xai_data = {
        'Rank': range(1, len(sorted_df) + 1),
        'Feature': sorted_df['Feature'].tolist(),
        'Importance': sorted_df['Importance'].tolist(),
        'Impact_Percent': [abs(imp / baseline_score) * 100 if baseline_score != 0 else 0 for imp in sorted_df['Importance']],
        'Category': ['Price-based' if any(kw in feat for kw in ['Price', 'VWAP', 'Average'])
                    else 'Financial Ratio' if any(kw in feat for kw in ['Ratio', 'Dividend'])
                    else 'Volume/Market' for feat in sorted_df['Feature'].tolist()]
    }
    xai_df = pd.DataFrame(xai_data)
    xai_df.to_excel(writer, sheet_name='Feature_Importance', index=False)

    # 4. Test Predictions vs Actual
    test_results_data = {
        'Date': test_dates_pred,
        'Actual_Price': y_test_actual,
        'Predicted_Price': y_test_pred,
        'Absolute_Error': np.abs(y_test_actual - y_test_pred),
        'Percentage_Error': np.abs((y_test_actual - y_test_pred) / y_test_actual) * 100,
        'Actual_Daily_Change_%': [0] + list(actual_daily_changes),
        'Predicted_Daily_Change_%': [0] + list(pred_daily_changes)
    }
    test_df = pd.DataFrame(test_results_data)
    test_df.to_excel(writer, sheet_name='Test_Predictions', index=False)

    # 5. Forecast Results
    forecast_data = {
        'Day': range(1, len(forecasts) + 1),
        'Date': future_dates,
        'Predicted_Price': forecasts,
        'Daily_Change_%': [((forecasts[0] - last_actual_price) / last_actual_price) * 100] + list(forecast_daily_changes),
        'Cumulative_Change_%': [((price - last_actual_price) / last_actual_price) * 100 for price in forecasts]
    }
    forecast_df = pd.DataFrame(forecast_data)
    forecast_df.to_excel(writer, sheet_name='Forecast_Results', index=False)

    # 6. Percentage Analysis Summary
    pct_summary_data = {
        'Analysis_Type': ['Test Period - Actual', 'Test Period - Predicted', 'Forecast Period'],
        'Daily_Change_Mean_%': [actual_daily_changes.mean(), pred_daily_changes.mean(), forecast_daily_changes.mean()],
        'Daily_Change_Std_%': [actual_daily_changes.std(), pred_daily_changes.std(), forecast_daily_changes.std()],
        'Cumulative_Change_%': [actual_cumulative, pred_cumulative, forecast_cumulative],
        'Max_Daily_Gain_%': [actual_daily_changes.max(), pred_daily_changes.max(), forecast_daily_changes.max()],
        'Max_Daily_Loss_%': [actual_daily_changes.min(), pred_daily_changes.min(), forecast_daily_changes.min()]
    }
    pct_summary_df = pd.DataFrame(pct_summary_data)
    pct_summary_df.to_excel(writer, sheet_name='Percentage_Analysis', index=False)

    # 7. Training History (Full Log)
    training_epochs = list(range(1, len(history.history['loss']) + 1))
    training_data = {
        'Epoch': training_epochs,
        'Training_Loss': history.history['loss'],
        'Validation_Loss': history.history['val_loss'],
        'Training_MAE': history.history['mae'],
        'Validation_MAE': history.history['val_mae'],
        'Learning_Rate': history.history.get('lr', [optimal_params['learning_rate']] * len(training_epochs))
    }
    training_df = pd.DataFrame(training_data)
    training_df.to_excel(writer, sheet_name='Training_History', index=False)

    # 8. Raw Data (Full Dataset Used)
    raw_data_sample = data.copy()

    # Reset index to get Date column back if it was set as index
    if 'Date' not in raw_data_sample.columns:
        raw_data_sample = raw_data_sample.reset_index()

    raw_data_sample['Date'] = pd.to_datetime(raw_data_sample['Date'])
    raw_data_sample['Split'] = 'Unknown'

    # Mark data splits
    train_start, train_end = pd.to_datetime('2014-01-02'), pd.to_datetime('2020-01-02')
    val_start, val_end = pd.to_datetime('2020-01-02'), pd.to_datetime('2022-01-02')
    test_start, test_end = pd.to_datetime('2022-01-02'), pd.to_datetime('2024-12-31')

    raw_data_sample.loc[(raw_data_sample['Date'] >= train_start) & (raw_data_sample['Date'] < train_end), 'Split'] = 'Train'
    raw_data_sample.loc[(raw_data_sample['Date'] >= val_start) & (raw_data_sample['Date'] < val_end), 'Split'] = 'Validation'
    raw_data_sample.loc[(raw_data_sample['Date'] >= test_start) & (raw_data_sample['Date'] <= test_end), 'Split'] = 'Test'

    # Add selected features only to keep manageable
    available_cols = ['Date', 'Split'] + [col for col in [target_col] + feature_cols if col in raw_data_sample.columns]
    selected_raw_data = raw_data_sample[available_cols].copy()
    selected_raw_data.to_excel(writer, sheet_name='Raw_Data_Sample', index=False)

    # 9. Configuration & Metadata
    config_data = {
        'Parameter': [
            'Bank Name', 'Target Variable', 'Number of Features', 'Time Steps (Sequence Length)',
            'Train Period', 'Validation Period', 'Test Period', 'Forecast Period',
            'Total Data Points', 'Train Samples', 'Validation Samples', 'Test Samples',
            'Optimization Used', 'Best Training Loss', 'Best Validation Loss',
            'Final Training Loss', 'Final Validation Loss', 'Early Stopping Triggered'
        ],
        'Value': [
            BANK_NAME, target_col, len(feature_cols), TIME_STEPS,
            '2014-2020 (6 years)', '2020-2022 (2 years)', '2022-2024 (2 years)', f'{FORECAST_DAYS} days',
            len(data), len(X_train), len(X_val), len(X_test),
            'Yes' if USE_OPTIMIZATION == 'y' else 'No',
            f"{min(history.history['loss']):.6f}", f"{min(history.history['val_loss']):.6f}",
            f"{history.history['loss'][-1]:.6f}", f"{history.history['val_loss'][-1]:.6f}",
            'Yes' if len(history.history['loss']) < 50 else 'No'
        ],
        'Description': [
            'Stock symbol analyzed', 'Prediction target', 'Number of input features',
            'Days of historical data per prediction', 'Training data time range',
            'Validation data time range', 'Test data time range', 'Forecast horizon',
            'Total rows in dataset', 'Training sequences created', 'Validation sequences',
            'Test sequences', 'Metaheuristic optimization', 'Lowest training loss achieved',
            'Lowest validation loss', 'Final training loss', 'Final validation loss',
            'Whether early stopping was activated'
        ]
    }
    config_df = pd.DataFrame(config_data)
    config_df.to_excel(writer, sheet_name='Configuration', index=False)

    # 10. Sample Predictions Detail (Last 30 days of test)
    sample_size = min(30, len(y_test_actual))
    sample_data = {
        'Date': test_dates_pred[-sample_size:],
        'Actual_Price': y_test_actual[-sample_size:],
        'Predicted_Price': y_test_pred[-sample_size:],
        'Absolute_Error': np.abs(y_test_actual[-sample_size:] - y_test_pred[-sample_size:]),
        'Percentage_Error': np.abs((y_test_actual[-sample_size:] - y_test_pred[-sample_size:]) / y_test_actual[-sample_size:]) * 100,
        'Prediction_Direction': ['Up' if y_test_pred[-sample_size:][i] > y_test_pred[-sample_size:][i-1] else 'Down' if i > 0 else 'Start'
                               for i in range(sample_size)],
        'Actual_Direction': ['Up' if y_test_actual[-sample_size:][i] > y_test_actual[-sample_size:][i-1] else 'Down' if i > 0 else 'Start'
                           for i in range(sample_size)]
    }
    sample_detail_df = pd.DataFrame(sample_data)
    sample_detail_df.to_excel(writer, sheet_name='Sample_Predictions_Detail', index=False)

    # Auto-adjust column widths for all sheets
    for sheet_name in writer.sheets:
        worksheet = writer.sheets[sheet_name]

        # Get the dataframe for this sheet
        if sheet_name == 'Model_Performance':
            df = performance_df
        elif sheet_name == 'Hyperparameters':
            df = hyperparams_df
        elif sheet_name == 'Feature_Importance':
            df = xai_df
        elif sheet_name == 'Test_Predictions':
            df = test_df
        elif sheet_name == 'Forecast_Results':
            df = forecast_df
        elif sheet_name == 'Percentage_Analysis':
            df = pct_summary_df
        elif sheet_name == 'Training_History':
            df = training_df
        elif sheet_name == 'Raw_Data_Sample':
            df = selected_raw_data
        elif sheet_name == 'Configuration':
            df = config_df
        elif sheet_name == 'Sample_Predictions_Detail':
            df = sample_detail_df
        else:
            continue  # Skip unknown sheets

        # Calculate column widths with special handling for dates and numbers (wider)
        for i, col in enumerate(df.columns):
            # Special handling for Date columns
            if 'Date' in str(col) or df[col].dtype == 'datetime64[ns]':
                worksheet.set_column(i, i, 22)  # Wider for dates
                # Format date columns
                date_format = writer.book.add_format({'num_format': 'yyyy-mm-dd'})
                worksheet.set_column(i, i, 22, date_format)
            # Special handling for numeric columns
            elif df[col].dtype in ['float64', 'int64'] and col not in ['Rank', 'Day', 'Epoch', 'Value']:
                # Format numbers with appropriate decimal places (wider columns)
                if 'Percentage' in col or '%' in col or 'Change' in col:
                    num_format = writer.book.add_format({'num_format': '0.000000'})
                    worksheet.set_column(i, i, 18, num_format)
                elif 'Price' in col or 'Error' in col:
                    num_format = writer.book.add_format({'num_format': '#,##0.000000'})
                    worksheet.set_column(i, i, 20, num_format)
                elif 'Loss' in col or 'MAE' in col or 'MSE' in col or 'RMSE' in col or 'Importance' in col or 'Normalized' in col:
                    num_format = writer.book.add_format({'num_format': '0.000000'})
                    worksheet.set_column(i, i, 18, num_format)
                else:
                    num_format = writer.book.add_format({'num_format': '#,##0'})
                    worksheet.set_column(i, i, 14, num_format)
            else:
                # Calculate the maximum width needed for text columns (wider)
                max_width = max(
                    df[col].astype(str).map(len).max(),  # max width of values
                    len(str(col))  # width of column name
                )
                # Set column width with more padding (wider)
                worksheet.set_column(i, i, min(max_width + 6, 50))

print(f"✅ Excel report generated: {excel_file}")
print(f"   📊 10 comprehensive sheets:")
print(f"      1. Model_Performance - Metrics & accuracy")
print(f"      2. Hyperparameters - Model configuration")
print(f"      3. Feature_Importance - XAI analysis")
print(f"      4. Test_Predictions - Full test results")
print(f"      5. Forecast_Results - Future predictions")
print(f"      6. Percentage_Analysis - Daily & cumulative changes")
print(f"      7. Training_History - Complete training log")
print(f"      8. Raw_Data_Sample - Original data with splits")
print(f"      9. Configuration - Pipeline metadata")
print(f"      10. Sample_Predictions_Detail - Last 30 days analysis")

print(f"\n🎯 Complete Pipeline finished successfully!")
print(f"Model trained on {BANK_NAME} closing price prediction with full explainability")
print(f"Test RMSE: {test_rmse:.2f} | Test MAE: {test_mae:.2f}")
print(f"📁 All outputs saved to: {OUTPUT_FOLDER}/")
print(f"✅ All steps completed: Training + Evaluation + Forecasting + XAI + Reporting!")