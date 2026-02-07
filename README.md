# Predictive Maintenance Project - Anomaly Detection in Hydroelectric Power Plants

## 📋 Description

This project aims to develop an anomaly detection system for valve operations in hydroelectric power plants. It focuses on analyzing valve closing sequences and detecting abnormal behaviors using advanced machine learning techniques.

The project is structured into two main tasks:
- **Task 1**: Data preprocessing and determination of valve closing/opening times using a TCN (Temporal Convolutional Network)
- **Task 2**: Anomaly detection with Autoencoder and classification of anomaly types using HDBSCAN

## 🏭 Industrial Context

The data comes from a hydroelectric power plant (KSL) with:
- **3 machine groups**: MG1, MG2, MG3
- **2 stages**: Mapragg and Sarelli
- **Measured signals**:
  - Active power (MW)
  - Ball valve position (open/closed)
  - Guide vane position (%)
  - Water pressure upstream and downstream (bar)

## 📁 Project Structure

```
.
├── GroupA_Task1.ipynb              # Data preprocessing and analysis
├── GroupA_Task2.ipynb              # Autoencoder and HDBSCAN for anomaly detection
├── GroupA_anomaliesGeneration.py   # Synthetic anomaly generation library
├── GroupA_Report.pdf               # Detailed project report
└── README.md                       # This file
```

## 🔧 Task 1: Preprocessing and Analysis

### Objectives
1. **Data preprocessing**:
   - Temporal signal synchronization
   - Gap detection and handling in data
   - Time series segmentation
   - Smoothing with exponential moving average (EMA)

2. **Transition detection**:
   - Identification of valve opening/closing events
   - Extraction of temporal windows around transitions

3. **Closing/opening time determination**:
   - Use of a TCN (Temporal Convolutional Network)
   - Accurate prediction of transition durations

4. **Anomaly detection**:
   - Analysis of closing sequences to identify abnormal behaviors

### Main Parameters
```python
GAP_THRESHOLD_SECONDS = 3600   # Threshold for segmentation (1 hour)
MIN_POINTS_PER_SEGMENT = 100   # Minimum number of points per segment
EMA_ALPHA = 0.1                # EMA smoothing factor
```

### Key Features
- **Gap analysis**: Identification of interruptions in data
- **Segmentation**: Division of time series into continuous segments
- **Temporal normalization**: Alignment of signals on a uniform temporal grid
- **Transition detection**: Automatic identification of valve state changes

## 🤖 Task 2: Anomaly Detection with Autoencoder

### Objectives
1. **Window extraction**:
   - 360-second windows (180 before + 180 after) centered on closing transitions
   - Separation of turbine regime (power > 0) and pump regime (power ≤ 0)

2. **Autoencoder training**:
   - Separate autoencoder for each regime (turbine/pump)
   - Dimension reduction and reconstruction of normal sequences
   - Calculation of reconstruction errors as anomaly score

3. **Anomaly type classification**:
   - Use of HDBSCAN for anomaly clustering
   - Joint probability estimation of anomaly types
   - Identification of recurring anomaly patterns

### Architecture
- **Training data**: Normal closing windows
- **Test data**: Normal and abnormal windows
- **Metric**: Reconstruction error (MSE) to detect anomalies

## 🧪 Synthetic Anomaly Generation

The `GroupA_anomaliesGeneration.py` module provides a comprehensive library for generating synthetic anomalies in valve closing sequences.

### Implemented Anomaly Types

1. **Spikes**: `inject_closing_spikes`
   - Isolated spikes in the closing sequence
   - Configurable amplitude in multiples of local standard deviation

2. **Level Shift**: `inject_closing_level_shift`
   - Constant mean shift over a segment
   - Simulates a sudden regime change

3. **Linear Drift**: `inject_closing_linear_drift`
   - Progressive linear drift over a segment
   - Simulates gradual degradation

4. **Variance Change**: `inject_closing_variance_change`
   - Increase or decrease in volatility
   - Simulates noise bursts or damping

5. **Sinusoidal**: `inject_closing_sinusoidal`
   - Added periodic oscillation
   - Simulates mechanical vibrations or resonances

6. **Delayed Closure**: `inject_closing_delayed_closure`
   - Temporal shift of the closing sequence
   - Simulates mechanical or control delays

7. **Water Hammer Spike**: `inject_closing_water_hammer_spike`
   - Amplification of an existing peak
   - Simulates dangerous pressure spikes

8. **Signal Dropout**: `inject_closing_signal_dropout`
   - Temporary signal loss (values set to zero)
   - Simulates sensor failures or communication issues

9. **Time Warp**: `inject_closing_time_warp`
   - Acceleration or deceleration of the sequence
   - Simulates closing that is too fast or too slow

### Features
- All anomalies are injected only in the **closing sequence** (indices [180, 360))
- Placement biased towards transition center (around index 200)
- Configurable parameters for each anomaly type
- Reproducibility via `random_state`

## 📊 Data

### Data Format
- **Input format**: Parquet files with columns:
  - `ts`: Timestamp
  - `signal_id`: Signal identifier
  - `value`: Measured value

### Available Signals
- `active_power`: Active power (MW)
- `ball_valve_open`: Valve open (boolean)
- `ball_valve_closed`: Valve closed (boolean)
- `guide_vane_position`: Guide vane position (%)
- `water_pressure_upstream`: Upstream pressure (bar)
- `water_pressure_downstream`: Downstream pressure (bar)

## 🚀 Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch scipy hdbscan optuna tqdm pyarrow
```

### Running Task 1
1. Open `GroupA_Task1.ipynb`
2. Configure preprocessing parameters
3. Execute cells to:
   - Load and map signals
   - Preprocess data
   - Extract transitions
   - Train TCN model
   - Detect anomalies

### Running Task 2
1. Open `GroupA_Task2.ipynb`
2. Configure data paths (`DATA_DIR`, `OUTPUT_DIR`)
3. Execute cells to:
   - Preprocess data and extract windows
   - Train autoencoders (turbine and pump)
   - Apply HDBSCAN for classification
   - Evaluate performance

### Anomaly Generation
```python
from GroupA_anomaliesGeneration import inject_closing_spikes, inject_closing_level_shift

# Example: Injecting spikes
window_perturbed, spike_indices = inject_closing_spikes(
    window=normal_window,
    n_spikes=5,
    magnitude_range=(2.0, 5.0),
    random_state=42
)

# Example: Injecting level shift
window_shifted, (start, end), shift = inject_closing_level_shift(
    window=normal_window,
    segment_length=50,
    shift_factor=3.0,
    random_state=42
)
```

## 📈 Results

The project enables:
- ✅ Efficient preprocessing of industrial sensor data
- ✅ Automatic detection of valve transitions
- ✅ Accurate prediction of closing/opening durations
- ✅ Identification of anomalies in closing sequences
- ✅ Classification of detected anomaly types
- ✅ Generation of synthetic anomalies for data augmentation

## 📝 Technical Notes

### Temporal Windows
- **Size**: 360 seconds (180 before + 180 after transition)
- **Centering**: On valve closing events
- **Normalization**: Standardization (mean=0, std=1)

### Operational Regimes
- **Turbine**: `active_power > 0` (electricity production)
- **Pump**: `active_power ≤ 0` (pumping)

### Gap Handling
- Forward fill up to 5 minutes
- Longer gaps left as NaN
- Automatic segmentation on gaps > 1 hour

## 👥 Authors

Group A - EPFL MA3 - Machine Learning for Predictive Maintenance

## 📄 License

This project is developed as part of an academic course at EPFL.

## 🔗 References

- Detailed report: `GroupA_Report.pdf`
- Notebook documentation: See comments in cells
