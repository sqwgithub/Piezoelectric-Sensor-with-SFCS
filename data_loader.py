"""
Data Loader Module for Capacitance and Pressure Sensor Data
Handles loading, preprocessing, and feature extraction
"""

import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split


def parse_list_column(value):
    """Parse string representation of list to actual list"""
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def extract_features(deltas):
    """
    Extract additional features from Deltas:
    1. Maximum value
    2. Position of maximum value (index)
    3. Second maximum value
    4. Position of second maximum value
    """
    deltas_array = np.array(deltas)
    
    # Find max value and its position
    max_val = np.max(deltas_array)
    max_pos = np.argmax(deltas_array)
    
    # Find second max value and its position
    sorted_indices = np.argsort(deltas_array)[::-1]
    second_max_val = deltas_array[sorted_indices[1]]
    second_max_pos = sorted_indices[1]
    
    return max_val, max_pos, second_max_val, second_max_pos


def load_data(train_path, eval_path, sequence_length=10):
    """
    Load and preprocess capacitance data for both classification and regression tasks
    
    Args:
        train_path: Path to training CSV file
        eval_path: Path to evaluation CSV file
        sequence_length: Length of time sequence for LSTM
        
    Returns:
        Dictionary containing processed data for both tasks
    """
    # Load datasets
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Evaluation data shape: {eval_df.shape}")
    
    # Parse Deltas column
    train_df['Deltas'] = train_df['Deltas'].apply(parse_list_column)
    eval_df['Deltas'] = eval_df['Deltas'].apply(parse_list_column)
    
    # Extract features for each sample
    train_features = []
    eval_features = []
    
    for deltas in train_df['Deltas']:
        max_val, max_pos, second_max_val, second_max_pos = extract_features(deltas)
        train_features.append([max_val, max_pos, second_max_val, second_max_pos])
    
    for deltas in eval_df['Deltas']:
        max_val, max_pos, second_max_val, second_max_pos = extract_features(deltas)
        eval_features.append([max_val, max_pos, second_max_val, second_max_pos])
    
    train_features = np.array(train_features)
    eval_features = np.array(eval_features)
    
    # Prepare Deltas as numpy arrays
    train_deltas = np.array([np.array(d) for d in train_df['Deltas']])
    eval_deltas = np.array([np.array(d) for d in eval_df['Deltas']])
    
    # Prepare labels for classification (touch detection: pressure >= 100)
    pressure_cols = ['pressuresensor1', 'pressuresensor2', 'pressuresensor3', 'pressuresensor4']
    train_touch_labels = (train_df[pressure_cols].values >= 100).astype(int)
    eval_touch_labels = (eval_df[pressure_cols].values >= 100).astype(int)
    
    # Prepare labels for regression (actual pressure values)
    train_pressure_values = train_df[pressure_cols].values
    eval_pressure_values = eval_df[pressure_cols].values
    
    # Create sequences for LSTM
    train_sequences, train_seq_features, train_seq_touch, train_seq_pressure, train_seq_touch_full = create_sequences(
        train_deltas, train_features, train_touch_labels, train_pressure_values, sequence_length
    )
    
    # For evaluation, keep time structure - don't shuffle
    eval_sequences, eval_seq_features, eval_seq_touch, eval_seq_pressure, eval_seq_touch_full = create_sequences(
        eval_deltas, eval_features, eval_touch_labels, eval_pressure_values, sequence_length
    )
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    
    # Fit scaler on training data
    scaler_deltas = StandardScaler()
    scaler_features = StandardScaler()
    
    # Reshape for scaling
    train_seq_flat = train_sequences.reshape(-1, train_sequences.shape[-1])
    scaler_deltas.fit(train_seq_flat)
    scaler_features.fit(train_seq_features.reshape(-1, train_seq_features.shape[-1]))
    
    # Transform data
    train_sequences_scaled = scaler_deltas.transform(
        train_sequences.reshape(-1, train_sequences.shape[-1])
    ).reshape(train_sequences.shape)
    
    eval_sequences_scaled = scaler_deltas.transform(
        eval_sequences.reshape(-1, eval_sequences.shape[-1])
    ).reshape(eval_sequences.shape)
    
    train_seq_features_scaled = scaler_features.transform(
        train_seq_features.reshape(-1, train_seq_features.shape[-1])
    ).reshape(train_seq_features.shape)
    
    eval_seq_features_scaled = scaler_features.transform(
        eval_seq_features.reshape(-1, eval_seq_features.shape[-1])
    ).reshape(eval_seq_features.shape)
    
    data = {
        'train': {
            'sequences': train_sequences_scaled,
            'features': train_seq_features_scaled,
            'touch_labels': train_seq_touch,  # For classification (N, 4)
            'touch_sequences': train_seq_touch_full,  # For regression (N, seq_length, 4)
            'pressure_values': train_seq_pressure,
            'raw_deltas': train_deltas,
            'raw_pressure': train_pressure_values,
            'raw_touch': train_touch_labels
        },
        'eval': {
            'sequences': eval_sequences_scaled,
            'features': eval_seq_features_scaled,
            'touch_labels': eval_seq_touch,  # For classification (N, 4)
            'touch_sequences': eval_seq_touch_full,  # For regression (N, seq_length, 4)
            'pressure_values': eval_seq_pressure,
            'raw_deltas': eval_deltas,
            'raw_pressure': eval_pressure_values,
            'raw_touch': eval_touch_labels
        },
        'scalers': {
            'deltas': scaler_deltas,
            'features': scaler_features
        },
        'sequence_length': sequence_length
    }
    
    print(f"\nSequence data shapes:")
    print(f"Train sequences: {train_sequences_scaled.shape}")
    print(f"Train features: {train_seq_features_scaled.shape}")
    print(f"Train touch labels: {train_seq_touch.shape}")
    print(f"Train pressure values: {train_seq_pressure.shape}")
    print(f"Eval sequences: {eval_sequences_scaled.shape}")
    print(f"Eval features: {eval_seq_features_scaled.shape}")
    
    return data


def create_sequences(deltas, features, touch_labels, pressure_values, seq_length):
    """
    Create time sequences for LSTM input
    
    Args:
        deltas: Array of delta values (N, delta_dim)
        features: Array of extracted features (N, 4)
        touch_labels: Array of touch labels (N, 4)
        pressure_values: Array of pressure values (N, 4)
        seq_length: Length of sequence
        
    Returns:
        Sequences for LSTM input
    """
    sequences = []
    seq_features = []
    seq_touch_labels = []  # Full sequence for regression model
    seq_touch_single = []  # Single label for classification model
    seq_pressure = []
    
    for i in range(len(deltas) - seq_length + 1):
        sequences.append(deltas[i:i+seq_length])
        seq_features.append(features[i:i+seq_length])
        seq_touch_labels.append(touch_labels[i:i+seq_length])  # Full sequence for regression
        seq_touch_single.append(touch_labels[i+seq_length-1])  # Single label for classification
        seq_pressure.append(pressure_values[i+seq_length-1])  # Use last item as label
    
    return (
        np.array(sequences),
        np.array(seq_features),
        np.array(seq_touch_single),  # For classification (N, 4)
        np.array(seq_pressure),
        np.array(seq_touch_labels)   # For regression (N, seq_length, 4)
    )


def get_data_statistics(data):
    """Calculate and return statistics about the dataset"""
    train_pressure = data['train']['raw_pressure']
    eval_pressure = data['eval']['raw_pressure']
    
    stats = {
        'train_samples': len(train_pressure),
        'eval_samples': len(eval_pressure),
        'train_touch_count': np.sum(train_pressure >= 100, axis=0),
        'train_no_touch_count': np.sum(train_pressure < 100, axis=0),
        'eval_touch_count': np.sum(eval_pressure >= 100, axis=0),
        'eval_no_touch_count': np.sum(eval_pressure < 100, axis=0),
        'train_pressure_mean': np.mean(train_pressure, axis=0),
        'train_pressure_std': np.std(train_pressure, axis=0),
        'eval_pressure_mean': np.mean(eval_pressure, axis=0),
        'eval_pressure_std': np.std(eval_pressure, axis=0),
    }
    
    return stats
