"""
Neural Network Models for Touch Classification and Pressure Regression
CNN + LSTM architecture for time-series capacitance data analysis
Using PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import os


class CNNLSTMClassifier(nn.Module):
    """
    CNN + LSTM model for touch classification
    Predicts whether each of 4 sensors is touched (binary classification)
    """
    
    def __init__(self, sequence_length, delta_dim, feature_dim=4, num_sensors=4):
        """
        Initialize the classification model
        
        Args:
            sequence_length: Length of time sequence
            delta_dim: Dimension of delta values
            feature_dim: Dimension of additional features (4: max, max_pos, 2nd_max, 2nd_max_pos)
            num_sensors: Number of pressure sensors (4)
        """
        super(CNNLSTMClassifier, self).__init__()
        
        self.sequence_length = sequence_length
        self.delta_dim = delta_dim
        self.feature_dim = feature_dim
        self.num_sensors = num_sensors
        
        # CNN layers for Deltas
        self.conv1 = nn.Conv1d(in_channels=delta_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Feature processing
        self.feature_fc = nn.Linear(feature_dim, 64)
        self.feature_bn = nn.BatchNorm1d(sequence_length)
        
        # Calculate pooled length
        self.pooled_length = sequence_length // 4  # Two MaxPool1d with kernel_size=2
        
        # Feature pooling to match CNN output
        if sequence_length != self.pooled_length:
            self.feature_pool = nn.AvgPool1d(kernel_size=sequence_length // self.pooled_length)
        else:
            self.feature_pool = None
        
        # LSTM layers
        combined_dim = 256 + 64  # CNN output + feature output
        self.lstm1 = nn.LSTM(input_size=combined_dim, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        
        # Dense layers
        self.fc1 = nn.Linear(64, 128)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        
        # Output layer
        self.output = nn.Linear(64, num_sensors)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, deltas, features):
        """
        Forward pass
        
        Args:
            deltas: (batch_size, sequence_length, delta_dim)
            features: (batch_size, sequence_length, feature_dim)
            
        Returns:
            output: (batch_size, num_sensors)
        """
        batch_size = deltas.size(0)
        
        # CNN processing of deltas
        # Reshape: (batch, seq, delta_dim) -> (batch, delta_dim, seq)
        x = deltas.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Reshape back: (batch, channels, seq) -> (batch, seq, channels)
        x = x.permute(0, 2, 1)
        
        # Feature processing
        f = self.feature_fc(features)
        f = self.feature_bn(f)
        f = self.relu(f)
        
        # Pool features to match CNN output length
        if self.feature_pool is not None:
            # (batch, seq, feat_dim) -> (batch, feat_dim, seq) -> pool -> (batch, feat_dim, pooled_seq)
            f = f.permute(0, 2, 1)
            f = self.feature_pool(f)
            f = f.permute(0, 2, 1)
        
        # Concatenate
        combined = torch.cat([x, f], dim=2)
        
        # LSTM processing
        lstm_out, _ = self.lstm1(combined)
        lstm_out = self.dropout1(lstm_out)
        
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout2(lstm_out)
        
        # Take the last time step output
        lstm_out = lstm_out[:, -1, :]
        
        # Dense layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        
        # Output
        out = self.output(out)
        out = self.sigmoid(out)
        
        return out


class CNNLSTMRegressor(nn.Module):
    """
    CNN + LSTM model for pressure regression
    Predicts actual pressure values for 4 sensors
    Now includes touch state information to handle multi-touch scenarios
    """
    
    def __init__(self, sequence_length, delta_dim, num_sensors=4):
        """
        Initialize the regression model
        
        Args:
            sequence_length: Length of time sequence
            delta_dim: Dimension of delta values
            num_sensors: Number of pressure sensors (4)
        """
        super(CNNLSTMRegressor, self).__init__()
        
        self.sequence_length = sequence_length
        self.delta_dim = delta_dim
        self.num_sensors = num_sensors
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=delta_dim, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Touch state processing (multi-touch pattern information)
        # Input: (batch, seq, num_sensors) binary touch states
        self.touch_fc = nn.Linear(num_sensors, 32)
        self.touch_bn = nn.BatchNorm1d(32)  # Normalize feature dimension (32)
        
        # Calculate pooled length
        self.pooled_length = sequence_length // 4  # Two MaxPool1d with kernel_size=2
        
        # Touch state pooling to match CNN output
        if sequence_length != self.pooled_length:
            self.touch_pool = nn.AvgPool1d(kernel_size=sequence_length // self.pooled_length)
        else:
            self.touch_pool = None
        
        # LSTM layers
        combined_dim = 256 + 32  # CNN output + touch state features
        self.lstm1 = nn.LSTM(input_size=combined_dim, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        
        # Dense layers
        self.fc1 = nn.Linear(64, 128)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        
        # Output layer
        self.output = nn.Linear(64, num_sensors)
        
        self.relu = nn.ReLU()
        
    def forward(self, deltas, touch_states):
        """
        Forward pass
        
        Args:
            deltas: (batch_size, sequence_length, delta_dim)
            touch_states: (batch_size, sequence_length, num_sensors) - binary touch information
            
        Returns:
            output: (batch_size, num_sensors)
        """
        # CNN processing
        # Reshape: (batch, seq, delta_dim) -> (batch, delta_dim, seq)
        x = deltas.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Reshape back: (batch, channels, seq) -> (batch, seq, channels)
        x = x.permute(0, 2, 1)
        
        # Touch state processing
        # Input: (batch, seq, num_sensors)
        t = self.touch_fc(touch_states)  # (batch, seq, 32)
        
        # Permute for BatchNorm: (batch, seq, 32) -> (batch, 32, seq)
        t = t.permute(0, 2, 1)
        t = self.touch_bn(t)  # Normalize along feature dimension
        t = self.relu(t)
        
        # Pool touch states to match CNN output length
        if self.touch_pool is not None:
            # (batch, 32, seq) -> pool -> (batch, 32, pooled_seq)
            t = self.touch_pool(t)
        
        # Permute back: (batch, 32, pooled_seq) -> (batch, pooled_seq, 32)
        t = t.permute(0, 2, 1)
        
        # Concatenate CNN output with touch state features
        combined = torch.cat([x, t], dim=2)
        
        # LSTM processing
        lstm_out, _ = self.lstm1(combined)
        lstm_out = self.dropout1(lstm_out)
        
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout2(lstm_out)
        
        # Take the last time step output
        lstm_out = lstm_out[:, -1, :]
        
        # Dense layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        
        # Output (no activation for regression)
        out = self.output(out)
        
        return out


class ClassificationTrainer:
    """Trainer class for classification model"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
    def train(self, train_data, validation_split=0.2, epochs=200, batch_size=32, patience=15):
        """
        Train the classification model with early stopping
        
        Args:
            train_data: Dictionary with 'sequences', 'features', 'touch_labels'
            validation_split: Fraction of data for validation
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience (number of epochs without improvement)
            
        Returns:
            Training history
        """
        # Convert to tensors
        sequences = torch.FloatTensor(train_data['sequences'])
        features = torch.FloatTensor(train_data['features'])
        labels = torch.FloatTensor(train_data['touch_labels'])
        
        # Split into train and validation
        n_val = int(len(sequences) * validation_split)
        indices = torch.randperm(len(sequences))
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        train_dataset = TensorDataset(sequences[train_idx], features[train_idx], labels[train_idx])
        val_dataset = TensorDataset(sequences[val_idx], features[val_idx], labels[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training history
        history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        print(f"Training on device: {self.device}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_deltas, batch_features, batch_labels in train_loader:
                batch_deltas = batch_deltas.to(self.device)
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_deltas, batch_features)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == batch_labels).sum().item()
                train_total += batch_labels.numel()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_deltas, batch_features, batch_labels in val_loader:
                    batch_deltas = batch_deltas.to(self.device)
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_deltas, batch_features)
                    loss = self.criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    
                    predictions = (outputs > 0.5).float()
                    val_correct += (predictions == batch_labels).sum().item()
                    val_total += batch_labels.numel()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            # Update history
            history['loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            
            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def predict(self, sequences, features, threshold=0.5):
        """Make predictions"""
        self.model.eval()
        
        sequences = torch.FloatTensor(sequences).to(self.device)
        features = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(sequences, features)
            predictions = (outputs > threshold).float()
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, sequences, features):
        """Get prediction probabilities"""
        self.model.eval()
        
        sequences = torch.FloatTensor(sequences).to(self.device)
        features = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(sequences, features)
        
        return outputs.cpu().numpy()
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Classification model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Classification model loaded from {filepath}")


class RegressionTrainer:
    """Trainer class for regression model"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
    def train(self, train_data, validation_split=0.2, epochs=150, batch_size=32, patience=20):
        """
        Train the regression model with early stopping
        
        Args:
            train_data: Dictionary with 'sequences', 'touch_sequences', 'pressure_values'
            validation_split: Fraction of data for validation
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience (number of epochs without improvement)
            
        Returns:
            Training history
        """
        # Convert to tensors
        sequences = torch.FloatTensor(train_data['sequences'])
        touch_sequences = torch.FloatTensor(train_data['touch_sequences'])  # Use full sequences
        labels = torch.FloatTensor(train_data['pressure_values'])
        
        # Split into train and validation
        n_val = int(len(sequences) * validation_split)
        indices = torch.randperm(len(sequences))
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        train_dataset = TensorDataset(sequences[train_idx], touch_sequences[train_idx], labels[train_idx])
        val_dataset = TensorDataset(sequences[val_idx], touch_sequences[val_idx], labels[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training history
        history = {
            'loss': [],
            'val_loss': [],
            'mae': [],
            'val_mae': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        print(f"Training on device: {self.device}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            for batch_deltas, batch_touch_labels, batch_labels in train_loader:
                batch_deltas = batch_deltas.to(self.device)
                batch_touch_labels = batch_touch_labels.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_deltas, batch_touch_labels)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - batch_labels)).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_train_mae = train_mae / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for batch_deltas, batch_touch_labels, batch_labels in val_loader:
                    batch_deltas = batch_deltas.to(self.device)
                    batch_touch_labels = batch_touch_labels.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_deltas, batch_touch_labels)
                    loss = self.criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(outputs - batch_labels)).item()
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mae = val_mae / len(val_loader)
            
            # Update history
            history['loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['mae'].append(avg_train_mae)
            history['val_mae'].append(avg_val_mae)
            
            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f} - "
                      f"Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def predict(self, sequences, touch_sequences):
        """Make predictions"""
        self.model.eval()
        
        sequences = torch.FloatTensor(sequences).to(self.device)
        touch_sequences = torch.FloatTensor(touch_sequences).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(sequences, touch_sequences)
        
        return outputs.cpu().numpy()
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Regression model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Regression model loaded from {filepath}")


def evaluate_classification_model(y_true, y_pred):
    """
    Evaluate classification model performance
    
    Args:
        y_true: True labels (N, 4)
        y_pred: Predicted labels (N, 4)
        
    Returns:
        Dictionary with evaluation metrics for each sensor
    """
    results = {}
    sensor_names = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4']
    
    for i, sensor_name in enumerate(sensor_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        report = classification_report(y_true[:, i], y_pred[:, i], output_dict=True, zero_division=0)
        
        results[sensor_name] = {
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    return results


def evaluate_regression_model(y_true, y_pred):
    """
    Evaluate regression model performance
    
    Args:
        y_true: True pressure values (N, 4)
        y_pred: Predicted pressure values (N, 4)
        
    Returns:
        Dictionary with evaluation metrics for each sensor
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {}
    sensor_names = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4']
    
    for i, sensor_name in enumerate(sensor_names):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
        
        results[sensor_name] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Correlation': correlation
        }
    
    return results
