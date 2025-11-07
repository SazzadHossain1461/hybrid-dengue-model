"""
Hybrid Dengue Prediction Model (TensorFlow Only)
===============================
Combines TensorFlow DNN (supervised learning) with RL-style optimization
for improved performance on the dengue prediction task.

NO PYTORCH REQUIRED - Uses only TensorFlow & Scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import os
from dataclasses import dataclass

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============ CONFIG ============
@dataclass
class HybridConfig:
    # Data
    dataset_path: str = 'dataset.csv'
    test_size: float = 0.2
    val_size: float = 0.2
    
    # Supervised Phase (DNN)
    dnn_epochs: int = 100
    dnn_batch_size: int = 32
    dnn_learning_rate: float = 0.001
    
    # RL Phase
    rl_epochs: int = 20
    rl_batch_size: int = 16
    rl_learning_rate: float = 2e-5
    rl_kl_beta: float = 0.02
    
    # Ensemble
    ensemble_mode: str = 'average'

cfg = HybridConfig()

# ============ PHASE 1: DATA PREPROCESSING ============
class DataPreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.preprocessor = None
        
    def load_and_split(self):
        """Load, preprocess, and split data"""
        print("[PHASE 1] Loading and preprocessing data...")
        
        df = pd.read_csv(self.cfg.dataset_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Outcome distribution:\n{df['Outcome'].value_counts()}\n")
        
        # Separate features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Define feature types
        numerical_features = ['Age', 'NS1', 'IgG', 'IgM']
        categorical_features = ['Gender', 'Area', 'AreaType', 'HouseType', 'District']
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Split: train (60%), val (20%), test (20%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Preprocess
        X_train = self.preprocessor.fit_transform(X_train).toarray()
        X_val = self.preprocessor.transform(X_val).toarray()
        X_test = self.preprocessor.transform(X_test).toarray()
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\n")
        
        return X_train, X_val, X_test, y_train.values, y_val.values, y_test.values

# ============ PHASE 2: SUPERVISED DNN ============
class SupervisedDNN:
    def __init__(self, input_dim, cfg):
        self.cfg = cfg
        self.model = self._build_model(input_dim)
        self.history = None
        
    def _build_model(self, input_dim):
        """Build TensorFlow model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.cfg.dnn_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train supervised model"""
        print("[PHASE 2] Training Supervised DNN...\n")
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.cfg.dnn_epochs,
            batch_size=self.cfg.dnn_batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        print("\n✓ Supervised training complete\n")
    
    def predict_proba(self, X):
        """Get probability predictions"""
        return self.model.predict(X, verbose=0).flatten()

# ============ PHASE 3: RL FINE-TUNING ============
class RLFineTuner:
    def __init__(self, tf_model, cfg):
        self.tf_model = tf_model
        self.cfg = cfg
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.rl_learning_rate)
        
    def compute_reward(self, y_true, y_pred_proba):
        """
        Reward function: high reward for correct predictions with high confidence
        """
        y_pred = (y_pred_proba > 0.5).astype(int)
        correct = (y_pred == y_true).astype(float)
        confidence = np.abs(y_pred_proba - 0.5) * 2  # 0-1 scale
        reward = correct * (0.5 + 0.5 * confidence)
        return reward
    
    def kl_penalty(self, y_pred_proba_old, y_pred_proba_new):
        """KL divergence penalty to prevent large policy shifts"""
        epsilon = 1e-7
        old_probs = np.clip(y_pred_proba_old, epsilon, 1 - epsilon)
        new_probs = np.clip(y_pred_proba_new, epsilon, 1 - epsilon)
        
        kl = old_probs * (np.log(old_probs) - np.log(new_probs))
        return np.mean(kl)
    
    def train(self, X_train, y_train, initial_proba):
        """Fine-tune model with RL objective"""
        print("[PHASE 3] RL Fine-tuning...\n")
        
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(self.cfg.rl_epochs):
            epoch_losses = []
            
            # Mini-batch training
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            
            for i in range(0, len(X_train), self.cfg.rl_batch_size):
                batch_idx = indices[i:i + self.cfg.rl_batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                with tf.GradientTape() as tape:
                    # Forward pass
                    y_pred_new = self.tf_model(X_batch, training=True)
                    y_pred_new = tf.squeeze(y_pred_new, axis=1)
                    
                    # Compute reward
                    y_pred_np = y_pred_new.numpy()
                    reward = self.compute_reward(y_batch, y_pred_np)
                    reward_tensor = tf.constant(reward, dtype=tf.float32)
                    
                    # Policy gradient loss: maximize reward
                    policy_loss = -tf.reduce_mean(
                        y_pred_new * reward_tensor + (1 - y_pred_new) * (1 - reward_tensor)
                    )
                    
                    # KL penalty
                    kl_value = self.kl_penalty(initial_proba[batch_idx], y_pred_np)
                    kl_tensor = tf.constant(kl_value, dtype=tf.float32)
                    
                    total_loss = policy_loss + self.cfg.rl_kl_beta * kl_tensor
                
                # Backward pass
                gradients = tape.gradient(total_loss, self.tf_model.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.tf_model.trainable_weights))
                epoch_losses.append(total_loss.numpy())
            
            avg_loss = np.mean(epoch_losses)
            print(f"RL Epoch {epoch + 1}/{self.cfg.rl_epochs} - Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("RL training converged early\n")
                    break
        
        print("✓ RL fine-tuning complete\n")

# ============ PHASE 4: ENSEMBLE & EVALUATION ============
class HybridEnsemble:
    def __init__(self, dnn_model, cfg):
        self.dnn_model = dnn_model
        self.cfg = cfg
        
    def predict(self, X):
        """Generate ensemble predictions"""
        dnn_pred = self.dnn_model.predict_proba(X)
        return dnn_pred
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        print("="*60)
        print("HYBRID MODEL EVALUATION")
        print("="*60 + "\n")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Hybrid Model - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('hybrid_confusion_matrix.png', dpi=300)
        print("\n✓ Saved 'hybrid_confusion_matrix.png'\n")
        
        # ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Hybrid Dengue Model')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('hybrid_roc_curve.png', dpi=300)
        print("✓ Saved 'hybrid_roc_curve.png'\n")
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

# ============ MAIN PIPELINE ============
def main():
    print("\n" + "="*60)
    print("HYBRID DENGUE PREDICTION MODEL")
    print("="*60 + "\n")
    
    # Phase 1: Data Preprocessing
    preprocessor = DataPreprocessor(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.load_and_split()
    
    # Phase 2: Train Supervised DNN
    dnn = SupervisedDNN(X_train.shape[1], cfg)
    dnn.train(X_train, y_train, X_val, y_val)
    
    # Save initial predictions for RL phase
    initial_proba = dnn.predict_proba(X_train)
    
    # Phase 3: RL Fine-tuning
    rl_tuner = RLFineTuner(dnn.model, cfg)
    rl_tuner.train(X_train, y_train, initial_proba)
    
    # Phase 4: Evaluation
    ensemble = HybridEnsemble(dnn, cfg)
    results = ensemble.evaluate(X_test, y_test)
    
    print("="*60)
    print("✓ HYBRID MODEL TRAINING COMPLETE!")
    print("="*60 + "\n")
    
    return dnn.model, results

if __name__ == "__main__":
    model, results = main()