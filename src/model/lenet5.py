"""
LeNet-5 Implementation for Tifinagh Character Recognition
Authors: Ouzaina Marwane
Date: 2024-2025
"""

import numpy as np
from .layers import ConvLayer, PoolingLayer, DenseLayer, ActivationLayer
from .optimizers import SGDOptimizer, AdamOptimizer

class LeNet5:
    """
    LeNet-5 CNN Architecture for Tifinagh character recognition
    
    Architecture:
    Input (32x32x1) -> C1 (28x28x6) -> S2 (14x14x6) -> C3 (10x10x16) -> S4 (5x5x16) -> C5 (120) -> F6 (84) -> Output (33)
    """
    
    def __init__(self, input_shape=(32, 32, 1), num_classes=33, learning_rate=0.001):
        """
        Initialize LeNet-5 architecture
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes (33 for Tifinagh)
            learning_rate: Learning rate for optimization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Initialize layers
        self.layers = []
        self._build_architecture()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def _build_architecture(self):
        """Build the LeNet-5 architecture"""
        # C1: Convolution Layer (6 filters, 5x5 kernel)
        self.layers.append(ConvLayer(filters=6, kernel_size=5, stride=1, padding=0))
        self.layers.append(ActivationLayer('tanh'))
        
        # S2: Average Pooling Layer (2x2 pool)
        self.layers.append(PoolingLayer(pool_size=2, stride=2, mode='average'))
        
        # C3: Convolution Layer (16 filters, 5x5 kernel)
        self.layers.append(ConvLayer(filters=16, kernel_size=5, stride=1, padding=0))
        self.layers.append(ActivationLayer('tanh'))
        
        # S4: Average Pooling Layer (2x2 pool)
        self.layers.append(PoolingLayer(pool_size=2, stride=2, mode='average'))
        
        # Flatten layer
        self.layers.append(FlattenLayer())
        
        # C5: Fully Connected Layer (120 neurons)
        self.layers.append(DenseLayer(120))
        self.layers.append(ActivationLayer('tanh'))
        
        # F6: Fully Connected Layer (84 neurons)
        self.layers.append(DenseLayer(84))
        self.layers.append(ActivationLayer('tanh'))
        
        # Output Layer (33 neurons)
        self.layers.append(DenseLayer(self.num_classes))
        self.layers.append(ActivationLayer('softmax'))
    
    def forward(self, X):
        """
        Forward propagation through the network
        
        Args:
            X: Input batch of images
            
        Returns:
            Output predictions
        """
        self.activations = []
        current_input = X
        
        for layer in self.layers:
            current_input = layer.forward(current_input)
            self.activations.append(current_input.copy())
        
        return current_input
    
    def backward(self, dout):
        """
        Backward propagation through the network
        
        Args:
            dout: Gradient from loss function
        """
        current_grad = dout
        
        for i in range(len(self.layers) - 1, -1, -1):
            current_grad = self.layers[i].backward(current_grad)
    
    def compute_loss(self, predictions, targets):
        """
        Compute cross-entropy loss
        
        Args:
            predictions: Model predictions
            targets: True labels (one-hot encoded)
            
        Returns:
            Loss value and gradient
        """
        # Clip predictions to prevent log(0)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        # Cross-entropy loss
        loss = -np.mean(np.sum(targets * np.log(predictions), axis=1))
        
        # Gradient of loss w.r.t predictions
        grad = (predictions - targets) / targets.shape[0]
        
        return loss, grad
    
    def compute_accuracy(self, predictions, targets):
        """
        Compute classification accuracy
        
        Args:
            predictions: Model predictions
            targets: True labels (one-hot encoded)
            
        Returns:
            Accuracy percentage
        """
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1)
        return np.mean(pred_classes == true_classes) * 100
    
    def train_step(self, X_batch, y_batch, optimizer):
        """
        Single training step
        
        Args:
            X_batch: Batch of input images
            y_batch: Batch of target labels
            optimizer: Optimizer instance
            
        Returns:
            loss, accuracy
        """
        # Forward pass
        predictions = self.forward(X_batch)
        
        # Compute loss
        loss, grad_loss = self.compute_loss(predictions, y_batch)
        
        # Backward pass
        self.backward(grad_loss)
        
        # Update parameters
        optimizer.update(self.layers)
        
        # Compute accuracy
        accuracy = self.compute_accuracy(predictions, y_batch)
        
        return loss, accuracy
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Input data
            
        Returns:
            Predicted class probabilities
        """
        return self.forward(X)
    
    def predict_classes(self, X):
        """
        Predict class labels
        
        Args:
            X: Input data
            
        Returns:
            Predicted class labels
        """
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            epochs=100, batch_size=32, optimizer='adam', verbose=True):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            optimizer: Optimizer type ('sgd' or 'adam')
            verbose: Print training progress
            
        Returns:
            Training history
        """
        # Initialize optimizer
        if optimizer == 'sgd':
            opt = SGDOptimizer(learning_rate=self.learning_rate)
        elif optimizer == 'adam':
            opt = AdamOptimizer(learning_rate=self.learning_rate)
        else:
            raise ValueError("Optimizer must be 'sgd' or 'adam'")
        
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Training phase
            epoch_loss = 0
            epoch_acc = 0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                loss, acc = self.train_step(X_batch, y_batch, opt)
                epoch_loss += loss
                epoch_acc += acc
            
            # Average metrics
            train_loss = epoch_loss / n_batches
            train_acc = epoch_acc / n_batches
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation phase
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_loss, _ = self.compute_loss(val_predictions, y_val)
                val_acc = self.compute_accuracy(val_predictions, y_val)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                if verbose and epoch % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%')
            else:
                if verbose and epoch % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.2f}%')
        
        return self.history
    
    def save_model(self, filepath):
        """Save model parameters"""
        model_data = {
            'layers': [],
            'history': self.history,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }
        
        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                model_data['layers'].append(layer.get_params())
            else:
                model_data['layers'].append(None)
        
        np.save(filepath, model_data)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model parameters"""
        model_data = np.load(filepath, allow_pickle=True).item()
        
        self.history = model_data['history']
        self.input_shape = model_data['input_shape']
        self.num_classes = model_data['num_classes']
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'set_params') and model_data['layers'][i] is not None:
                layer.set_params(model_data['layers'][i])
        
        print(f"Model loaded from {filepath}")

class FlattenLayer:
    """Flatten layer to convert 2D feature maps to 1D vector"""
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, X):
        """Forward pass"""
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, dout):
        """Backward pass"""
        return dout.reshape(self.input_shape)