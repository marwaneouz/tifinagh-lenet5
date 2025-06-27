"""
Neural Network Layers Implementation
Contains all layer types used in LeNet-5 architecture
"""

import numpy as np

class ConvLayer:
    """
    Convolutional layer implementation
    """
    
    def __init__(self, filters, kernel_size, stride=1, padding=0):
        """
        Initialize convolutional layer
        
        Args:
            filters: Number of filters (output channels)
            kernel_size: Size of convolution kernel
            stride: Stride for convolution
            padding: Padding size
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Parameters will be initialized on first forward pass
        self.weights = None
        self.bias = None
        
        # For backpropagation
        self.input_cache = None
        self.dW = None
        self.db = None
    
    def initialize_parameters(self, input_channels):
        """Initialize weights and biases"""
        # Xavier initialization
        fan_in = input_channels * self.kernel_size * self.kernel_size
        fan_out = self.filters * self.kernel_size * self.kernel_size
        
        self.weights = np.random.randn(
            self.filters, input_channels, self.kernel_size, self.kernel_size
        ) * np.sqrt(2.0 / (fan_in + fan_out))
        
        self.bias = np.zeros((self.filters, 1))
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Output tensor after convolution
        """
        # Initialize parameters if first time
        if self.weights is None:
            _, input_channels, _, _ = X.shape
            self.initialize_parameters(input_channels)
        
        self.input_cache = X
        
        batch_size, input_channels, input_height, input_width = X.shape
        
        # Calculate output dimensions
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.filters, output_height, output_width))
        
        # Add padding if needed
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            X_padded = X
        
        # Perform convolution
        for n in range(batch_size):
            for f in range(self.filters):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.kernel_size
                        end_j = start_j + self.kernel_size
                        
                        region = X_padded[n, :, start_i:end_i, start_j:end_j]
                        output[n, f, i, j] = np.sum(region * self.weights[f]) + self.bias[f]
        
        return output
    
    def backward(self, dout):
        """
        Backward propagation
        
        Args:
            dout: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        X = self.input_cache
        batch_size, input_channels, input_height, input_width = X.shape
        _, _, output_height, output_width = dout.shape
        
        # Initialize gradients
        dX = np.zeros_like(X)
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)
        
        # Add padding if needed
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            dX_padded = np.pad(dX, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            X_padded = X
            dX_padded = dX
        
        # Compute gradients
        for n in range(batch_size):
            for f in range(self.filters):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.kernel_size
                        end_j = start_j + self.kernel_size
                        
                        # Gradient w.r.t. weights
                        region = X_padded[n, :, start_i:end_i, start_j:end_j]
                        self.dW[f] += dout[n, f, i, j] * region
                        
                        # Gradient w.r.t. input
                        dX_padded[n, :, start_i:end_i, start_j:end_j] += dout[n, f, i, j] * self.weights[f]
                
                # Gradient w.r.t. bias
                self.db[f] += np.sum(dout[:, f, :, :])
        
        # Remove padding from dX if it was added
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
        
        return dX
    
    def get_params(self):
        """Get layer parameters"""
        return {'weights': self.weights, 'bias': self.bias}
    
    def set_params(self, params):
        """Set layer parameters"""
        self.weights = params['weights']
        self.bias = params['bias']

class PoolingLayer:
    """
    Pooling layer implementation (Max and Average pooling)
    """
    
    def __init__(self, pool_size=2, stride=2, mode='max'):
        """
        Initialize pooling layer
        
        Args:
            pool_size: Size of pooling window
            stride: Stride for pooling
            mode: 'max' or 'average'
        """
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode
        self.input_cache = None
        self.mask = None
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: Input tensor
            
        Returns:
            Pooled output
        """
        self.input_cache = X
        batch_size, channels, input_height, input_width = X.shape
        
        # Calculate output dimensions
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, output_height, output_width))
        
        if self.mode == 'max':
            self.mask = np.zeros_like(X)
        
        for n in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size
                        
                        region = X[n, c, start_i:end_i, start_j:end_j]
                        
                        if self.mode == 'max':
                            output[n, c, i, j] = np.max(region)
                            # Create mask for backprop
                            mask_region = (region == output[n, c, i, j])
                            self.mask[n, c, start_i:end_i, start_j:end_j] = mask_region
                        elif self.mode == 'average':
                            output[n, c, i, j] = np.mean(region)
        
        return output
    
    def backward(self, dout):
        """
        Backward propagation
        
        Args:
            dout: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        X = self.input_cache
        batch_size, channels, input_height, input_width = X.shape
        _, _, output_height, output_width = dout.shape
        
        dX = np.zeros_like(X)
        
        for n in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size
                        
                        if self.mode == 'max':
                            dX[n, c, start_i:end_i, start_j:end_j] += dout[n, c, i, j] * self.mask[n, c, start_i:end_i, start_j:end_j]
                        elif self.mode == 'average':
                            dX[n, c, start_i:end_i, start_j:end_j] += dout[n, c, i, j] / (self.pool_size * self.pool_size)
        
        return dX

class DenseLayer:
    """
    Fully connected (dense) layer implementation
    """
    
    def __init__(self, units):
        """
        Initialize dense layer
        
        Args:
            units: Number of output units
        """
        self.units = units
        self.weights = None
        self.bias = None
        self.input_cache = None
        self.dW = None
        self.db = None
    
    def initialize_parameters(self, input_size):
        """Initialize weights and biases"""
        # Xavier initialization
        self.weights = np.random.randn(input_size, self.units) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, self.units))
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: Input tensor
            
        Returns:
            Output after linear transformation
        """
        # Initialize parameters if first time
        if self.weights is None:
            input_size = X.shape[1]
            self.initialize_parameters(input_size)
        
        self.input_cache = X
        output = np.dot(X, self.weights) + self.bias
        return output
    
    def backward(self, dout):
        """
        Backward propagation
        
        Args:
            dout: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        X = self.input_cache
        m = X.shape[0]
        
        # Compute gradients
        self.dW = np.dot(X.T, dout) / m
        self.db = np.sum(dout, axis=0, keepdims=True) / m
        dX = np.dot(dout, self.weights.T)
        
        return dX
    
    def get_params(self):
        """Get layer parameters"""
        return {'weights': self.weights, 'bias': self.bias}
    
    def set_params(self, params):
        """Set layer parameters"""
        self.weights = params['weights']
        self.bias = params['bias']

class ActivationLayer:
    """
    Activation layer implementation
    """
    
    def __init__(self, activation):
        """
        Initialize activation layer
        
        Args:
            activation: Activation function ('tanh', 'sigmoid', 'relu', 'softmax')
        """
        self.activation = activation
        self.input_cache = None
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: Input tensor
            
        Returns:
            Activated output
        """
        self.input_cache = X
        
        if self.activation == 'tanh':
            return np.tanh(X)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(X, -500, 500)))
        elif self.activation == 'relu':
            return np.maximum(0, X)
        elif self.activation == 'softmax':
            exp_x = np.exp(X - np.max(X, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
    
    def backward(self, dout):
        """
        Backward propagation
        
        Args:
            dout: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        X = self.input_cache
        
        if self.activation == 'tanh':
            return dout * (1 - np.tanh(X) ** 2)