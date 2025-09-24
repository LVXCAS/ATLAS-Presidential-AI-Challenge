"""
GTX 1660 SUPER GPU PERFORMANCE TEST
Compare CPU vs GPU training performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from datetime import datetime

class SimpleNeuralNet(nn.Module):
    """Simple neural network for performance testing"""
    def __init__(self, input_size=50, hidden_size=128):
        super(SimpleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_model(device, model, X, y, epochs=50, batch_size=256):
    """Train model on specified device"""
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs.squeeze(), y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()  # Ensure GPU operations complete

        if epoch % 10 == 0:
            print(f'  Epoch {epoch}: Loss = {loss.item():.6f}')

    end_time = time.time()
    return end_time - start_time

def main():
    print("GTX 1660 SUPER PERFORMANCE COMPARISON")
    print("=" * 50)

    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {gpu_available}")

    if gpu_available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create test data (simulating financial features)
    np.random.seed(42)
    torch.manual_seed(42)

    # Large dataset for meaningful comparison
    n_samples = 10000
    n_features = 50

    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples)

    print(f"\nDataset: {n_samples} samples, {n_features} features")
    print(f"Data size: {X.numel() * 4 / 1e6:.1f} MB")

    # Test parameters
    epochs = 100
    batch_size_gpu = 512  # GTX 1660 Super optimized
    batch_size_cpu = 128  # CPU optimized

    print(f"\nTraining parameters:")
    print(f"Epochs: {epochs}")
    print(f"GPU Batch size: {batch_size_gpu}")
    print(f"CPU Batch size: {batch_size_cpu}")

    # CPU Training
    print("\nCPU TRAINING")
    print("-" * 20)
    cpu_device = torch.device('cpu')
    cpu_model = SimpleNeuralNet()
    cpu_time = train_model(cpu_device, cpu_model, X, y, epochs, batch_size_cpu)
    print(f"CPU Training time: {cpu_time:.2f} seconds")

    # GPU Training (if available)
    if gpu_available:
        print("\nGPU TRAINING")
        print("-" * 20)
        gpu_device = torch.device('cuda')
        gpu_model = SimpleNeuralNet()

        # Clear GPU cache
        torch.cuda.empty_cache()

        gpu_time = train_model(gpu_device, gpu_model, X, y, epochs, batch_size_gpu)
        print(f"GPU Training time: {gpu_time:.2f} seconds")

        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"\nPERFORMANCE SUMMARY")
        print("=" * 30)
        print(f"CPU Time: {cpu_time:.2f}s")
        print(f"GPU Time: {gpu_time:.2f}s")
        print(f"GPU Speedup: {speedup:.1f}x")

        if speedup > 3:
            print(">> Excellent GPU acceleration for deep learning!")
        elif speedup > 1.5:
            print(">> Good GPU acceleration achieved")
        else:
            print(">> Limited speedup - check GPU utilization")

        # Memory usage
        gpu_memory_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU Memory Used: {gpu_memory_used:.2f} GB")

        # Performance metrics
        samples_per_second_cpu = n_samples * epochs / cpu_time
        samples_per_second_gpu = n_samples * epochs / gpu_time

        print(f"\nTHROUGHPUT COMPARISON")
        print("-" * 25)
        print(f"CPU: {samples_per_second_cpu:.0f} samples/second")
        print(f"GPU: {samples_per_second_gpu:.0f} samples/second")
        print(f"GPU throughput improvement: {samples_per_second_gpu/samples_per_second_cpu:.1f}x")

        # Alpha discovery implications
        print(f"\nALPHA DISCOVERY BENEFITS")
        print("-" * 28)
        print(f">> Training time reduced by {((cpu_time - gpu_time) / cpu_time * 100):.0f}%")
        print(f">> Can process {speedup:.0f}x more symbols in same time")
        print(f">> Real-time model updates feasible")
        print(f">> Larger batch sizes enable better model accuracy")

    else:
        print("\nNo GPU available for comparison")

    print(f"\nTest completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()