import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import argparse

def setup_distributed():
    # Initialize process group
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank

def get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda")
    return torch.device("cpu")

class SimpleCNN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        return self.net(x)

def generate_synthetic_data(num_samples, batch_size, local_rank=None, world_size=None):
    # For distributed training, each GPU gets a subset
    if local_rank is not None and world_size is not None:
        num_samples_per_rank = num_samples // world_size
        torch.manual_seed(42 + local_rank)  # Different seed per GPU
        x = torch.randn(num_samples_per_rank, 1, 28, 28)
        y = torch.randint(0, 10, (num_samples_per_rank,))
    else:
        x = torch.randn(num_samples, 1, 28, 28)
        y = torch.randint(0, 10, (num_samples,))
    
    return torch.utils.data.DataLoader(list(zip(x, y)), batch_size=batch_size)

def benchmark(device, epochs=5, batch_size=64, num_samples=10000, local_rank=None, world_size=None):
    model = SimpleCNN().to(device)
    
    # Wrap model in DDP for distributed training
    if local_rank is not None:
        model = DDP(model, device_ids=[local_rank])
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    dataloader = generate_synthetic_data(num_samples, batch_size, local_rank, world_size)

    # Only print from rank 0 to avoid duplicate output
    is_main = local_rank is None or local_rank == 0
    if is_main:
        print(f"Starting training on device: {device}, world_size: {world_size or 1}")
    
    start = time.time()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if is_main:
            print(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss:.4f}")

    end = time.time()
    if is_main:
        print(f"\nFinished in {end - start:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size per GPU")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Total number of samples (divided across GPUs)")
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="Hidden dimension size to scale model")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank passed from distributed launcher")
    args = parser.parse_args()
    # Initialize the distributed environment if needed
    local_rank = None
    world_size = None
    if args.local_rank != -1:
        local_rank = setup_distributed()
        world_size = dist.get_world_size()

    device = get_device()
    benchmark(device, args.epochs, args.batch_size, args.num_samples, local_rank, world_size)