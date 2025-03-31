import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# --- Configuration ---
NUM_EPOCHS = 2
BATCH_SIZE = 4
NUM_WORKERS = 2

# --- 1. Tiny Model ---
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5))
    
    def forward(self, x):
        return self.net(x)

# --- 2. Synthetic Dataset ---
class FakeData(Dataset):
    def __len__(self):
        return 100
    
    def __getitem__(self, idx):
        return torch.randn(10), torch.randint(0, 5, (1,)).item()

# --- 3. Main Training ---
def main():
    # Initialize DDP
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    
    print(f"Rank {rank} (GPU {local_rank}) initialized | World size: {world_size}")

    # Model + DDP
    model = TestModel().cuda()
    ddp_model = DDP(model, device_ids=[local_rank])
    
    # DataLoader with DistributedSampler
    dataset = FakeData()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)

    # Training loop
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        ddp_model.train()
        
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Verification point 1: Unique batches per GPU
            print(f"Rank {rank} | Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | "
                  f"Sample IDs: {sampler.sample_indices[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]}")

        # Verification point 2: Model synchronization
        if rank == 0:
            print(f"\nEpoch {epoch} complete (All-Rank Avg Loss: {loss.item():.4f})\n")

    # Verification point 3: GPU memory check
    print(f"Rank {rank} final CUDA memory: {torch.cuda.memory_allocated(local_rank)/1e6:.2f} MB")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()