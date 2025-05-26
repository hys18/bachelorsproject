# Import the necessary libraries
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import subprocess

import os
import trimesh
import tqdm
from pathlib import Path

from skimage.measure import marching_cubes

# set GPU
gpu_id = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

current_path = Path.cwd() # Set to current working directory

torch.manual_seed(42) # Set random seed for reproducibility

# 1. Create NeRF+PE model
# Positional encoding as a PyTorch module
class PE(nn.Module):
    def __init__(self, in_D=3, L=10):
        super(PE, self).__init__()
        self.in_D = in_D
        self.L = L
        # Store frequency_bands as non-trainable buffer: fixed
        frequency_bands = 2.0 ** torch.arange(L, dtype=torch.float32) * np.pi # (,10)
        self.register_buffer("frequency_bands", frequency_bands) 
        
    def forward(self, x):    
        x = x.unsqueeze(-1) # (N, 3, 1)
        # Sinusoidal encoding
        sin_x = torch.sin(self.frequency_bands*x) # (N, 30, 10)
        cos_x = torch.cos(self.frequency_bands*x) # (N, 30, 10)
        encoded_x = torch.cat([sin_x, cos_x], dim=-1) # (N, 30, 20)
        return encoded_x.view(x.shape[0],-1).contiguous() # (N, 600)

class NeRF(nn.Module):
    def __init__(self, in_D=3, L=10, hidden_D=128):
        super(NeRF, self).__init__()      
        # Input encoding and positional encoding model
        self.pe=PE(in_D, L)
        self.encoded_D = in_D * 2 * L
        # MLP layers  
        self.fc1 = nn.Linear(self.encoded_D, hidden_D) # Take in 3 coordinates as input
        self.fc2 = nn.Linear(hidden_D, hidden_D)
        self.fc3 = nn.Linear(hidden_D, hidden_D)
        self.fc4 = nn.Linear(hidden_D, 1) # Produce an occupancy value
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor):   
        # Apply positional encoding
        x = self.pe(x) # (N, 3) -> (N, encoded_D)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x)) # Apply sigmoid function        
        return x      
    
# 2. Load dataset & dataloader
original_mesh_path = current_path / "models_for_total_liver/original_mesh.ply"
train_data = np.load(original_mesh_path)
x_train = train_data[:, :3].astype(np.float32)
y_train = train_data[:, 3].astype(np.float32).reshape(-1, 1) 
train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

# 3. Initialize model & loss function & optimizer
model = NeRF(in_D=3, L=10).to(device).cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001) 

# Set up for tensorboard
tensorboard_logdir = "logs/occupancy_nerfpe"
writer = SummaryWriter(log_dir=tensorboard_logdir)

# 4. Train model
epochs = 500
for epoch in range(epochs):
    model.train()
    total_loss=0
    for batch in train_loader:
        coordinate_batch, occupancy_batch = batch
        coordinate_batch, occupancy_batch = coordinate_batch.cuda(), occupancy_batch.cuda()
        
        optimizer.zero_grad()
        occupancy_pred = model(coordinate_batch)
        loss = criterion(occupancy_pred, occupancy_batch)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

    avg_loss = total_loss / len(train_loader) 
    writer.add_scalar("Loss/train", avg_loss, epoch)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

writer.close()
torch.save(model.state_dict(), "occupancy_nerfpe.pt")

# 5. Build 3D grid of points within [-0.5, 0.5]
resolution = 256 # 256 x 256 x 256 points
x = np.linspace(-0.5, 0.5, resolution)
y = np.linspace(-0.5, 0.5, resolution) 
z = np.linspace(-0.5, 0.5, resolution)

# 6. Create dataset from the 3D grid points for evaluation
coords = np.array(np.meshgrid(x, y, z, indexing='ij')).astype(np.float32)
coords = np.moveaxis(coords, 0, -1).reshape(-1, 3) # Reordering for (x,y,z)
coords_tensor = torch.tensor(coords, dtype=torch.float32, requires_grad=False).to(device)

mesh_data = TensorDataset(coords_tensor)
test_loader = DataLoader(mesh_data, batch_size=4096, shuffle=False)

# 7. Evaluate model on 3D grid and combine predictions 
model.eval()

with torch.no_grad():
    occupancy_pred = []
    for batch in tqdm.tqdm(test_loader, ncols=100):
        coordinates = batch[0].cuda()  
        predictions = model(coordinates)
        occ_pred = predictions.detach().clone().cpu().numpy().flatten()
        occupancy_pred.append(occ_pred)
occupancy_pred = np.concatenate(occupancy_pred, axis=0)
occupancy_pred = occupancy_pred.reshape(resolution, resolution, resolution)

# 8. Apply Marching Cubes algorithm by reshaping it and extract mesh from the predictions
voxel_size = (1/resolution,)*3
vertices, faces, normals, values = marching_cubes(occupancy_pred, level=0.5, spacing=voxel_size)
mesh = trimesh.Trimesh(vertices, faces) 
# Normalize mesh
mesh.fix_normals()
# centralize mesh
mesh.vertices += [-0.5,-0.5,-0.5]
# Invert surface toward outside
mesh.invert()

# Save mesh to a ply file
output_mesh_path = "nerfpe_reconstructed_mesh.ply"
mesh.export(output_mesh_path)
print(f"Mesh reconstruction complete. Saved at {output_mesh_path}")
