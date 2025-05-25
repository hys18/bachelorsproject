# Import the necessary libraries
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import glob
import subprocess

import os
import trimesh
import tqdm
from pathlib import Path

from skimage.measure import marching_cubes

current_path = Path.cwd() # Set to current working directory

# Set GPU
gpu_id = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42) # Set random seed for reproducibility

# 1. Create MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        hidden_D = 512
        # layers
        self.fc1 = nn.Linear(3, hidden_D) # take in 3 coordinates as input
        self.fc2 = nn.Linear(hidden_D, hidden_D)
        self.fc3 = nn.Linear(hidden_D, hidden_D)
        self.fc4 = nn.Linear(hidden_D, hidden_D)
        self.fc5 = nn.Linear(hidden_D, hidden_D)
        self.fc6 = nn.Linear(hidden_D, 1) # Produce an occupancy value
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor):   
        x = F.relu(self.fc1(x)) # Apply relu activation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.sigmoid(self.fc6(x)) # Apply sigmoid function        
        return x
    
# 2. Load dataset & dataloader
original_mesh_path = current_path / "models_for_total_liver/original_mesh.ply"
train_data = np.load(original_mesh_path)
x_train = train_data[:, :3].astype(np.float32)
y_train = train_data[:, 3].astype(np.float32).reshape(-1, 1) 
train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

# 3. Initialize model & loss function & optimizer
model = MLP().to(device).cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001) 

# Set up for tensorboard
tensorboard_logdir = "logs/occupancy_mlp"
port = 6009
subprocess.Popen(["tensorboard", "--logdir", tensorboard_logdir, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"TensorBoard will run at: http://localhost:{port}")
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
torch.save(model.state_dict(), "mlp.pt")

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
occupancy_pred = np.concat(occupancy_pred, axis=0)
occupancy_pred = occupancy_pred.reshape(resolution, resolution, resolution)

# 8. Apply Marching Cubes algorithm by reshaping it and extract mesh from the predictions
voxel_size = (1/resolution,)*3
vertices, faces, normals, values = marching_cubes(occupancy_pred, level=0.5, spacing=voxel_size)
mesh = trimesh.Trimesh(vertices, faces) 
# Normalize mesh
mesh.fix_normals()
# Centralize mesh
mesh.vertices += [-0.5,-0.5,-0.5]
# Invert surface toward outside
mesh.invert()

# Save mesh to a ply file
output_mesh_path = "standard_mlp_reconstructed_mesh.ply"
mesh.export(output_mesh_path)
print(f"Mesh reconstruction complete. Saved at {output_mesh_path}")
