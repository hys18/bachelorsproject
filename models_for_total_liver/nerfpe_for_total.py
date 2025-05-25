# Import the necessary libraries
import torch
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
from skimage.measure import marching_cubes
from pathlib import Path

current_path = Path.cwd() # Set to current working directory

# Set GPU
gpu_id = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Make output directories
os.makedirs("nerfpe_reconstructed_meshes", exist_ok=True)

# Create NeRFPE model
torch.manual_seed(42)

class PE(nn.Module):
    def __init__(self, in_D=3, L=10):
        super(PE, self).__init__()
        self.in_D = in_D
        self.L = L
        frequency_bands = 2.0 ** torch.arange(L, dtype=torch.float32) * np.pi
        self.register_buffer("frequency_bands", frequency_bands)
        
    def forward(self, x):    
        x = x.unsqueeze(-1)
        sin_x = torch.sin(self.frequency_bands*x) 
        cos_x = torch.cos(self.frequency_bands*x)
        encoded_x = torch.cat([sin_x, cos_x], dim=-1)
        return encoded_x.view(x.shape[0],-1).contiguous() 

class NeRF(nn.Module):
    def __init__(self, in_D=3, L=10, hidden_D=128):
        super(NeRF, self).__init__()    
        self.pe=PE(in_D, L)
        self.encoded_D = in_D * 2 * L
        self.fc1 = nn.Linear(self.encoded_D, hidden_D)
        self.fc2 = nn.Linear(hidden_D, hidden_D)
        self.fc3 = nn.Linear(hidden_D, hidden_D)
        self.fc4 = nn.Linear(hidden_D, 1) 
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor):   
        x = self.pe(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))       
        return x

# Build 3D grid for evaluation
resolution = 256
x = np.linspace(-0.5, 0.5, resolution)
y = np.linspace(-0.5, 0.5, resolution)
z = np.linspace(-0.5, 0.5, resolution)
coords = np.array(np.meshgrid(x, y, z, indexing='ij')).astype(np.float32)
coords = np.moveaxis(coords, 0, -1).reshape(-1, 3)

# Load all .npy files
npy_dir = current_path / 'original_meshes_ply'
file_list = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))

# Launch TensorBoard
tensorboard_logdir = "total_logs/occupancy_nerfpe"

# Training & Evaluating Predictions & Extracting Meshes
for fpath in file_list:
    liver_id = os.path.splitext(os.path.basename(fpath))[0]
    print(f"\nProcessing: {liver_id}")

    # Load dataset
    train_data = np.load(fpath)
    x_train = train_data[:, :3].astype(np.float32)
    y_train = train_data[:, 3].astype(np.float32).reshape(-1, 1)
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

    # Model
    model = NeRF(in_D=3, L=10).to(device).cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(tensorboard_logdir, liver_id))

    # Train
    epochs = 500
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            coordinate_batch, occupancy_batch = batch
            coordinate_batch, occupancy_batch = coordinate_batch.cuda(), occupancy_batch.cuda()
            optimizer.zero_grad()
            occupancy_pred = model(coordinate_batch)
            loss = criterion(occupancy_pred, occupancy_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    writer.close()
    
    # Create dataset from the 3D grid points for evaluation
    coords_tensor = torch.tensor(coords, dtype=torch.float32, requires_grad=False).to(device)
    mesh_data = TensorDataset(coords_tensor)
    test_loader = DataLoader(mesh_data, batch_size=4096, shuffle=False)

    # Evaluate on grid
    model.eval()
    occupancy_pred = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc=f"Infer: {liver_id}", ncols=100):
            coordinates = batch[0].cuda()
            predictions = model(coordinates)
            occ_pred = predictions.detach().cpu().numpy().flatten()
            occupancy_pred.append(occ_pred)
            
    occupancy_pred = np.concatenate(occupancy_pred, axis=0)
    occupancy_pred = occupancy_pred.reshape(resolution, resolution, resolution)

    voxel_size = (1/resolution,) * 3
    vertices, faces, normals, values = marching_cubes(occupancy_pred, level=0.5, spacing=voxel_size)


    mesh = trimesh.Trimesh(vertices, faces)
    mesh.fix_normals()
    mesh.vertices += [-0.5, -0.5, -0.5]
    mesh.invert()

    mesh_path = f"nerfpe_reconstructed_meshes/{liver_id}.ply"
    mesh.export(mesh_path)
    print(f"Saved mesh: {mesh_path}")