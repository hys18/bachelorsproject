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
gpu_id = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Make output directories
os.makedirs("siren_reconstructed_meshes", exist_ok=True)

torch.manual_seed(42)

# Create siren model
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)        
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self):
        super(Siren, self).__init__()
        hidden_D = 512
        self.fc1 = SineLayer(3, hidden_D, is_first=True, omega_0=30)
        self.fc2 = SineLayer(hidden_D, hidden_D, omega_0=30)
        self.fc3 = SineLayer(hidden_D, hidden_D, omega_0=30)
        self.fc4 = SineLayer(hidden_D, hidden_D, omega_0=30)
        self.fc5 = SineLayer(hidden_D, hidden_D, omega_0=30)
        self.fc6 = nn.Linear(hidden_D, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.Sigmoid(x)
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
tensorboard_logdir = "total_logs/occupancy_siren"
port = 6008
subprocess.Popen(["tensorboard", "--logdir", tensorboard_logdir, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"TensorBoard running at: http://localhost:{port}")

# Training & Evaluating Predictions & Extracting Meshes
for fpath in file_list:
    liver_id = os.path.splitext(os.path.basename(fpath))[0]
    
    # Load dataset
    train_data = np.load(fpath)
    x_train = train_data[:, :3].astype(np.float32)
    y_train = train_data[:, 3].astype(np.float32).reshape(-1, 1)
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

    # Model
    model = Siren().to(device)
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

    mesh_path = f"siren_reconstructed_meshes/{liver_id}.ply"
    mesh.export(mesh_path)
    print(f"Saved mesh: {mesh_path}")
