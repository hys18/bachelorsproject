# Implement necessary modules
import trimesh
from mesh_utils import MeshEvaluator
import csv
from pathlib import Path

current_path = Path.cwd()

# Load predicted meshes
mlp_path = current_path / 'models_for_one_liver/standard_mlp_reconstructed_mesh.ply'
nerf_path = current_path / 'models_for_one_liver/nerfpe_reconstructed_mesh.ply'
ffpe_path = current_path / 'models_for_one_liver/ffpe_reconstructed_mesh.ply'
siren_path = current_path / 'models_for_one_liver/siren_reconstructed_mesh.ply'
original_path = current_path / 'models_for_one_liver/original_mesh.ply'

mlp_mesh = trimesh.load(mlp_path)
nerfpe_mesh = trimesh.load(nerf_path)
ffpe_mesh = trimesh.load(ffpe_mesh)
siren_mesh = trimesh.load(siren_path)
original_mesh = trimesh.load(original_path)

# Initialize MeshEvaluator
mesh_evaluator = MeshEvaluator(
    N_pointcloud=100_000,
    N_cube=128,
    min_max_range=[-0.5, 0.5],
    winding_number_threshold=0.5,
    hash_resolution=512,
    verbose=True,
    random_seed=42
)

# Define pairs for comparison
mesh_pairs = [
    ("mlp", mlp_mesh, original_mesh),
    ("nerfpe", nerfpe_mesh, original_mesh),
    ("ffpe", ffpe_mesh, original_mesh),
    ("siren", siren_mesh, original_mesh)
]

# Store results in a list
results_table = []

for name, pred_mesh, gt_mesh in mesh_pairs:
    metrics, _ = mesh_evaluator.eval_mesh(pred_mesh=pred_mesh, gt_mesh=gt_mesh)
    row = [name] + [f"{metrics[k]:.4f}" for k in metrics]
    if not results_table:
        headers = ["Comparison"] + list(metrics.keys())
        results_table.append(headers)
    results_table.append(row)

save_path = "/home/bsc18/workplace/FINAL SUBMISSION/evaluation_metrics/single_mesh_comparison_results.csv"  

# Save as a csv file
with open(save_path, "w", newline="") as f:
    writer = csv.writer(f)
    for row in results_table:
        writer.writerow(row)
