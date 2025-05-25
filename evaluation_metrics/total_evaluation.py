import trimesh
from mesh_utils import MeshEvaluator
import csv
import glob
import os
from pathlib import Path

current_path = Path.cwd() # Set to current working directory

# Load predicted meshes for a model
pred_dir = current_path / 'evaluation_metrics/standard_mlp_reconstructed_meshes'
# pred_dir = current_path / 'evaluation_metrics/nerfpe_reconstructed_meshes'
# pred_dir = current_path / 'evaluation_metrics/siren_reconstructed_meshes'
# pred_dir = current_path / 'evaluation_metrics/ffpe_reconstructed_meshes'
print(pred_dir)
pred_list = sorted(glob.glob(os.path.join(pred_dir, "*.ply")))
gt_dir = current_path / 'models_for_total_liver/original_meshes.ply'
gt_list = sorted(glob.glob(os.path.join(gt_dir, "*.ply")))

# initiate mesh evaluator
mesh_evaluator = MeshEvaluator(
    N_pointcloud=100_000,
    N_cube=128,
    min_max_range=[-0.5, 0.5],
    winding_number_threshold=0.5,
    hash_resolution=512,
    verbose=True,
    random_seed=42
)

# Evaluate against the ground truth and add to table
results_table = []

for i in range(len(pred_list)):
    pred_mesh = trimesh.load(pred_list[i])
    gt_mesh = trimesh.load(gt_list[i]) 
    mesh_pair = [
        (i, pred_mesh, gt_mesh),
    ]
    metrics, _ = mesh_evaluator.eval_mesh(pred_mesh=pred_mesh, gt_mesh=gt_mesh)
    row = [i] + [f"{metrics[k]:.4f}" for k in metrics]
    if not results_table:
        headers = ["Comparison"] + list(metrics.keys())
        results_table.append(headers)
    results_table.append(row)

# Save as CSV
save_path = current_path / "evaluation_metrics/ffpe_evaluation_all.csv"
with open(save_path, "w", newline="") as f:
    writer = csv.writer(f)
    for row in results_table:
        writer.writerow(row)
