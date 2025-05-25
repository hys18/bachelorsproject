"""
This file contains some useful functions for mesh processing.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Disclaimer: This file is taken and modified from many sources.

References:
+ Occupancy Networks: https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/eval.py#L29
+ Shape as Points: https://github.com/autonomousvision/shape_as_points/blob/main/src/eval.py#L23
+ DeepSDF: https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/metrics/chamfer.py#L9
+ KDTree: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
+ KDTree.query: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
+ Point Cloud Utils: https://github.com/fwilliams/point-cloud-utils/blob/master/point_cloud_utils/__init__.py#L84
"""

from typing import Callable

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import igl

import skimage
import trimesh
from trimesh.sample import sample_surface # https://trimesh.org/trimesh.sample.html#trimesh.sample.sample_surface

import torch
from torch import nn




class MeshEvaluator:
    def __init__(self, 
                 N_pointcloud: int = 100_000, 
                 N_cube: int = 128, 
                 min_max_range: tuple[float, float] | list[float, float] = (-0.5, 0.5), 
                 winding_number_threshold: float = 0.5, 
                 hash_resolution: int = 512,
                 verbose: bool = False,
                 random_seed: int = 42,
                 Fscore_thresholds: np.ndarray = np.linspace(1./1000, 1, 1000),
                 ):
        """
        Args:
            N_pointcloud (int): Number of points to sample from the predicted mesh.
            N_cube (int): Number of points to sample from the ground truth mesh.
            min_max_range (tuple[float, float] | list[float, float]): Range of the cube.
            winding_number_threshold (float): Threshold for the winding number.
            hash_resolution (int): Resolution of the hash grid.
            verbose (bool): Whether to print verbose output.
            random_seed (int): Random seed for the sampling.
            Fscore_thresholds (np.ndarray): Thresholds for the F-score.


        How to use:
        mesh_evaluator = MeshEvaluator(
            N_pointcloud=100_000,
            N_cube=128,
            min_max_range=[-0.5, 0.5],
            winding_number_threshold=0.5,
            hash_resolution=512,
            verbose=False,
            random_seed=42,
        )
        metrics, debug_metrics_dict = mesh_evaluator.eval_mesh(mesh, gt_mesh)
        """
        self.N_pointcloud = N_pointcloud
        self.N_cube = N_cube
        self.min_max_range = min_max_range
        self.winding_number_threshold = winding_number_threshold
        self.hash_resolution = hash_resolution
        self.verbose = verbose
        self.random_seed = random_seed
        self.Fscore_thresholds = Fscore_thresholds

    def eval_mesh(self, pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh):
        ### Sample surface points from the predicted mesh. And get the normals of the points
        pred_points, pred_face_index = sample_surface(pred_mesh, self.N_pointcloud, seed=self.random_seed)
        pred_normals = pred_mesh.face_normals[pred_face_index]

        ### Sample surface points from the ground truth mesh. And get the normals of the points
        gt_points, gt_face_index = sample_surface(gt_mesh, self.N_pointcloud, seed=self.random_seed)
        gt_normals = gt_mesh.face_normals[gt_face_index]


        ### -------------------------- Evaluation -------------------------- ###
        ### Evaluate the pointcloud
        eval_dict, debug_dict = self.eval_pointcloud(pred_points, gt_points, pred_normals, gt_normals)

        ### Calculate the VIoU and Dice score
        viou, dice = self.calc_viou(pred_mesh, gt_mesh, get_dice=True)

        ### Add VIoU and Dice score to the evaluation dictionary
        ### Higher is better
        eval_dict['VIoU'] = viou 
        eval_dict['Dice'] = dice

        return eval_dict, debug_dict
        
    
    def eval_pointcloud(self,
                        pred_pointcloud: np.ndarray, gt_pointcloud: np.ndarray, 
                        pred_normals: np.ndarray, gt_normals: np.ndarray,
                        ):
        
        ### Calculate the Chamfer distance between the predicted pointcloud and the ground truth pointcloud
        CD_dict, all_dict = self.calc_chamfer_distance(pred_pointcloud, gt_pointcloud, pred_normals, gt_normals)

        ### Combine the Chamfer distance dictionary
        eval_dict = {**CD_dict}

        return eval_dict, all_dict



    def calc_chamfer_distance(self, 
                              pred_pointcloud: np.ndarray, gt_pointcloud: np.ndarray, 
                              pred_normals: np.ndarray, gt_normals: np.ndarray,
                              ) -> tuple[dict, dict]:
        ### Calculate the Chamfer distance between the predicted pointcloud and the ground truth pointcloud
        occ_cd = OccNet_CD(pred_pointcloud, gt_pointcloud, pred_normals, gt_normals)
        deepSDF_cd = DeepSDF_CD(pred_pointcloud, gt_pointcloud)

        ### Chamfer distance dictionary
        CD_dict = {
            'OccNet chamfer-L1': occ_cd['chamfer-L1'], ### Lower is better
            'OccNet chamfer-L2': occ_cd['chamfer-L2'], ### Lower is better
            'OccNet normal consistency': occ_cd['normal consistency'],  ### Higher is better
            'OccNet f-scores': occ_cd['f-scores'][9], ### threshold = 1.0% Higher is better
            'DeepSDF chamfer': deepSDF_cd, ### Lower is better
        }

        ### Put all other information in one dictionary
        all_dict = {**occ_cd, 'DeepSDF chamfer':deepSDF_cd}

        return CD_dict, all_dict


    def calc_viou(self, pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh, get_dice: bool = True) -> tuple[float, float]:
        cube_points, gt_occ = get_cube_winding_number(gt_mesh, N=self.N_cube, min_max_norm=self.min_max_range, 
                                                      winding_number_threshold=self.winding_number_threshold, verbose=self.verbose)
        pred_occ = check_mesh_contains(pred_mesh, cube_points, hash_resolution=self.hash_resolution)
        viou, dice = compute_iou(pred_occ, gt_occ, get_dice=get_dice)

        return viou, dice
    





### ----------------------------------------------- ###
### Metric
### ----------------------------------------------- ###

### --------------------------------------------------------- Occupancy Chamfer Distance --------------------------------------------------------- ###
### After SciPy v1.6.0, cKDTree is functionally identical to KDTree. 
from scipy.spatial import KDTree
import numpy as np
import trimesh

def OccNet_CD(pred_pointcloud: np.ndarray, gt_pointcloud: np.ndarray, 
              pred_normals: np.ndarray = None, gt_normals: np.ndarray = None,
              Fscore_thresholds: np.ndarray = np.linspace(1./1000, 1, 1000)
              ) -> dict:
    """
    Compute the Chamfer Distance between two occupancy grids based on Occupancy Networks paper.
    """
    ### -------------------------- Completeness -------------------------- ###
    ### Completeness: how far are the points of the target point cloud from thre predicted point cloud
    ### Calculate KDTree(pred_pointcloud) then query(gt_pointcloud)
    ### For each point in `gt_pointcloud`, the KDTree finds the closest point in `pred_pointcloud`.
   
    # completeness_raw, completeness_normals = distance_p2p(gt_pointcloud, gt_normals, pred_pointcloud, pred_normals)
    # accuracy_raw, accuracy_normals = distance_p2p(pred_pointcloud, pred_normals, gt_pointcloud, gt_normals)

    # completeness = completeness_raw.mean()
    # accuracy = accuracy_raw.mean()
    # completeness2 = (completeness_raw**2).mean()
    # accuracy2 = (accuracy_raw**2).mean()
    # completeness_normals = completeness_normals.mean()
    # accuracy_normals = accuracy_normals.mean()  
    
    
    
    
    completeness, completeness_normals = distance_p2p(
        points_src=gt_pointcloud, normals_src=gt_normals,
        points_tgt=pred_pointcloud, normals_tgt=pred_normals
    )
    recall = get_threshold_percentage(completeness, Fscore_thresholds)
    completeness2 = completeness**2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()


    ### -------------------------- Accuracy -------------------------- ###
    ### Accuracy: how far are the points of the predicted pred_pointcloud from the target pred_pointcloud
    ### Calculate KDTree(gt_pointcloud) then query(pred_pointcloud)
    ### For each point in `pred_pointcloud`, the KDTree finds the closest point in `gt_pointcloud`.
    accuracy, accuracy_normals = distance_p2p(
        points_src=pred_pointcloud, normals_src=pred_normals,
        points_tgt=gt_pointcloud, normals_tgt=gt_normals
    )
    precision = get_threshold_percentage(accuracy, Fscore_thresholds)
    accuracy2 = accuracy**2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()


    ### -------------------------- Chamfer distance -------------------------- ###
    chamferL1 = 0.5 * (completeness + accuracy)
    chamferL2 = 0.5 * (completeness2 + accuracy2)

    ### -------------------------- Normals correctness -------------------------- ###
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals


    # ### -------------------------- F-score -------------------------- ###
    # recall = get_threshold_percentage(completeness, Fscore_thresholds)
    # precision = get_threshold_percentage(accuracy, Fscore_thresholds)
    F_scores = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]

    # recall = get_threshold_percentage(completeness_raw, Fscore_thresholds)
    # precision = get_threshold_percentage(accuracy_raw, Fscore_thresholds)
    # F_scores = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]
    
    ### Output dictionary
    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'chamfer-L1': chamferL1,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer-L2': chamferL2,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normal consistency': normals_correctness,
        'f-scores': F_scores,
    }

    return out_dict



def distance_p2p(points_src, normals_src, points_tgt, normals_tgt) -> tuple[np.ndarray, np.ndarray]:
    ''' Computes minimal distances of each point in points_src to points_tgt.

    TL;DR: For each point in `source`, the KDTree finds the closest point in `target`.

    Args:
        points_src (numpy array): source points. These are the "source" points for which you are querying the nearest neighbors from the KDTree.
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points. This is the set of points used to build the KDTree, representing the "target" points in the space.
        normals_tgt (numpy array): target normals

    Returns:
        dist (numpy array): For each point in points_src, the KDTree finds the closest point in points_tgt and calculates the distance between them. This distance is stored in the dist array.
        normals_dot_product (numpy array): The dot product between the normals of the source and target points.
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to method not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def distance_p2m(points, mesh):
    ''' Compute minimal distances of each point in points to mesh.

    Args:
        points (numpy array): points array
        mesh (trimesh): mesh

    '''
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


### From Shape as Points
def get_threshold_percentage(dist: np.ndarray, thresholds: np.ndarray) -> list[float]:
    """Evaluates a point cloud by calculating percentage of points within thresholds.
    
    Args:
        dist (np.ndarray): Calculated distances between points.
        thresholds (np.ndarray): Threshold values for the F-score calculation. Defaults to np.linspace(1./1000, 1, 1000)
    
    Returns:
        list[float]: For each threshold t, returns mean percentage of points where dist <= t
    """
    in_threshold = [(dist <= t).mean() for t in thresholds]

    return in_threshold

### --------------------------------------------------------- DeepSDF --------------------------------------------------------- ###
### The implementation of DeepSDF chamfer distance is not clear with the Occupancy Networks paper.
### DeepSDF says it normalizes by the number of points but the code does not seem to do that.
### https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/metrics/chamfer.py#L9

def DeepSDF_CD(pred_pointcloud: np.ndarray, gt_pointcloud: np.ndarray) -> float:
    # one direction
    gen_points_kd_tree = KDTree(pred_pointcloud)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_pointcloud)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_pointcloud)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_pointcloud)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer






### --------------------------------------------------------- VIoU --------------------------------------------------------- ###
# + Converted TriangleHash from c++ to python
# + Added get_cube_winding_number to get the cube points and the winding number for each point in volumetric manner.
    
### How to use:
# def get_iou(pred_mesh: trimesh.Trimesh, 
#             gt_mesh: trimesh.Trimesh, 
#             N: int = 128,
#             min_max_norm: tuple[float, float] = (-0.5, 0.5), 
#             winding_number_threshold: float = 0.5, 
#             hash_resolution: int = 512, 
#             verbose: bool = True,
#             ) -> float:
#     cube_points, gt_occ = get_cube_winding_number(gt_mesh, N=N, min_max_norm=min_max_norm, winding_number_threshold=winding_number_threshold, verbose=verbose)
#     pred_occ = check_mesh_contains(pred_mesh, cube_points, hash_resolution=hash_resolution)
#     viou = compute_iou(pred_occ, gt_occ)
#     return viou

def compute_iou(occ1: np.ndarray, occ2: np.ndarray, get_dice: bool = False) -> float | tuple[float, float]:
    """Computes the Intersection over Union (IoU) and Dice score for two sets of occupancy values.
    
    Args:
        occ1 (np.ndarray): First set of occupancy values
        occ2 (np.ndarray): Second set of occupancy values
        get_dice (bool): Whether to return Dice score along with IoU. Defaults to False
    
    Returns:
        float | tuple[float, float]: A tuple containing:
            - IoU value between the two occupancy sets
            - Dice score (if get_dice=True) or just IoU value again (if get_dice=False)
    """
    ### Convert inputs to numpy arrays
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    ### Reshape to 2D if needed
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    ### Convert to boolean masks
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    ### Compute intersection and union areas
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    ### Calculate IoU
    iou = area_intersect / area_union


    if get_dice:
        ### Calculate Dice score
        ### Dice = 2|X∩Y|/(|X|+|Y|)
        dice = 2 * area_intersect / (occ1.sum(axis=-1) + occ2.sum(axis=-1))
        return iou, dice
        
    else:
        return iou



def get_cube_winding_number(mesh, 
                            N: int = 128,
                            min_max_norm: tuple[float, float] = (-1, 1),
                            winding_number_threshold: float = 0.5,
                            verbose: bool = False,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Get the cube points and the winding number for each point in volumetric manner.

    Args:
        mesh (trimesh.Trimesh): The mesh to get the winding number for. 
        N (int): The number of points in each dimension. Defaults to 128
        min_max_norm (tuple[float, float]): The minimum and maximum values for the cube points. Defaults to (-1, 1)
        winding_number_threshold (float): The threshold for the winding number. Defaults to 0.5
        verbose (bool): Whether to print the verbose output. Defaults to False

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - cube_points: Array of shape (N^3, 3) containing the 3D coordinates of points in the cube
            - occupancies_winding: Boolean array of shape (N^3,) indicating inside (True) or outside (False) points
    """
    ### Get the cube points
    points_range = np.linspace(min_max_norm[0], min_max_norm[1], N)
    cube_points = np.meshgrid(points_range, points_range, points_range, indexing="ij")
    cube_points = np.stack(cube_points, axis=-1).reshape(-1, 3) ### (N^3, 3)

    ### Calculate winding number
    inside_surface_values = igl.fast_winding_number_for_meshes(mesh.vertices, mesh.faces, cube_points) ### shape (N^3,)
    
    ### Create labels
    occupancies_winding = np.piecewise(
        inside_surface_values,
        [inside_surface_values < winding_number_threshold, inside_surface_values >= winding_number_threshold],
        [0, 1], ### 0: outside, 1: inside
    ).astype(bool) ### shape (N^3,)

    if verbose:
        print(f"cube_points: {cube_points.shape=}, {cube_points.max(axis=0)=}, {cube_points.min(axis=0)=}")
        print(f"inside_surface_values: {inside_surface_values.shape=}, {inside_surface_values.max()=}, {inside_surface_values.min()=}")
        print(f"occupancies_winding: {occupancies_winding.shape=}, {occupancies_winding.sum()=}")

    return cube_points, occupancies_winding



def check_mesh_contains(mesh, points, hash_resolution: int = 512):
    """Check if the points are inside the mesh

    Args:
        mesh (trimesh.Trimesh): The mesh to check the points inside
        points (np.ndarray): The points to check
        hash_resolution (int): The resolution of the hash grid. Defaults to 512

    Returns:
        np.ndarray: Boolean array of shape (n_points,) indicating inside (True) or outside (False) points
    """
    intersector = MeshIntersector(mesh, hash_resolution)
    contains = intersector.query(points)
    return contains


class MeshIntersector:
    """Mesh Intersector
    
    Args:
        mesh (trimesh.Trimesh): The mesh to check the points inside
        resolution (int): The resolution of the hash grid. Defaults to 512
    """
    def __init__(self, mesh, resolution: int = 512):
        triangles = mesh.vertices[mesh.faces].astype(np.float64)
        n_tri = triangles.shape[0]

        self.resolution = resolution
        self.bbox_min = triangles.reshape(3 * n_tri, 3).min(axis=0)
        self.bbox_max = triangles.reshape(3 * n_tri, 3).max(axis=0)
        # Tranlate and scale it to [0.5, self.resolution - 0.5]^3
        self.scale = (resolution - 1) / (self.bbox_max - self.bbox_min)
        self.translate = 0.5 - self.scale * self.bbox_min

        self._triangles = triangles = self.rescale(triangles)
        # assert(np.allclose(triangles.reshape(-1, 3).min(0), 0.5))
        # assert(np.allclose(triangles.reshape(-1, 3).max(0), resolution - 0.5))

        triangles2d = triangles[:, :, :2]
        self._tri_intersector2d = TriangleIntersector2d(
            triangles2d, resolution)

    def query(self, points):
        # Rescale points
        points = self.rescale(points)

        # placeholder result with no hits we'll fill in later
        # contains = np.zeros(len(points), dtype=np.bool)
        contains = np.zeros(len(points), dtype=bool) ### Update for numpy > 1.20

        # cull points outside of the axis aligned bounding box
        # this avoids running ray tests unless points are close
        inside_aabb = np.all(
            (0 <= points) & (points <= self.resolution), axis=1)
        if not inside_aabb.any():
            return contains

        # Only consider points inside bounding box
        mask = inside_aabb
        points = points[mask]

        # Compute intersection depth and check order
        points_indices, tri_indices = self._tri_intersector2d.query(points[:, :2])

        triangles_intersect = self._triangles[tri_indices]
        points_intersect = points[points_indices]

        depth_intersect, abs_n_2 = self.compute_intersection_depth(
            points_intersect, triangles_intersect)

        # Count number of intersections in both directions
        smaller_depth = depth_intersect >= points_intersect[:, 2] * abs_n_2
        bigger_depth = depth_intersect < points_intersect[:, 2] * abs_n_2
        points_indices_0 = points_indices[smaller_depth]
        points_indices_1 = points_indices[bigger_depth]

        nintersect0 = np.bincount(points_indices_0, minlength=points.shape[0])
        nintersect1 = np.bincount(points_indices_1, minlength=points.shape[0])
        
        # Check if point contained in mesh
        contains1 = (np.mod(nintersect0, 2) == 1)
        contains2 = (np.mod(nintersect1, 2) == 1)
        if (contains1 != contains2).any():
            print('Warning: contains1 != contains2 for some points.')
        contains[mask] = (contains1 & contains2)
        return contains

    def compute_intersection_depth(self, points, triangles):
        t1 = triangles[:, 0, :]
        t2 = triangles[:, 1, :]
        t3 = triangles[:, 2, :]

        v1 = t3 - t1
        v2 = t2 - t1
        # v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
        # v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)

        normals = np.cross(v1, v2)
        alpha = np.sum(normals[:, :2] * (t1[:, :2] - points[:, :2]), axis=1)

        n_2 = normals[:, 2]
        t1_2 = t1[:, 2]
        s_n_2 = np.sign(n_2)
        abs_n_2 = np.abs(n_2)

        mask = (abs_n_2 != 0)
    
        depth_intersect = np.full(points.shape[0], np.nan)
        depth_intersect[mask] = \
            t1_2[mask] * abs_n_2[mask] + alpha[mask] * s_n_2[mask]

        # Test the depth:
        # TODO: remove and put into tests
        # points_new = np.concatenate([points[:, :2], depth_intersect[:, None]], axis=1)
        # alpha = (normals * t1).sum(-1)
        # mask = (depth_intersect == depth_intersect)
        # assert(np.allclose((points_new[mask] * normals[mask]).sum(-1),
        #                    alpha[mask]))
        return depth_intersect, abs_n_2

    def rescale(self, array):
        array = self.scale * array + self.translate
        return array


class TriangleIntersector2d:
    def __init__(self, triangles, resolution=128):
        self.triangles = triangles
        self.tri_hash = TriangleHash(triangles, resolution)

    def query(self, points):
        point_indices, tri_indices = self.tri_hash.query(points)
        point_indices = np.array(point_indices, dtype=np.int64)
        tri_indices = np.array(tri_indices, dtype=np.int64)
        points = points[point_indices]
        triangles = self.triangles[tri_indices]
        mask = self.check_triangles(points, triangles)
        point_indices = point_indices[mask]
        tri_indices = tri_indices[mask]
        return point_indices, tri_indices

    def check_triangles(self, points, triangles):
        # contains = np.zeros(points.shape[0], dtype=np.bool)
        contains = np.zeros(points.shape[0], dtype=bool) ### Update for numpy > 1.20

        A = triangles[:, :2] - triangles[:, 2:]
        A = A.transpose([0, 2, 1])
        y = points - triangles[:, 2]

        detA = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        
        mask = (np.abs(detA) != 0.)
        A = A[mask]
        y = y[mask]
        detA = detA[mask]

        s_detA = np.sign(detA)
        abs_detA = np.abs(detA)

        u = (A[:, 1, 1] * y[:, 0] - A[:, 0, 1] * y[:, 1]) * s_detA
        v = (-A[:, 1, 0] * y[:, 0] + A[:, 0, 0] * y[:, 1]) * s_detA

        sum_uv = u + v
        contains[mask] = (
            (0 < u) & (u < abs_detA) & (0 < v) & (v < abs_detA)
            & (0 < sum_uv) & (sum_uv < abs_detA)
        )
        return contains


class TriangleHash:
    """
    A spatial hash structure for triangles to enable efficient point-in-triangle queries

    This is a modified version of the TriangleHash class from Occupancy Networks.
    Converted from `triangle_hash.pyx` c++ to python.
    
    Args:
        triangles (np.ndarray): Array of triangles with shape (n_triangles, 3, 2)
        resolution (int): Resolution of the spatial hash grid
    """
    def __init__(self, triangles: np.ndarray, resolution: int):
        self.spatial_hash = [[] for _ in range(resolution * resolution)]
        self.resolution = resolution
        self._build_hash(triangles)


    def _build_hash(self, triangles: np.ndarray) -> None:
        """Build the spatial hash structure
        
        Args:
            triangles (np.ndarray): Array of triangles with shape (n_triangles, 3, 2)
        """
        assert triangles.shape[1] == 3 and triangles.shape[2] == 2, f"Invalid triangle shape: {triangles.shape}"
        
        n_tri = triangles.shape[0]
        
        for i_tri in range(n_tri):
            ### Compute bounding box
            bbox_min = np.min(triangles[i_tri], axis=0)
            bbox_max = np.max(triangles[i_tri], axis=0)
            
            ### Clamp to grid boundaries
            bbox_min = np.clip(bbox_min, 0, self.resolution - 1).astype(int)
            bbox_max = np.clip(bbox_max, 0, self.resolution - 1).astype(int)
            
            ### Find all voxels where bounding box intersects
            for x in range(bbox_min[0], bbox_max[0] + 1):
                for y in range(bbox_min[1], bbox_max[1] + 1):
                    spatial_idx = self.resolution * x + y
                    self.spatial_hash[spatial_idx].append(i_tri)


    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Query the spatial hash structure for points
        
        Args:
            points (np.ndarray): Array of points with shape (n_points, 2)
            
        Returns:
            tuple[np.ndarray, np.ndarray]: Arrays of point indices and triangle indices that potentially intersect
        """
        assert points.shape[1] == 2, f"Invalid points shape: {points.shape}"
        
        points_indices = []
        tri_indices = []
        
        for i_point, point in enumerate(points):
            x, y = point.astype(int)
            
            if not (0 <= x < self.resolution and 0 <= y < self.resolution):
                continue
                
            spatial_idx = self.resolution * x + y
            for i_tri in self.spatial_hash[spatial_idx]:
                points_indices.append(i_point)
                tri_indices.append(i_tri)
                
        return np.array(points_indices, dtype=np.int32), np.array(tri_indices, dtype=np.int32)


