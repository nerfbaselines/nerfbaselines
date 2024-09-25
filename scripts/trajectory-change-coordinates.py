#!/usr/bin/env python3

import numpy as np
from argparse import ArgumentParser
from nerfbaselines.datasets import load_dataset
from nerfbaselines.io import load_trajectory, save_trajectory
from nerfbaselines.utils import apply_transform


def _estimate_similarity_transform(points1, points2):
    # Ensure the points are in the correct shape (N, 3)
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    
    if points1.shape != points2.shape:
        raise ValueError("The input points must have the same shape")
    
    # Calculate the centroids of each set of points
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    
    # Center the points around their centroids
    centered1 = points1 - centroid1
    centered2 = points2 - centroid2

    # Compute cross-covariance matrix
    H = np.dot(centered1.T, centered2)
    
    # SVD decomposition of the covariance matrix
    U, S, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Ensure a right-handed coordinate system (reflection correction)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1  # Fix the last row instead of second
        R = np.dot(Vt.T, U.T)

    # Compute the scaling factor
    scale = S.sum() / np.sum(centered1 ** 2)

    # Get the similarity transformation matrix (4x4 homogeneous matrix)
    P = np.eye(4, dtype=np.float32)
    P[:3, :3] = R * scale
    P[:3, 3] = centroid2 - np.dot(centroid1, R.T) * scale
    P[3, 3] = 1
    
    return P


parser = ArgumentParser()
parser.add_argument('--trajectory', help='Path to the trajectory file (.json) with the old coordinates')
parser.add_argument('--data', help='Old dataset path used when the trajectory was generated')
parser.add_argument('--new-data', help='New dataset path with the new coordinates')
parser.add_argument('--output', help='Path to the output file trajectory with the new coordinates')
args = parser.parse_args()

with open(args.trajectory, 'r') as f:
    trajectory = load_trajectory(f)

# By using split=None, we can use all images in the datasets
old_dataset = load_dataset(args.data, split=None, load_features=False)
new_dataset = load_dataset(args.new_data, split=None, load_features=False)

old_cameras = old_dataset['cameras']#[np.argsort(old_dataset['image_paths'])]
assert len(old_cameras) == len(new_dataset['cameras']), 'The number of cameras must be the same in both datasets'
new_cameras = new_dataset['cameras']#[np.argsort(new_dataset['image_paths'])]

# Compute similarity transormation 
old_points = old_cameras.poses[..., :3, 3]
new_points = new_cameras.poses[..., :3, 3]
transform = _estimate_similarity_transform(old_points, new_points)

# Apply transformation to the trajectory
trajectory["frames"] = [{
    **frame,
    "pose": apply_transform(transform, frame['pose'])
} for frame in trajectory["frames"]]
if trajectory.get("source") is not None:
    trajectory["source"]["keyframes"] = [{
        **keyframe,
        "pose": apply_transform(transform, keyframe['pose'])
    } for keyframe in trajectory["source"]["keyframes"]]
with open(args.output, 'w') as f:
   save_trajectory(trajectory, f)
print(f'Trajectory with new coordinates saved at {args.output}')
