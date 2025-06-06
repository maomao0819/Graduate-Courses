import open3d as o3d
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
import pandas as pd
import re

def load_point_cloud(points3D_df):

    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB']) / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd

def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])          # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # R, G, B
    return axes

def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat

def get_sorted_img_idx(images_df, set_name='train'):
    # load point

    # image_dict = {}
    # for image_idx in range(len(images_df["NAME"])):
    #     image_name = images_df["NAME"][image_idx]
    #     # if image_name.find(set_name) != -1:
    #     if set_name in image_name:
    #         index = int(float(re.findall(r'-?\d+\.?\d*', image_name)[0]))
    #         image_dict[image_idx] = index
    # image_dict = sorted(image_dict.items(), key=lambda x:x[1])
    # images_index = []
    # for (idx, nums) in image_dict:
    #     images_index.append(idx)

    # n_points = []
    # for image_idx in range(len(images_df["NAME"])):
    #     image_name = images_df["NAME"][image_idx]
    #     if image_name.find(set_name) != -1:
    #         index = int(float(re.findall(r'-?\d+\.?\d*', image_name)[0]))
    #         n_points.append(index)
    # images_index = np.argsort(n_points)

    image_name_series = images_df["NAME"]
    n_points = [int(float(re.findall(r'-?\d+\.?\d*', image_name_series[image_idx])[0])) for image_idx in range(len(image_name_series)) if set_name in image_name_series[image_idx]]
    images_index = np.argsort(n_points)

    return images_index

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print('[Usage] python3 transform_cube.py /PATH/TO/points3D.txt')
    #     sys.exit(1)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # load point cloud
    points3D_df = pd.read_pickle("data/points3D.pkl")
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)

    # load axes
    axes = load_axes()
    vis.add_geometry(axes)

    images_df = pd.read_pickle("data/images.pkl")
    trans = np.load("translation_pred.npy")
    rota = R.from_quat(np.load("rotation_pred.npy")).as_matrix()


    images_index = get_sorted_img_idx(images_df, set_name='train')

    center_p2D = np.array([[0, 0, 1]]).reshape((3, 1))

    vector_origin = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).T

    center_p3Dlist = []
    for image_idx in images_index:
        # centor ,top left, top right, bottom left, bottom right
        fivePoints = []
        center_p3D = (np.linalg.inv(rota[image_idx]) @ center_p2D + trans[image_idx].reshape((3, 1)))
        
        length = 0.15
        width = 0.3

        vechori = length * (np.linalg.inv(rota[image_idx]) @ vector_origin)

        # camera center
        fivePoints.append(center_p3D.T[0])
        # image plane
        fivePoints.append(center_p3D.T[0] + vechori.T[0] + width*vechori.T[1] + width*vechori.T[2])
        fivePoints.append(center_p3D.T[0] + vechori.T[0] + width*vechori.T[1] - width*vechori.T[2])
        fivePoints.append(center_p3D.T[0] + vechori.T[0] - width*vechori.T[1] + width*vechori.T[2])
        fivePoints.append(center_p3D.T[0] + vechori.T[0] - width*vechori.T[1] - width*vechori.T[2])
        
        point_cloud = o3d.geometry.PointCloud()
        pint = np.array(np.array(fivePoints))
        point_cloud.points = o3d.utility.Vector3dVector(pint)
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [3, 4], [2, 4]]
        colors = [[0, 1, 0] for i in range(len(lines))]
        point_cloud.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        line = o3d.geometry.LineSet()
        line.lines = o3d.utility.Vector2iVector(lines)
        line.colors = o3d.utility.Vector3dVector(colors)
        line.points = o3d.utility.Vector3dVector(pint)
        vis.add_geometry(point_cloud)
        vis.add_geometry(line)
        center_p3Dlist.append(center_p3D.T[0])

    # trajectory
    center_p3Dlist = np.array(center_p3Dlist)
    lines = [[idx, idx+1] for idx in range(len(center_p3Dlist)-1)]
    colors = [[1, 0, 0] for i in range(len(lines))]

    trajectory_line = o3d.geometry.LineSet()
    trajectory_line.colors = o3d.utility.Vector3dVector(colors)
    trajectory_line.points = o3d.utility.Vector3dVector(center_p3Dlist)
    trajectory_line.lines = o3d.utility.Vector2iVector(lines)

    vis.add_geometry(trajectory_line)

    # just set a proper initial camera view
    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)

    vis.run()
    vis.destroy_window()