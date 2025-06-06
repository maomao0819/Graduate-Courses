import cv2
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd
import re
import imageio.v2 as imageio
from tqdm import tqdm


def load_point_cloud(points3D_df):

    xyz = np.vstack(points3D_df["XYZ"])
    rgb = np.vstack(points3D_df["RGB"]) / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])  # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # R, G, B
    return axes


def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler("xyz", rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat


def update_cube():
    global cube, cube_vertices, R_euler, t, scale

    transform_mat = get_transform_mat(R_euler, t, scale)

    transform_vertices = (
        transform_mat @ np.concatenate([cube_vertices.transpose(), np.ones([1, cube_vertices.shape[0]])], axis=0)
    ).transpose()

    cube.vertices = o3d.utility.Vector3dVector(transform_vertices)
    cube.compute_vertex_normals()
    cube.paint_uniform_color([1, 0.706, 0])
    vis.update_geometry(cube)


def toggle_key_shift(vis, action, mods):
    global shift_pressed
    if action == 1:  # key down
        shift_pressed = True
    elif action == 0:  # key up
        shift_pressed = False
    return True


def update_tx(vis):
    global t, shift_pressed
    t[0] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_ty(vis):
    global t, shift_pressed
    t[1] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_tz(vis):
    global t, shift_pressed
    t[2] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_rx(vis):
    global R_euler, shift_pressed
    R_euler[0] += -1 if shift_pressed else 1
    update_cube()


def update_ry(vis):
    global R_euler, shift_pressed
    R_euler[1] += -1 if shift_pressed else 1
    update_cube()


def update_rz(vis):
    global R_euler, shift_pressed
    R_euler[2] += -1 if shift_pressed else 1
    update_cube()


def update_scale(vis):
    global scale, shift_pressed
    scale += -0.05 if shift_pressed else 0.05
    update_cube()


class Point3D:
    def __init__(self, color, position):
        self.color = color
        self.p3D = position
        self.p2D = [0, 0]
        self.dist = 0

    # sort by distance
    def __lt__(self, other):
        return self.dist < other.dist


if __name__ == "__main__":

    # Construct points in 6 surfaces
    front = np.array([[i + 1, j + 1, 0] for i in range(8) for j in range(8)]) / 10
    back = front.copy()
    back[:, 2] = 0.9
    left = np.array([[0, i, j] for i in range(10) for j in range(10)]) / 10
    right = left.copy()
    right[:, 0] = 0.9
    top = np.array([[i + 1, 0, j] for i in range(8) for j in range(10)]) / 10
    bottom = top.copy()
    bottom[:, 1] = 0.9
    cube = np.array([bottom, front, top, back, left, right])

    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.6, 0.2], [0.6, 0.2, 0.6], [0.5, 0.5, 0.5]]

    points3D = []
    for planeIdx in range(cube.shape[0]):
        color = colors[planeIdx]
        for pointIdx in range(cube[planeIdx].shape[0]):
            p = Point3D(color, cube[planeIdx][pointIdx])
            points3D.append(p)

    # camera parameters
    intrinsic_parameters = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
    distortion_parameters = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352])
    center_p2D = np.array([[0, 0, 1]]).reshape((3, 1))

    # load dataframe
    images_df = pd.read_pickle("data/images.pkl")
    trans = np.load("translation_pred.npy")
    rota = R.from_quat(np.load("rotation_pred.npy")).as_matrix()

    # create cube transform matrix
    if os.path.exists("cube_transform_mat.npy") == False:
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

        # load cube
        cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        cube_vertices = np.asarray(cube.vertices).copy()
        vis.add_geometry(cube)

        R_euler = np.array([0, 0, 0]).astype(float)
        t = np.array([0, 0, 0]).astype(float)
        scale = 1.0
        update_cube()

        # just set a proper initial camera view
        vc = vis.get_view_control()
        vc_cam = vc.convert_to_pinhole_camera_parameters()
        initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
        initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
        initial_cam[-1, -1] = 1.0
        setattr(vc_cam, "extrinsic", initial_cam)
        vc.convert_from_pinhole_camera_parameters(vc_cam)

        # set key callback
        shift_pressed = False
        vis.register_key_action_callback(340, toggle_key_shift)
        vis.register_key_action_callback(344, toggle_key_shift)
        vis.register_key_callback(ord("A"), update_tx)
        vis.register_key_callback(ord("S"), update_ty)
        vis.register_key_callback(ord("D"), update_tz)
        vis.register_key_callback(ord("Z"), update_rx)
        vis.register_key_callback(ord("X"), update_ry)
        vis.register_key_callback(ord("C"), update_rz)
        vis.register_key_callback(ord("V"), update_scale)

        print("[Keyboard usage]")
        print("Translate along X-axis\tA / Shift+A")
        print("Translate along Y-axis\tS / Shift+S")
        print("Translate along Z-axis\tD / Shift+D")
        print("Rotate    along X-axis\tZ / Shift+Z")
        print("Rotate    along Y-axis\tX / Shift+X")
        print("Rotate    along Z-axis\tC / Shift+C")
        print("Scale                 \tV / Shift+V")

        vis.run()
        vis.destroy_window()

        """
        print('Rotation matrix:\n{}'.format(R.from_euler('xyz', R_euler, degrees=True).as_matrix()))
        print('Translation vector:\n{}'.format(t))
        print('Scale factor: {}'.format(scale))
        """

        np.save("cube_transform_mat.npy", get_transform_mat(R_euler, t, scale))
        np.save("cube_vertices.npy", np.asarray(cube.vertices))

    transform_mat = np.load("cube_transform_mat.npy")

    # load point
    valid_img_dict = {}
    for image_idx in range(len(images_df["NAME"])):
        image_name = images_df["NAME"][image_idx]
        if "valid" in image_name:
            index = int(float(re.findall(r"-?\d+\.?\d*", image_name)[0]))
            valid_img_dict[image_idx] = index

    # key : value -> rotaIdx : image_idx
    valid_img_dict = sorted(valid_img_dict.items(), key=lambda x: x[1])

    os.makedirs("frames", exist_ok=True)

    frames = []
    valid_img_dict_pbar = tqdm((valid_img_dict), total=len(valid_img_dict))
    for save_idx, (pred_idx, image_idx) in enumerate(valid_img_dict_pbar):
        t = trans[pred_idx]
        r = rota[pred_idx]

        # WCS project to CCS
        for p3D in points3D:
            pos = np.array(p3D.p3D).reshape(1, 3)
            transform_p3D = (
                transform_mat @ np.concatenate([pos.transpose(), np.ones([1, pos.shape[0]])], axis=0)
            ).transpose()
            center_p3D = np.linalg.inv(r) @ center_p2D + t.reshape((3, 1))

            p3D.dist = np.linalg.norm(transform_p3D - center_p3D.T)
            X_T = transform_p3D.T - t.reshape((3, 1))
            lambda_u = intrinsic_parameters @ r @ X_T

            p2D = (lambda_u / lambda_u[2]).T[0][:2]
            p3D.p2D = p2D

        points3D.sort(reverse=True)

        img = cv2.imread(os.path.join("data", "frames", "valid_img" + str(image_idx) + ".jpg"))

        # cube
        for p3D in points3D:
            cv2.circle(
                img,
                center=(int(p3D.p2D[0]), int(p3D.p2D[1])),
                radius=10,
                color=(p3D.color[0] * 255, p3D.color[1] * 255, p3D.color[2] * 255),
                thickness=-1,
            )

        frames.append(img)
        img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
        cv2.imwrite(os.path.join("frames", str(save_idx) + ".jpg"), img)

        valid_img_dict_pbar.set_description(f"Image [{save_idx+1}/{len(valid_img_dict)}]")

    # output to video and gif
    writer = cv2.VideoWriter("AR.mp4", cv2.VideoWriter_fourcc("m", "p", "4", "v"), 15, np.shape(img)[:-1][::-1])
    for frame in frames:
        writer.write(frame)
    writer.release()
    frames = []
    for read_idx in range(save_idx):
        frames.append(imageio.imread(os.path.join("frames", str(read_idx) + ".jpg")))
    imageio.mimsave("AR.gif", frames, "GIF")
