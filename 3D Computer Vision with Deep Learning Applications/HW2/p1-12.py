import os
from logging import raiseExceptions
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2
from tqdm import trange
import open3d as o3d
import sys
import re

def average(x):
    return list(np.mean(x, axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

# def log_bebug(x):
#     print(np.shape(x))
#     print(x)

def distance(vector, axis=None):
    return np.linalg.norm(vector, axis=axis)

def unit_vector(vector):
    return vector / distance(vector)

def cosine_angle(vector_1, vector_2):
    unit_vector_1 = unit_vector(vector_1)
    unit_vector_2 = unit_vector(vector_2)
    angle = np.arccos(np.clip(np.dot(unit_vector_1, unit_vector_2), -1.0, 1.0))
    return np.cos(angle)

def solve_length(p3D, v):
    # u, v, Rab, Rac, Rbc, Cab, Cac, Cbc, K1, K2
    Rab = distance(p3D[0] - p3D[1])
    Rac = distance(p3D[0] - p3D[2])
    Rbc = distance(p3D[1] - p3D[2])
    Cab = cosine_angle(v[0], v[1])
    Cac = cosine_angle(v[0], v[2])
    Cbc = cosine_angle(v[1], v[2])

    if np.abs(Rac) < 1e-6 or np.abs(Rab) < 1e-6:
        return -1, -1, -1, True

    K1 = (Rbc / Rac) ** 2
    K2 = (Rbc / Rab) ** 2

    # G0~G4
    G4 = (K1 * K2 - K1 - K2) ** 2 - 4 * K1 * K2 * Cbc ** 2
    G3 = 4 * (K1 * K2 - K1 - K2) * K2 * (1 - K1) * Cab + 4 * K1 * Cbc * ((K1 * K2 - K1 + K2) * Cac + 2 * K2 * Cab * Cbc)
    G2 = (2 * K2 * (1 - K1) * Cab) ** 2 + 2 * (K1 * K2 - K1 - K2) * (K1 * K2 + K1 - K2) + 4 * K1 * ((K1 - K2) * Cbc ** 2 + K1 * (1 - K2) * Cac ** 2 - 2 * (1 + K1) * K2 * Cab * Cac * Cbc)
    G1 = 4 * (K1 * K2 + K1 - K2) * K2 * (1 - K1) * Cab + 4 * K1 * ((K1 * K2 - K1 + K2) * Cac * Cbc + 2 * K1 * K2 * Cab * Cac ** 2)
    G0 = (K1 * K2 + K1 - K2) ** 2 - 4 * K1 ** 2 * K2 * Cac ** 2
    G = [G4, G3, G2, G1, G0]

    x = np.roots(G)
    x = x[np.isreal(x)].real

    # a
    a_square = Rab ** 2 / (1 + x ** 2 - 2 * x * Cab)
    a = np.sqrt(a_square)
    a = a[np.isreal(a)].real

    # y
    m = 1 - K1
    p = 2 * (K1 * Cac - x * Cbc)
    q = x ** 2 - K1
    m_prime = 1
    p_prime = 2 * (- x * Cbc)
    q_prime = x ** 2 * (1 - K2) + 2 * x * K2 * Cab - K2
    y = (m * q_prime - m_prime * q) / (p * m_prime - p_prime * m)        
    
    # b, c
    b = x * a
    c = y * a
    
    return a, b, c, False

def trilateration(P, D):
    P1 = P[0]
    P2 = P[1]
    P3 = P[2]
    r1 = D[0]
    r2 = D[1]
    r3 = D[2]

    p1 = np.array([0, 0, 0])
    p2 = np.array([P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2]])
    p3 = np.array([P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2]])

    v1 = p2 - p1
    v2 = p3 - p1
    
    Xn = unit_vector(v1)
    Zn = unit_vector(np.cross(v1, v2))
    Yn = np.cross(Xn, Zn)
    
    i = np.dot(Xn, v2)
    
    d = np.dot(Xn, v1)
    
    j = np.dot(Yn, v2)
    
    X = ((r1 ** 2) - (r2 ** 2) + (d ** 2)) / (2 * d)
    Y = (((r1 ** 2) - (r3 ** 2) + (i ** 2) + (j ** 2)) / (2 * j)) - ((i / j) * (X))
    Z1 = np.sqrt(max(0, r1 ** 2 - X ** 2 - Y ** 2))
    Z2 = -Z1
    
    K1 = P1 + X * Xn + Y * Yn + Z1 * Zn
    K2 = P1 + X * Xn + Y * Yn + Z2 * Zn

    return K1, K2

def ransac(points2D, points3D, n_sample=3):
    n_point = points2D.shape[0]
    choice = np.random.choice(n_point, size=n_sample, replace=False)
    idx_sample = np.full(n_point, False)
    idx_unsample = np.full(n_point, True)
    idx_sample[choice] = True
    idx_unsample[choice] = False
    check_X = np.array([points3D[idx_unsample][0]])
    check_V = np.array([points2D[idx_unsample][0]])
    return points2D[idx_sample], points2D[idx_unsample], points3D[idx_sample], points3D[idx_unsample], check_V, check_X

def get_2D3Dcorrespondences(query, model, intrinsic_parameters, distortion_parameters):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query, desc_model, k=2)

    gmatches = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0, 2))
    points3D = np.empty((0, 3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D, kp_query[query_idx]))
        points3D = np.vstack((points3D, kp_model[model_idx]))

    points2D = cv2.undistortPoints(points2D, intrinsic_parameters, distortion_parameters)[:, 0]
    return points2D, points3D

def get_X_T(p3D, translation):
    return p3D.T - translation.reshape((3, 1))

def solve_p3p(p2D, p3D, check_V, checkX):
    check_V = np.insert(check_V, check_V.shape[1], values = np.ones((1, check_V.shape[0])), axis = 1)
    
    v = np.insert(p2D, p2D.shape[1], values = np.ones((1, p2D.shape[0])), axis = 1)
    
    a, b, c, should_continue = solve_length(p3D, v)
    if should_continue:
        return 0, 0, True

    rotation_best = []
    translation_best = []
    min_dist = np.inf

    for root_idx in range(len(a)):
        # T
        translation = [T1, T2] = trilateration(p3D, [a[root_idx], b[root_idx], c[root_idx]])
        translation = np.array(translation)
        
        lambda_signs = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
        for translation_idx in range(translation.shape[0]):
            # lambda
            lambdas = lambda_signs * distance(p3D.T - translation[translation_idx].reshape((3, 1)), axis = 0) / distance(v.T, axis = 0)

            for each_lambda in lambdas:
                X_T = get_X_T(p3D, translation[translation_idx])
                try:
                    rotation = (each_lambda * v.T) @ np.linalg.inv(X_T)
                except np.linalg.LinAlgError:
                    continue
                det_rotation = np.abs(np.linalg.det(rotation))
                if np.abs(det_rotation - 1) < 1e-3 and np.allclose(np.dot(rotation.T, rotation), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):

                    check_X_T = get_X_T(checkX, translation[translation_idx])
                    v_pred = np.dot(rotation, check_X_T)
                    for lambda_element in each_lambda:
                        dist = distance(v_pred - lambda_element * check_V.T)
                        if dist < min_dist:
                            min_dist = dist
                            rotation_best = rotation
                            translation_best = np.array(translation[translation_idx])

    return translation_best, rotation_best, False

def pnpsolver(query, model, intrinsic_parameters, distortion_parameters):

    points2D, points3D = get_2D3Dcorrespondences(query, model, intrinsic_parameters, distortion_parameters)
    
    rotation_best = []
    translation_best = []
    n_inlier = 0
    
    prob = 0.99
    error_rate = 0.5
    valid_num = 3
    n_ransec_time = int(np.ceil(np.log(1 - prob) / np.log(1 - (1 - error_rate) ** (valid_num + 3))))
    for _ in range(n_ransec_time):
        # sample 3 correspondence pairs
        p2D_sample, p2D_unsample, p3D_sample, p3D_unsample, check_V, check_X = ransac(points2D, points3D)
        v_unsample = np.insert(p2D_unsample, p2D_unsample.shape[1], values = np.ones((1, p2D_unsample.shape[0])), axis = 1)
        translation_best_tmp, rotation_best_tmp, should_continue = solve_p3p(p2D_sample, p3D_sample, check_V, check_X)

        if should_continue or len(translation_best_tmp) == 0:
            continue

        X_T_unsample = get_X_T(p3D_unsample, translation_best_tmp)
        lambda_v = np.dot(rotation_best_tmp, X_T_unsample)
        dist = lambda_v / v_unsample.T
        dist = (dist.max(axis = 0) - dist.min(axis = 0)) / dist.max(axis = 0)
        epsilon = 0.1
        if dist[dist < epsilon].shape[0] > n_inlier:
            n_inlier = dist[dist < epsilon].shape[0]
            rotation_best = rotation_best_tmp
            translation_best = translation_best_tmp
                    
    return rotation_best, translation_best, n_inlier

def qtoa(q):
    if q[3] > 1:
        q = unit_vector(q)
    angle = 2 * np.arccos(q[3])
    s = np.sqrt(1 - q[3] * q[3])
    if s < 0.001:
        x = q[0]
        y = q[1]
        z = q[2]
    else:
        x = q[0] / s
        y = q[1] / s
        z = q[2] / s
    return angle

def calculate_error(tList, tvec_gts, rotqs, rotq_gts):
    translation_error = []
    for idx in range(len(tList)):
        translation_error.append(distance(tvec_gts[idx] - tList[idx]))

    rotation_error = []
    for idx in range(len(rotqs)):
        q = R.from_matrix(R.from_quat(rotqs[idx]).as_matrix() @ np.linalg.inv(R.from_quat(rotq_gts[idx]).as_matrix())).as_quat()
        angle = qtoa(q[0])
        rotation_error.append(angle)

    return np.median(translation_error), np.median(rotation_error)
    
if __name__ == "__main__":

    if (os.path.exists("translation_pred.npy") or os.path.exists("rotation_pred.npy")) == False:
        images_df = pd.read_pickle("data/images.pkl")
        train_df = pd.read_pickle("data/train.pkl")
        points3D_df = pd.read_pickle("data/points3D.pkl")
        point_desc_df = pd.read_pickle("data/point_desc.pkl")

        # Process model descriptors
        desc_df = average_desc(train_df, points3D_df)
        kp_model = np.array(desc_df["XYZ"].to_list())
        desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

        # camera parameters
        intrinsic_parameters = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])    
        distortion_parameters = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352])

        tList = []
        tvecs = []
        rotqs = []
        tvec_gts = []
        rotq_gts = []
        n_image = 293
        images_pbar = trange(1, n_image+1, desc="Image")
        for image_idx in images_pbar:

            # Load query image
            # fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
            # rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

            # Load query keypoints and descriptors
            points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == image_idx]
            kp_query = np.array(points["XY"].to_list())
            desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

            # Get camera pose groudtruth 
            ground_truth = images_df.loc[images_df["IMAGE_ID"] == image_idx]
            rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
            tvec_gt = ground_truth[["TX","TY","TZ"]].values

            tvec_gts.append(tvec_gt)
            rotq_gts.append(rotq_gt)

            # Find correspondance and solve pnp
            rotq, tvec, n_inlier = pnpsolver((kp_query, desc_query), (kp_model, desc_model), intrinsic_parameters, distortion_parameters)

            tList.append(-rotq@tvec)
            tvecs.append(tvec)
            rotqs.append(R.from_matrix(rotq).as_quat())

            images_pbar.set_description(f"Image [{image_idx}/{n_image}]")

        np.save("translation_pred.npy", tvecs)
        np.save("rotation_pred.npy", rotqs)
            
    else:
        images_df = pd.read_pickle("data/images.pkl")
        tvec_gts = []
        rotq_gts = []
        n_image = 293
        for image_idx in range(1, n_image+1):
            # Get camera pose groudtruth 
            ground_truth = images_df.loc[images_df["IMAGE_ID"] == image_idx]
            rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
            tvec_gt = ground_truth[["TX","TY","TZ"]].values

            tvec_gts.append(tvec_gt)
            rotq_gts.append(rotq_gt)

        tvecs = np.load("translation_pred.npy")
        rotqs = np.load("rotation_pred.npy")
        rotas = R.from_quat(rotqs).as_matrix()
        tList = [-(rotas[idx])@(tvecs[idx]) for idx in range(min(len(rotas), len(tvecs)))]
        
    translation_error, rotation_error = calculate_error(tList, tvec_gts, rotqs, rotq_gts)

    print("Q1-2 T =", np.median(translation_error))
    print("Q1-2 angle =", np.median(rotation_error))