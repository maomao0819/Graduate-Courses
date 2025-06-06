import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(no_images, basedir, half_res=False, testskip=1):
    split_types = ['train', 'val', 'test']
    print(basedir)
    if './/' in basedir or '/home' in basedir:
        basedir = basedir[3:]

    if os.path.isdir(basedir):
        transforms_jsons = [os.path.basename(file) for file in os.listdir(basedir) if 'json' in file]
        splits = [split_type for split_type in split_types for transforms_json in transforms_jsons if split_type in transforms_json]
        if len(splits) == 0:
            splits = ['test']
        transforms_jsons = [os.path.join(basedir, transforms_json) for transforms_json in transforms_jsons]

    elif os.path.isfile(basedir):
        splits = [split_type for split_type in split_types if split_type in os.path.basename(basedir)]
        if no_images or len(splits) == 0:
            splits = ['test']
        
        transforms_jsons = [basedir]
        basedir = os.path.join(basedir, '..')

    # print('split', splits)
    # print('transforms_jsons', transforms_jsons)
    metas = {}
    # for s in splits:
    #     with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
    #         metas[s] = json.load(fp)
    for s in splits:
        for transforms_json in transforms_jsons:
            if no_images or s in transforms_json:
                with open(transforms_json, 'r') as fp:
                    metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    filenames_dict = {}
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        filenames = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            filename = frame['file_path'] + '.png'
            fname = os.path.join(basedir, filename)
            filenames.append(filename)
            if no_images:
                imgs.append(np.zeros((800, 800, 4)))
            else:
                imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        filenames_dict[s] = filenames

    # i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    i_split = {split_type: None for split_type in split_types}
    for idx in range(len(splits)):
        i_split[splits[idx]] = np.arange(counts[idx], counts[idx+1])

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split, filenames_dict


