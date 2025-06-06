import os
import re
import cv2
import argparse
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import pandas as pd
import glob
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str, default='')
parser.add_argument('--seg_dir', type=str, default='')
parser.add_argument('--bbox_dir', type=str, default='')
parser.add_argument('--new_frames_dir', type=str, default='')
parser.add_argument('--new_audio_dir', type=str, default='')

args = parser.parse_args()

Frame_dir = args.new_frames_dir
Audio_dir = args.new_audio_dir

os.makedirs(Frame_dir, exist_ok=True)
os.makedirs(Audio_dir, exist_ok=True)

# file_list = [file.split('_')[0] for file in os.listdir(args.seg_folder) if file.endswith('.csv')]

seg_filenames = glob.glob(os.path.join(args.seg_dir, '*.csv'))
seg_filenames = [x for x in sorted(seg_filenames)]

box_filenames = glob.glob(os.path.join(args.bbox_dir, '*.csv'))
box_filenames = [x for x in sorted(box_filenames)]


def get_data(seg_file, box_file):
    seg = pd.read_csv(seg_file)
    box = pd.read_csv(box_file)

    start = seg['start_frame'].to_list()
    end = seg['end_frame'].to_list()

    sampled_frames = [list(range(s, e+1)) for s, e in zip(start, end)]

    # start = seg['start_frame'].to_numpy()[:, np.newaxis]
    # end = seg['end_frame'].to_numpy()[:, np.newaxis]
    # sampled_frames = start.copy()
    # iters = 4*cfg.samples
    # for i in range(1, iters):
    #     sampled_frames = np.hstack((sampled_frames, start+(end-start)*(i/iters)))
    # sampled_frames = sampled_frames.astype(int).tolist()
    last_pid = -1
    for i, (pid, frames) in enumerate(zip(seg['person_id'], sampled_frames)):
        if last_pid != pid:
            memory = box[box['person_id'] == pid]
        last_pid = pid
        frames = [frame for frame in frames if memory[(memory['frame_id'] == frame)].values.tolist()[
            0][-1] != -1]
        sampled_frames[i] = frames

    seg['sampled_frames'] = sampled_frames
    seg.drop(columns=['start_frame', 'end_frame'], inplace=True)
    bbox = []
    last_pid = -1
    for pid, frames in zip(seg['person_id'], sampled_frames):
        if last_pid != pid:
            memory = box[box['person_id'] == pid]
        last_pid = pid
        # print(memory)
        # print(memory[(memory['frame_id'] == 1)].values.tolist()[0][-4:])
        bbox.append([tuple(memory[(memory['frame_id'] == frame)].values.tolist()[0][-4:]) for frame in frames])

    seg['bbox'] = bbox
    # seg['hash'] = seg_file.split('/')[-1].split('_')[0]
    if 'ttm' in seg.columns:
        seg.drop(columns=['ttm'], inplace=True)
    # seg.drop(columns=['person_id'], inplace=True)
    # print(seg.columns)
    return seg.values.tolist()  # pid, frames, bboxs


for (seg_file, box_file) in tqdm(zip(seg_filenames, box_filenames), total=len(seg_filenames), desc='Processing'):
    # assert seg_file.split('/')[-1].split('_seg')[0] == box_file.split('/')[-1].split('_bbox')[0]
    videoname = seg_file.split('/')[-1].split('_seg')[0]

    data = get_data(seg_file, box_file)
    data = [x for x in data if len(x[-1]) != 0]

    frames_dict = defaultdict(list)

    for i, d in enumerate(data):
        pid, frames, bboxs = d
        for frame, bbox in zip(frames, bboxs):
            frames_dict[frame].append((pid, bbox))

    cap = cv2.VideoCapture(os.path.join(args.video_dir, f'{videoname}.mp4'))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(num_frames):
        ret, frame = cap.read()
        if frames_dict[i]:
            for pid, bbox in frames_dict[i]:
                if os.path.exists(os.path.join(Frame_dir, f'{videoname}_{i}_{pid}.jpg')):
                    continue
                x1, y1, x2, y2 = bbox
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                avg_x, avg_y = (x1 + x2) / 2., (y1 + y2) / 2.
                x1 = (x1 - avg_x) * 1.3 + avg_x
                x2 = (x2 - avg_x) * 1.3 + avg_x
                y1 = (y1 - avg_y) * 1.4 + avg_y
                y2 = (y2 - avg_y) * 1.4 + avg_y

                y1 = int(max(0, y1))
                x1 = int(max(0, x1))
                y2 = int(min(frame.shape[0], y2))
                x2 = int(min(frame.shape[1], x2))

                top, height = y1, y2 - y1
                left, width = x1, x2 - x1
                crop_img = frame[top:top+height, left:left+width]

                # img = transforms.functional.crop(img, top, left, height, width)
                # img.save(os.path.join(Frame_dir, f'{videoname}_{i}_{pid}.jpg'), "JPEG")

                cv2.imwrite(os.path.join(Frame_dir, f'{videoname}_{i}_{pid}.jpg'), crop_img)

    video = VideoFileClip(os.path.join(args.video_dir, f'{videoname}.mp4'))
    audio = video.audio
    audio.write_audiofile(os.path.join(Audio_dir, f'{videoname}.wav'))
