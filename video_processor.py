import os
import torch
import torchvision
from torchvision import transforms
from torchvision.io import read_video
from PIL import Image

# Paths
videos_path = './data/'  # Replace with your actual path
video_files = [os.path.join(videos_path, f) for f in os.listdir(videos_path) if f.endswith('.mp4')]

dataset_root = 'datasets/colorization_dataset/'
train_A_dir = os.path.join(dataset_root, 'train', 'A')
train_B_dir = os.path.join(dataset_root, 'train', 'B')
os.makedirs(train_A_dir, exist_ok=True)
os.makedirs(train_B_dir, exist_ok=True)

def extract_and_process_frames(video_path, frame_rate=1, height=480, start_frame=0):
    video_frames, _, info = read_video(video_path, start_pts=0, end_pts=None, pts_unit='sec')
    fps = info['video_fps']
    total_frames = video_frames.shape[0]

    frame_interval = max(int(fps / frame_rate), 1)

    for i in range(start_frame, total_frames, frame_interval):
        frame = video_frames[i]
        frame_pil = transforms.ToPILImage()(frame)
        orig_width, orig_height = frame_pil.size
        aspect_ratio = orig_width / orig_height
        new_width = int(height * aspect_ratio)
        frame_resized = frame_pil.resize((new_width, height), Image.BICUBIC)
        frame_gray = frame_resized.convert('L')

        frame_number = f"{i:06d}"
        frame_basename = os.path.splitext(os.path.basename(video_path))[0]
        color_frame_path = os.path.join(train_B_dir, f"{frame_basename}_{frame_number}.png")
        gray_frame_path = os.path.join(train_A_dir, f"{frame_basename}_{frame_number}.png")

        frame_resized.save(color_frame_path)
        frame_gray.save(gray_frame_path)

for video_file in video_files:
    print(f"Processing {video_file}...")
    extract_and_process_frames(video_file, frame_rate=1, height=480)
