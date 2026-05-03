"""
AVE Dataset for audio-visual event classification.

AVE contains ~4,143 10-second videos across 28 event classes.
Uses OGM-GE-compatible format: pre-computed pickle spectrograms + extracted frames.

Reference:
Tian et al., "Audio-Visual Event Localization in Unconstrained Videos"
ECCV 2018

Expected directory structure (created by scripts/prepare_ave.py):
    root/
        visual/{video_id}/frame_00001.jpg, ...
        audio_spec/{video_id}.pkl
        stat.txt                    # sorted class names
        my_train.txt                # class_index,video_id
        my_test.txt                 # class_index,video_id
"""

import copy
import csv
import os
import pickle
from typing import Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AVEDataset(Dataset):
    """AVE Dataset matching OGM-GE's data loading format.

    Parameters
    ----------
    root : str
        Root directory of the AVE dataset.
    split : str
        'train' or 'test'.
    num_frames : int
        Number of frames to sample per video.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 3,
    ):
        self.root = root
        self.split = split
        self.num_frames = num_frames

        self.visual_path = os.path.join(root, "visual")
        self.audio_spec_path = os.path.join(root, "audio_spec")
        self.stat_path = os.path.join(root, "stat.txt")
        self.train_txt = os.path.join(root, "my_train.txt")
        self.test_txt = os.path.join(root, "my_test.txt")

        # Load class names
        self.classes = []
        with open(self.stat_path) as f:
            reader = csv.reader(f)
            for row in reader:
                self.classes.append(row[0])
        self.classes = sorted(self.classes)

        # Load samples
        csv_file = self.train_txt if split == "train" else self.test_txt
        self.samples = []
        self.data2class = {}

        with open(csv_file) as f:
            reader = csv.reader(f)
            for item in reader:
                if len(item) < 2:
                    continue
                class_idx = int(item[0])
                video_id = item[1]
                audio_path = os.path.join(self.audio_spec_path, video_id + ".pkl")
                visual_dir = os.path.join(self.visual_path, video_id)
                if os.path.exists(audio_path) and os.path.exists(visual_dir):
                    self.samples.append(video_id)
                    self.data2class[video_id] = class_idx

        print(f"AVE [{split}]: {len(self.samples)} samples, {len(self.classes)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id = self.samples[idx]
        label = self.data2class[video_id]

        # Load audio spectrogram from pickle
        audio_path = os.path.join(self.audio_spec_path, video_id + ".pkl")
        spectrogram = pickle.load(open(audio_path, "rb"))
        if not isinstance(spectrogram, torch.Tensor):
            spectrogram = torch.FloatTensor(spectrogram)
        # Add channel dimension for ResNet18: (H, W) -> (1, H, W)
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)

        # Load visual frames (matching OGM-GE's frame sampling)
        visual_dir = os.path.join(self.visual_path, video_id)
        file_num = len([f for f in os.listdir(visual_dir) if f.endswith(".jpg")])

        if self.split == "train":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        pick_num = self.num_frames
        seg = max(1, int(file_num / pick_num))

        image_n = None
        for i in range(pick_num):
            t = seg * i + 1
            frame_name = f"frame_{t:05d}.jpg"
            frame_path = os.path.join(visual_dir, frame_name)

            if os.path.exists(frame_path):
                image = Image.open(frame_path).convert("RGB")
            else:
                # Fallback: use first available frame
                frames = sorted(f for f in os.listdir(visual_dir) if f.endswith(".jpg"))
                if frames:
                    image = Image.open(os.path.join(visual_dir, frames[min(i, len(frames) - 1)])).convert("RGB")
                else:
                    # Return zero tensor if no frames
                    return {
                        "audio": spectrogram,
                        "visual": torch.zeros(3, pick_num, 224, 224),
                        "label": torch.tensor(label, dtype=torch.long),
                    }

            img_tensor = transform(image).unsqueeze(1).float()  # (3, 1, 224, 224)
            if image_n is None:
                image_n = copy.copy(img_tensor)
            else:
                image_n = torch.cat((image_n, img_tensor), 1)  # (3, T, 224, 224)

        return {
            "audio": spectrogram,
            "visual": image_n,
            "label": torch.tensor(label, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        return len(self.classes)
