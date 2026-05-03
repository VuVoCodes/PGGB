"""
Kinetics-Sounds Dataset for audio-visual action recognition.

Kinetics-Sounds is derived from Kinetics, focusing on 31 action classes
that can be recognized both visually and aurally.

Reference:
Arandjelovic & Zisserman, "Look, Listen and Learn"
ICCV 2017
"""

import os
import torch
import numpy as np
import librosa
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Optional, Callable, List, Tuple
import torchvision.transforms as transforms


class KineticsSoundsDataset(Dataset):
    """
    Kinetics-Sounds Dataset for audio-visual action recognition.

    Supports two modes:
    1. Directory-based: scans train/val directories for class_name/video_id/ folders
    2. Split-file-based: uses OGM-GE-style split files (ogm_train.txt, ogm_test.txt)
       for fair comparison with published baselines

    Expected directory structure:
    root/
        train/
            class_name/
                video_id/
                    frames/
                    audio.wav (or mel_spec.npy)
        val/
            ...
        ogm_train.txt  (optional, for OGM-GE split matching)
        ogm_test.txt   (optional)
    """

    # 31 classes from "Look, Listen and Learn" (Arandjelovic & Zisserman, ICCV 2017)
    CLASSES = [
        "blowing_nose", "blowing_out_candles", "bowling", "chopping_wood",
        "dribbling_basketball", "laughing", "mowing_lawn", "playing_accordion",
        "playing_bagpipes", "playing_bass_guitar", "playing_clarinet",
        "playing_drums", "playing_guitar", "playing_harmonica", "playing_keyboard",
        "playing_organ", "playing_piano", "playing_saxophone", "playing_trombone",
        "playing_trumpet", "playing_violin", "playing_xylophone", "ripping_paper",
        "shoveling_snow", "shuffling_cards", "singing", "stomping_grapes",
        "tap_dancing", "tapping_guitar", "tapping_pen", "tickling"
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        visual_transform: Optional[Callable] = None,
        audio_transform: Optional[Callable] = None,
        sr: int = 16000,
        n_mels: int = 128,
        num_frames: int = 3,
    ):
        self.root = root
        self.split = split
        self.sr = sr
        self.n_mels = n_mels
        self.num_frames = num_frames

        self.class_to_idx = {c: i for i, c in enumerate(self.CLASSES)}

        # Visual transforms matching OGM-GE
        if visual_transform is None:
            if split == "train":
                self.visual_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
            else:
                self.visual_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
        else:
            self.visual_transform = visual_transform

        self.audio_transform = audio_transform

        # Build video path index from all downloaded data
        self._video_index = self._build_video_index()

        # Load samples using OGM-GE split files if available, else directory scan
        self.samples = self._load_samples()

    def _build_video_index(self) -> Dict[Tuple[str, int, int], str]:
        """Build index mapping (youtube_id, start, end) -> video_path."""
        index = {}
        for data_split in ['train', 'val']:
            split_dir = os.path.join(self.root, data_split)
            if not os.path.exists(split_dir):
                continue
            for class_name in os.listdir(split_dir):
                class_path = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_path):
                    continue
                for video_id in os.listdir(class_path):
                    video_path = os.path.join(class_path, video_id)
                    if not os.path.isdir(video_path):
                        continue
                    # Parse video_id: {youtube_id}_{start}_{end}
                    parts = video_id.rsplit('_', 2)
                    if len(parts) == 3:
                        try:
                            key = (parts[0], int(parts[1]), int(parts[2]))
                            index[key] = video_path
                        except ValueError:
                            pass
        return index

    def _load_samples_from_split_file(self, split_file: str) -> List:
        """Load samples from OGM-GE-style split file."""
        samples = []
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                vid = parts[0]
                label = int(parts[2])

                # Parse: youtube_id_startframe_endframe
                segments = vid.rsplit('_', 2)
                if len(segments) == 3:
                    try:
                        yt_id = segments[0]
                        start = int(segments[1])
                        end = int(segments[2])
                        key = (yt_id, start, end)
                        if key in self._video_index:
                            samples.append((self._video_index[key], label))
                    except ValueError:
                        pass
        return samples

    def _load_samples(self) -> List:
        """Load samples using OGM-GE splits if available, else directory scan."""
        # Try OGM-GE split files first
        if self.split == "train":
            split_file = os.path.join(self.root, "ogm_train.txt")
        else:
            split_file = os.path.join(self.root, "ogm_test.txt")

        if os.path.exists(split_file):
            samples = self._load_samples_from_split_file(split_file)
            if samples:
                return samples

        # Fallback: directory scan
        samples = []
        split_dir = os.path.join(self.root, self.split)
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory not found: {split_dir}")
            return samples

        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            if class_name not in self.class_to_idx:
                continue
            label = self.class_to_idx[class_name]
            for video_id in os.listdir(class_path):
                video_path = os.path.join(class_path, video_id)
                if os.path.isdir(video_path):
                    samples.append((video_path, label))

        return samples

    def _load_audio(self, video_path: str) -> torch.Tensor:
        """Load audio as mel spectrogram. Uses pre-extracted .npy if available."""
        try:
            # Try pre-extracted spectrogram first (much faster)
            npy_path = os.path.join(video_path, "mel_spec.npy")
            if os.path.exists(npy_path):
                log_mel_spec = np.load(npy_path)
                spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0)
                spec_tensor = torch.nn.functional.interpolate(
                    spec_tensor.unsqueeze(0), size=(128, 128), mode='bilinear'
                ).squeeze(0)
                return spec_tensor

            # Fallback: compute from wav
            audio_path = None
            for ext in ['.wav', '.mp3', '.flac']:
                candidate = os.path.join(video_path, f"audio{ext}")
                if os.path.exists(candidate):
                    audio_path = candidate
                    break

            if audio_path is None:
                return torch.zeros(1, 128, 128)

            waveform, sr = librosa.load(audio_path, sr=self.sr)
            mel_spec = librosa.feature.melspectrogram(
                y=waveform, sr=sr, n_mels=self.n_mels
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)

            spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0)
            spec_tensor = torch.nn.functional.interpolate(
                spec_tensor.unsqueeze(0), size=(128, 128), mode='bilinear'
            ).squeeze(0)
            return spec_tensor

        except Exception as e:
            print(f"Error loading audio from {video_path}: {e}")
            return torch.zeros(1, 128, 128)

    def _load_visual(self, video_path: str) -> torch.Tensor:
        """Load and stack video frames (OGM-GE style: evenly spaced)."""
        try:
            frame_dir = os.path.join(video_path, "frames")
            if not os.path.exists(frame_dir):
                frame_dir = video_path

            frames = sorted([
                f for f in os.listdir(frame_dir)
                if f.endswith(('.jpg', '.png', '.jpeg'))
            ])

            if not frames:
                return torch.zeros(3, self.num_frames, 224, 224)

            # OGM-GE style: evenly spaced frames
            # seg = file_num / pick_num, indices at [seg*0+1, seg*1+1, seg*2+1] (0-indexed: [0, seg, 2*seg])
            file_num = len(frames)
            seg = file_num / self.num_frames
            indices = [min(int(seg * i), file_num - 1) for i in range(self.num_frames)]

            frame_tensors = []
            for idx in indices:
                frame_path = os.path.join(frame_dir, frames[idx])
                image = Image.open(frame_path).convert('RGB')
                frame_tensors.append(self.visual_transform(image))

            # Stack as (C, T, H, W) matching OGM-GE temporal format
            stacked = torch.stack(frame_tensors, dim=0)  # (T, C, H, W)
            return stacked.permute(1, 0, 2, 3)  # (C, T, H, W)

        except Exception as e:
            print(f"Error loading visual from {video_path}: {e}")
            return torch.zeros(3, self.num_frames, 224, 224)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_path, label = self.samples[idx]

        audio = self._load_audio(video_path)
        visual = self._load_visual(video_path)

        return {
            "audio": audio,
            "visual": visual,
            "label": torch.tensor(label, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        return len(self.CLASSES)
