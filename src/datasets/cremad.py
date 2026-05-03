"""
CREMA-D Dataset for audio-visual emotion recognition.

EXACTLY matches OGM-GE preprocessing (Peng et al., CVPR 2022):
https://github.com/GeWu-Lab/OGM-GE_CVPR2022

CREMA-D contains 7,442 video clips of actors expressing 6 emotions:
- NEU (Neutral), HAP (Happy), SAD (Sad), FEA (Fear), DIS (Disgust), ANG (Anger)

Reference:
Cao et al., "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset"
IEEE Transactions on Affective Computing, 2014
"""

import csv
import os
import torch
import numpy as np
import librosa
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict
import torchvision.transforms as transforms


class CREMADDataset(Dataset):
    """
    CREMA-D Dataset matching OGM-GE preprocessing exactly.

    Expected directory structure:
    root/
        AudioWAV/                    # Audio files (.wav)
        Image-01-FPS/               # Video frames at 1 FPS
            {clip_id}/
                frame_0001.jpg
                ...
        train.csv                   # OGM-GE train split
        test.csv                    # OGM-GE test split

    Audio preprocessing (OGM-GE exact):
        - Sample rate: 22050 Hz
        - Duration: 3 seconds (tiled if shorter)
        - Clip amplitude to [-1, 1]
        - STFT: n_fft=512, hop_length=353
        - Log magnitude: log(abs(spectrogram) + 1e-7)
        - Output shape: (1, 257, 187)

    Visual preprocessing (OGM-GE exact):
        - Train: RandomResizedCrop(224), RandomHorizontalFlip
        - Test: Resize(224, 224)
        - ImageNet normalization
    """

    # OGM-GE class mapping (NOT alphabetical!)
    EMOTIONS = ['NEU', 'HAP', 'SAD', 'FEA', 'DIS', 'ANG']
    EMOTION_TO_IDX = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

    # Audio parameters (OGM-GE exact)
    SAMPLE_RATE = 22050
    DURATION_SEC = 3
    N_FFT = 512
    HOP_LENGTH = 353

    def __init__(
        self,
        root: str,
        split: str = "train",
        fps: int = 1,
        num_frames: int = 1,  # OGM-GE default for CREMA-D
    ):
        """
        Args:
            root: Root directory of CREMA-D dataset
            split: 'train' or 'test'
            fps: Frames per second for video (default: 1, matching OGM-GE)
            num_frames: Number of frames to sample (default: 1 for CREMA-D per OGM-GE)
        """
        self.root = root
        self.split = split
        self.fps = fps
        self.num_frames = num_frames

        # Paths
        self.audio_dir = os.path.join(root, "AudioWAV")
        self.visual_dir = os.path.join(root, f"Image-{fps:02d}-FPS")

        # Visual transforms (OGM-GE exact)
        if split == "train":
            self.visual_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.visual_transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Load samples from CSV (OGM-GE exact split)
        self.samples = self._load_samples()

    def _load_samples(self):
        """Load samples from OGM-GE train/test CSV files."""
        samples = []

        csv_file = os.path.join(self.root, f"{self.split}.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(
                f"CSV file not found: {csv_file}\n"
                f"Please copy train.csv and test.csv from OGM-GE repository to {self.root}"
            )

        with open(csv_file, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                if len(item) < 2:
                    continue

                clip_id = item[0]
                emotion = item[1]

                audio_path = os.path.join(self.audio_dir, f"{clip_id}.wav")
                visual_path = os.path.join(self.visual_dir, clip_id)

                # Only include if both audio and visual exist (OGM-GE behavior)
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    label = self.EMOTION_TO_IDX[emotion]
                    samples.append((audio_path, visual_path, label))

        print(f"Loaded {len(samples)} samples for {self.split} split")
        return samples

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio (OGM-GE exact).

        Steps:
        1. Load at 22050 Hz
        2. Tile to 3 seconds
        3. Clip to [-1, 1]
        4. STFT with n_fft=512, hop_length=353
        5. Log magnitude: log(abs + 1e-7)
        """
        try:
            # Load audio at 22050 Hz (OGM-GE exact)
            samples, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE)

            # Tile to 3 seconds (OGM-GE exact)
            target_length = self.SAMPLE_RATE * self.DURATION_SEC
            resamples = np.tile(samples, 3)[:target_length]

            # Clip amplitude to [-1, 1] (OGM-GE exact)
            resamples = np.clip(resamples, -1.0, 1.0)

            # STFT (OGM-GE exact)
            spectrogram = librosa.stft(resamples, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)

            # Log magnitude (OGM-GE exact)
            spectrogram = np.log(np.abs(spectrogram) + 1e-7)

            # Convert to tensor with channel dimension
            # Shape: (1, 257, 187) for n_fft=512, hop_length=353, 3s at 22050Hz
            spec_tensor = torch.FloatTensor(spectrogram).unsqueeze(0)

            return spec_tensor

        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return zeros with expected shape
            freq_bins = self.N_FFT // 2 + 1  # 257
            time_frames = (self.SAMPLE_RATE * self.DURATION_SEC) // self.HOP_LENGTH + 1  # ~187
            return torch.zeros(1, freq_bins, time_frames)

    def _load_visual(self, visual_path: str) -> torch.Tensor:
        """
        Load video frames (OGM-GE exact).

        Steps:
        1. List all frames in directory
        2. Randomly select num_frames (train) or take first num_frames (test)
        3. Apply transforms
        4. Return as (C, T, H, W) matching OGM-GE format
        """
        try:
            if not os.path.isdir(visual_path):
                raise FileNotFoundError(f"Visual path not a directory: {visual_path}")

            frame_files = sorted(os.listdir(visual_path))
            if len(frame_files) == 0:
                raise FileNotFoundError(f"No frames in: {visual_path}")

            # OGM-GE BUG REPRODUCTION: They compute random indices but then use
            # sequential 'i' to index frame_files, so they always load first N frames.
            # To reproduce their results exactly, we match this behavior.
            #
            # OGM-GE code (line 84-90):
            #   select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
            #   select_index.sort()
            #   for i in range(self.args.fps):
            #       img = Image.open(..., image_samples[i])  # BUG: uses 'i' not select_index[i]

            # Load first num_frames frames (matching OGM-GE actual behavior)
            frames = torch.zeros((self.num_frames, 3, 224, 224))
            for i in range(min(self.num_frames, len(frame_files))):
                img_path = os.path.join(visual_path, frame_files[i])
                img = Image.open(img_path).convert('RGB')
                frames[i] = self.visual_transform(img)

            # OGM-GE permutes to (C, T, H, W) for model input
            visual = frames.permute(1, 0, 2, 3)  # Shape: (C, T, H, W) = (3, num_frames, 224, 224)

            return visual

        except Exception as e:
            print(f"Error loading visual {visual_path}: {e}")
            return torch.zeros(3, self.num_frames, 224, 224)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path, visual_path, label = self.samples[idx]

        audio = self._load_audio(audio_path)
        visual = self._load_visual(visual_path)

        return {
            "audio": audio,
            "visual": visual,
            "label": torch.tensor(label, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        return len(self.EMOTIONS)
