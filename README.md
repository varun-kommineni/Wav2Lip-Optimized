# Wav2Lip Optimized

An optimized version of [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) specifically for processing multiple audio files with a single video input.

## Overview

This optimization significantly improves performance when generating multiple lip-sync videos from the same video source with different audio tracks. Instead of repeating the face detection process for each audio file, this optimized version performs face detection only once and reuses those results across all audio files.

## Key Improvements

- **One-time face detection**: Face detection model runs only once regardless of how many audio files you process
- **Modular design**: Introduced `process_audio_file` function to handle individual audio processing
- **Faster batch processing**: Ideal for content creators who need to generate multiple language versions or audio variants

## Prerequisites

All dependencies and pre-installation requirements remain the same as the [original Wav2Lip repository](https://github.com/Rudrabha/Wav2Lip).

## Installation

1. Clone this repository
2. Download the pre-trained model from the original Wav2Lip project
3. Install dependencies as specified in the original repository

## Usage

```bash
python optimised_inference.py \
   --checkpoint_path checkpoints/wav2lip.pth \
   --face /path/to/input/video.mp4 \
   --audio /path/to/audio/folder \
   --output_dir /path/to/output/folder
