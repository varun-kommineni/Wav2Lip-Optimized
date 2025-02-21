from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
import time

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models with multiple audio files')

# [Previous argument definitions remain the same]
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models with multiple audio files')

parser.add_argument('--checkpoint_path', type=str, 
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
                    help='Filepath of video/image that contains faces to use', required=True)

parser.add_argument('--audio_dir', type=str, 
                    help='Directory containing audio files to process', required=True)

parser.add_argument('--output_dir', type=str, help='Prefix for output video paths', 
                    default='results/result_')

parser.add_argument('--static', type=bool, 
                    help='If True, then use only first video frame for inference', default=False)

parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
                    help='Batch size for face detection', default=16)

parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, 
            help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
    
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    
    start_time = time.time()
    
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size
    
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size), desc="Face detection"):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    
    face_detect_time = time.time() - start_time
    print(f'Face detection completed in {face_detect_time:.2f} seconds')
    
    return results



def datagen(frames, mels, face_det_results):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    for i, m in enumerate(mels):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def process_audio_file(audio_path, full_frames, fps, face_det_results, model):
    output_path = args.output_dir + os.path.basename(audio_path).rsplit('.', 1)[0] + '.mp4'
    
    # Time audio preprocessing
    audio_prep_start = time.time()
    if not audio_path.endswith('.wav'):
        print(f'Extracting raw audio from {audio_path}...')
        temp_wav = f'temp/temp_{os.path.basename(audio_path)}.wav'
        command = f'ffmpeg -y -i "{audio_path}" -strict -2 "{temp_wav}"'
        subprocess.call(command, shell=True)
        audio_path = temp_wav

    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    audio_prep_time = time.time() - audio_prep_start
    print(f'Audio preprocessing completed in {audio_prep_time:.2f} seconds')
    
    if np.isnan(mel.reshape(-1)).sum() > 0:
        print(f'Warning: Mel contains NaN for {audio_path}. Skipping.')
        return

    # Time mel chunks preparation
    mel_prep_start = time.time()
    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    mel_prep_time = time.time() - mel_prep_start
    print(f'Mel chunks preparation completed in {mel_prep_time:.2f} seconds')

    print(f"Length of mel chunks for {os.path.basename(audio_path)}: {len(mel_chunks)}")
    
    num_frames_needed = len(mel_chunks)
    num_frames_available = len(full_frames)
    num_face_results = len(face_det_results)
    
    print(f"Audio requires {num_frames_needed} frames, video has {num_frames_available} frames, face detection has {num_face_results} results")
    
    # Time frame preparation
    frame_prep_start = time.time()
    if num_face_results < num_frames_needed:
        print(f"Extending face detection results by cycling through available {num_face_results} results")
        extended_face_det_results = []
        for i in range(num_frames_needed):
            extended_face_det_results.append(face_det_results[i % num_face_results])
    else:
        extended_face_det_results = face_det_results

    if num_frames_available < num_frames_needed:
        print(f"Looping video frames to match audio length")
        extended_frames = []
        for i in range(num_frames_needed):
            extended_frames.append(full_frames[i % num_frames_available])
    else:
        extended_frames = full_frames[:num_frames_needed]
    frame_prep_time = time.time() - frame_prep_start
    print(f'Frame preparation completed in {frame_prep_time:.2f} seconds')

    frame_h, frame_w = full_frames[0].shape[:-1]
    temp_video_path = f'temp/temp_video_{os.path.basename(audio_path)}.avi'
    out = cv2.VideoWriter(temp_video_path, 
                         cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    # Time lip generation
    lip_gen_start = time.time()
    batch_size = args.wav2lip_batch_size
    gen = datagen(extended_frames.copy(), mel_chunks, extended_face_det_results)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                            total=int(np.ceil(float(len(mel_chunks))/batch_size)),
                                            desc=f"Processing {os.path.basename(audio_path)}")):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()
    lip_gen_time = time.time() - lip_gen_start
    print(f'Lip generation completed in {lip_gen_time:.2f} seconds')

    # Time final video creation
    video_creation_start = time.time()
    command = f'ffmpeg -y -i "{audio_path}" -i "{temp_video_path}" -strict -2 -q:v 1 "{output_path}"'
    subprocess.call(command, shell=platform.system() != 'Windows')
    video_creation_time = time.time() - video_creation_start
    print(f'Final video creation completed in {video_creation_time:.2f} seconds')
    
    try:
        os.remove(temp_video_path)
    except:
        pass
    


def main():
    start_time = time.time()
    
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    # Time video frame reading
    if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)
    
    print(f"Number of frames available for inference: {len(full_frames)}")

    # Time face detection
    face_detection_start = time.time()
    print("Running face detection...")
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(full_frames)
        else:
            face_det_results = face_detect([full_frames[0]])
            face_det_results = face_det_results * len(full_frames)
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames]
    face_detection_time = time.time() - face_detection_start
    print(f'Face detection phase completed in {face_detection_time:.2f} seconds')

    # Time model loading
    model_loading_start = time.time()
    model = load_model(args.checkpoint_path)
    model_loading_time = time.time() - model_loading_start
    print(f'Model loading completed in {model_loading_time:.2f} seconds')
    print("Model loaded")

    os.makedirs('temp', exist_ok=True)
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    if os.path.isdir(args.audio_dir):
        audio_files = []
        for ext in ['wav', 'mp3', 'm4a', 'aac', 'flac']:
            audio_files.extend(glob(os.path.join(args.audio_dir, f'*.{ext}')))
        
        if not audio_files:
            raise ValueError(f"No audio files found in {args.audio_dir}")
        
        print(f"Found {len(audio_files)} audio files to process")
        video_generation_start = time.time()
        for audio_file in audio_files:
            each_audio_start = time.time() 
            print(f"\nProcessing {os.path.basename(audio_file)}...")
            process_audio_file(audio_file, full_frames, fps, face_det_results, model)
            each_audio_end = time.time()
            print(f"for audio {os.path.basename(audio_file)} it has taken {each_audio_end - each_audio_start}")
        
        video_generation_end = time.time()
            
    else:
        raise ValueError('--audio_dir must be a valid directory containing audio files')
    
    total_time = time.time() - start_time
    print(f"\nTotal execution summary:")
    print(f"Face detection time: {face_detection_time:.2f}s")
    print(f"Model loading time: {model_loading_time:.2f}s")
    print(f"Total execution time: {total_time:.2f}s")
    print("video generation time: ",video_generation_end - video_generation_start)

if __name__ == '__main__':
    main()

