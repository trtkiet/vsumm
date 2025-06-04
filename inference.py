from __future__ import print_function
import os
import os.path as osp
import argparse
import h5py
import time
import datetime
import numpy as np
import cv2 # Add OpenCV import

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils import Logger, read_json, write_json, save_checkpoint
from models import *
from rewards import compute_reward
from generate_dataset import Generate_Dataset
import vsum_tools

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options
parser.add_argument('-s', '--source', type=str, default='', help="path to your custom video file") 
parser.add_argument('-d', '--dataset', type=str, default='', help="path to h5 dataset(optional)")

# Model options
parser.add_argument('--input-dim', type=int, default=1024, help="input dimension (default: 1024)")
parser.add_argument('--hidden-dim', type=int, default=256, help="hidden unit dimension of DSN (default: 256)")
parser.add_argument('--num-layers', type=int, default=1, help="number of RNN layers (default: 1)")
parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")
parser.add_argument('--weights', type=str, default='log/summe-split0/model_epoch60.pth.tar', help="pretrained DSN model parameters")

# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
parser.add_argument('--evaluate', action='store_true', help="whether to do evaluation only")
parser.add_argument('--save-dir', type=str, default='log', help="path to save output (default: 'log/')")
# parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--verbose', action='store_true', help="whether to show detailed test results")
# parser.add_argument('--save-results', action='store_true', help="whether to save output results")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False

@torch.no_grad()
def main():
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")
    
    # Ensure save_dir exists
    if args.save_dir and not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    start_time = time.time()
    if args.source != '' and args.dataset != None:
        print(f"Making dataset from your video {args.source}")
        gen_data = Generate_Dataset(args.source, args.dataset,)
        gen_data.generate()
        gen_data.h5_file.close()
        print(f"dataset Done. {args.dataset}")

    print("Initialize dataset {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'a')
    num_videos = len(dataset.keys())
    print(f"# total videos : {num_videos}.")

    print(f"Initialize model")
    model = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    
    print(f"Load pretrained model : {args.weights}")
    model.load_state_dict(torch.load(args.weights))
    print(f"Load pretrained model Done.")
    
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    # start_time = time.time()
    model.eval()
    
    for id in range(num_videos):
        key = f'video_{id}'
        seq = dataset[key]['features'][...] # sequence of features, (seq_len, dim)
        seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)
        if use_gpu: seq = seq.cuda()
        probs = model(seq) # output shape (1, seq_len, 1)
        probs = probs.data.cpu().squeeze().numpy()
        cps = dataset[key]['change_points'][...]
        num_frames = dataset[key]['n_frames'][()]
        nfps = dataset[key]['n_frame_per_seg'][...].tolist()
        positions = dataset[key]['picks'][...]
        
        machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions, 0.15)
        # Check if the dataset already exists before creating it
        if key + '/score' in dataset:
            del dataset[key + '/score']
        dataset.create_dataset(key + '/score', data=probs)
        
        if key + '/machine_summary' in dataset:
            del dataset[key + '/machine_summary']
        dataset.create_dataset(key + '/machine_summary', data=machine_summary)

        if args.source and osp.isfile(args.source):
            source_video_filename_no_ext = osp.splitext(osp.basename(args.source))[0]
            output_video_name = f"{source_video_filename_no_ext}_{key}_summary.mp4"
            output_video_path = osp.join(args.save_dir, output_video_name)
            print(f"Attempting to create summary video for {key} at {output_video_path} from {args.source}")
            save_summary_video(args.source, machine_summary, output_video_path, num_frames)
        elif args.source:
            print(f"Warning: Source video path '{args.source}' provided but file does not exist. Skipping summary video generation for {key}.")
        else:
            print(f"Warning: `args.source` not provided. Skipping summary video generation for {key}.")

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    dataset.close()

def save_summary_video(original_video_path, machine_summary, output_video_path, total_frames_original):
    """
    Creates a video from selected frames based on machine_summary.

    Args:
    - original_video_path (str): Path to the source video.
    - machine_summary (np.ndarray): Binary array indicating selected frames.
    - output_video_path (str): Path to save the generated summary video.
    - total_frames_original (int): Total number of frames in the original video.
    """
    selected_frames_indices = np.where(machine_summary == 1)[0]

    if len(selected_frames_indices) == 0:
        print(f"No frames selected for summary from {original_video_path}. No video will be created at {output_video_path}.")
        return

    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {original_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Fallback if fps is 0 or not readable, common in some video files or cv2 versions
    if fps == 0:
        print(f"Warning: FPS read as 0 for {original_video_path}. Defaulting to 25 FPS for output.")
        fps = 25 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        print(f"Error: Could not get valid frame dimensions from {original_video_path}. Width: {width}, Height: {height}")
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Common codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for {output_video_path}. Check codec and permissions.")
        cap.release()
        return
        
    print(f"Writing summary video to {output_video_path} with FPS: {fps}, Resolution: {width}x{height}")

    current_frame_index_in_video = 0
    selected_frame_pointer = 0 # Pointer to the current index in selected_frames_indices

    while cap.isOpened() and selected_frame_pointer < len(selected_frames_indices):
        ret, frame = cap.read()
        if not ret:
            # End of video reached
            if selected_frame_pointer < len(selected_frames_indices):
                print(f"Warning: Reached end of video at frame {current_frame_index_in_video} but more selected frames were expected (e.g., {selected_frames_indices[selected_frame_pointer]}).")
            break

        # Check if the current frame from video is one of the selected frames
        if current_frame_index_in_video == selected_frames_indices[selected_frame_pointer]:
            out.write(frame)
            selected_frame_pointer += 1
        
        current_frame_index_in_video += 1

        # Optimization: if all selected frames are processed, no need to read further
        if selected_frame_pointer == len(selected_frames_indices):
            break
        
        # Safety break: if video frame index somehow goes way beyond what's expected
        # (This might happen if machine_summary length doesn't match video)
        # machine_summary should have length total_frames_original
        if current_frame_index_in_video > total_frames_original + 100: # Adding a small buffer
             print(f"Warning: Video frame index {current_frame_index_in_video} significantly exceeds total frames {total_frames_original}. Stopping video generation.")
             break


    if selected_frame_pointer < len(selected_frames_indices):
        print(f"Warning: Not all selected frames were written. Processed {selected_frame_pointer}/{len(selected_frames_indices)} selected frames.")
        print(f"Last processed original frame index: {current_frame_index_in_video-1}. Next expected selected frame index: {selected_frames_indices[selected_frame_pointer] if selected_frame_pointer < len(selected_frames_indices) else 'N/A'}")


    cap.release()
    out.release()
    if selected_frame_pointer > 0 : # Only print success if some frames were written
        print(f"Summary video saved to {output_video_path} ({selected_frame_pointer} frames written).")
    elif len(selected_frames_indices) > 0 : # Selected frames existed but none were written
        print(f"Failed to write any frames to {output_video_path} despite {len(selected_frames_indices)} selected frames.")


if __name__ == '__main__':
    main()
