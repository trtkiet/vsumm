import h5py
import cv2
import os
import os.path as osp
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(description="Create a summary video from an original video and an H5 summary file.")
parser.add_argument('-o', '--origin', type=str, required=True, help="path to original video file")
parser.add_argument('-p', '--path', type=str, required=True, help="path to h5 result file containing the summary")
# parser.add_argument('-d', '--frm-dir', type=str, required=True, help="path to frame directory") # Kept commented out
parser.add_argument('-i', '--idx', type=int, default=0, help="index of the video summary in the H5 file (e.g., for 'video_0')")
parser.add_argument('--fps', type=int, default=30, help="frames per second for the output summary video")
parser.add_argument('--width', type=int, default=1280, help="frame width for the output summary video")
parser.add_argument('--height', type=int, default=720, help="frame height for the output summary video")
parser.add_argument('--save-dir', type=str, default='log', help="directory to save the summary video")
parser.add_argument('--save-name', type=str, default='summary.mp4', help="name for the saved summary video (e.g., summary.mp4)")

def create_summary_video(summary_mask, vid_writer, origin_video_capture, target_width, target_height):
    """
    Reads frames from the original video based on the summary_mask and writes them to the vid_writer.

    Args:
        summary_mask (np.array): A binary array (or an array of scores where >0 is considered selected)
                                 indicating which frames to include in the summary.
        vid_writer (cv2.VideoWriter): OpenCV video writer object for the output summary video.
        origin_video_capture (cv2.VideoCapture): OpenCV video capture object for the original video.
        target_width (int): Target width for the frames in the summary video.
        target_height (int): Target height for the frames in the summary video.
    """
    num_original_frames = int(origin_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if len(summary_mask) > num_original_frames:
        print(f"Warning: Summary mask length ({len(summary_mask)}) is greater than original video frames ({num_original_frames}). "
              f"The summary mask will be truncated to the length of the video.")
        summary_mask = summary_mask[:num_original_frames]
    elif len(summary_mask) < num_original_frames:
        print(f"Warning: Summary mask length ({len(summary_mask)}) is shorter than original video frames ({num_original_frames}). "
              f"Only the first {len(summary_mask)} frames of the original video will be considered for the summary.")

    frames_written = 0
    for frame_idx, is_selected in enumerate(summary_mask):
        # Consider selected if the value is 1 (for binary masks) or > 0 (if scores are used)
        if is_selected > 0: 
            origin_video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = origin_video_capture.read()
            if not ret or frame is None:
                print(f"Warning: Could not read frame {frame_idx} from the original video. Skipping.")
                continue
            
            try:
                resized_frame = cv2.resize(frame, (target_width, target_height))
                vid_writer.write(resized_frame)
                frames_written += 1
            except cv2.error as e:
                print(f"Error resizing or writing frame {frame_idx}: {e}. Skipping.")
                continue
                
    print(f"Successfully wrote {frames_written} frames to the summary video.")

if __name__ == '__main__':
    args = parser.parse_args()

    # Validate input paths
    if not osp.isfile(args.origin):
        print(f"Error: Original video file not found at '{args.origin}'")
        sys.exit(1)

    if not osp.isfile(args.path):
        print(f"Error: H5 result file not found at '{args.path}'")
        sys.exit(1)

    # Create save directory if it doesn't exist
    if not osp.exists(args.save_dir):
        try:
            os.makedirs(args.save_dir)
            print(f"Created save directory: {args.save_dir}")
        except OSError as e:
            print(f"Error creating save directory '{args.save_dir}': {e}")
            sys.exit(1)
    
    output_video_path = osp.join(args.save_dir, args.save_name)

    origin_vid_cap = None
    h5_file = None
    summary_vid_writer = None

    try:
        # Open the original video
        origin_vid_cap = cv2.VideoCapture(args.origin)
        if not origin_vid_cap.isOpened():
            print(f"Error: Could not open original video file: '{args.origin}'")
            sys.exit(1)

        # Open the HDF5 file and get the summary
        h5_file = h5py.File(args.path, 'r')
        summary_key_path = f'video_{args.idx}/machine_summary'
        
        if summary_key_path not in h5_file:
            available_keys = []
            def find_keys(name, obj):
                if isinstance(obj, h5py.Dataset) and 'machine_summary' in name:
                    available_keys.append(name.rsplit('/',1)[0]) # Get the video_X part
            h5_file.visititems(find_keys)
            unique_video_keys = sorted(list(set(available_keys)))
            print(f"Error: Summary key '{summary_key_path}' not found in H5 file '{args.path}'.")
            if unique_video_keys:
                print(f"Available video summary groups might be: {unique_video_keys}")
            else:
                print("No 'machine_summary' datasets found in the H5 file.")
            sys.exit(1)
            
        summary_data = h5_file[summary_key_path][...]
        
        # Setup video writer for the summary video
        # Using 'MP4V' for .mp4, alternatives like 'XVID' for .avi could be used.
        fourcc = cv2.VideoWriter_fourcc(*'mp4')
        summary_vid_writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            args.fps,
            (args.width, args.height)
        )
        if not summary_vid_writer.isOpened():
            print(f"Error: Could not open video writer for '{output_video_path}'. Check codec and permissions.")
            sys.exit(1)

        print(f"Processing video: '{args.origin}'")
        print(f"Using summary from H5 file: '{args.path}', key path: '{summary_key_path}'")
        print(f"Outputting summary video to: '{output_video_path}' with FPS: {args.fps}, Resolution: {args.width}x{args.height}")

        # Create the summary video
        create_summary_video(summary_data, summary_vid_writer, origin_vid_cap, args.width, args.height)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if h5_file:
            h5_file.close()
        if origin_vid_cap:
            origin_vid_cap.release()
        if summary_vid_writer:
            summary_vid_writer.release()
        # cv2.destroyAllWindows() # Usually not needed for script-based processing without imshow

    print(f"Video processing finished. Summary video saved to '{output_video_path}'")