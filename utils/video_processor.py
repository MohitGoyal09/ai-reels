import cv2
import os
import glob
from pathlib import Path 
import numpy as np

def extract_frames(video_path, base_output_dir, fps=1, use_keyframes=True, max_frames=30):
    """
    Extracts frames from a single video file with advanced options for better results.

    Args:
        video_path (str): Path to the input video file.
        base_output_dir (str): The main directory where subdirectories for each video's frames will be created.
        fps (int): Number of frames to extract per second.
        use_keyframes (bool): Whether to prioritize keyframes (shots with significant changes) over regular intervals.
        max_frames (int): Maximum number of frames to extract to prevent processing too many similar frames.

    Returns:
        list: A list of tuples, where each tuple contains (frame_file_path, frame_number_in_video).
              Returns an empty list if an error occurs.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print(f"Warning: Video FPS is 0 for {video_path}. Defaulting to 30 for interval calculation.")
        video_fps = 30
        
    frame_interval = int(video_fps / fps)
    if frame_interval == 0:
        frame_interval = 1 

    # Create a unique subdirectory for this video's frames
    video_name_stem = Path(video_path).stem # Gets filename without extension (e.g., "2025-06-02_11-31-19_UTC")
    video_specific_output_dir = os.path.join(base_output_dir, video_name_stem)
    os.makedirs(video_specific_output_dir, exist_ok=True) # Create if it doesn't exist

    extracted_frames_info = []
    current_frame_count = 0
    saved_frame_index = 0
    
    # For scene change detection
    prev_frame = None
    scene_thresholds = []
    frame_diffs = []
    all_frames = []
    
    # First pass - if keyframes are enabled, analyze the video for scene changes
    if use_keyframes:
        print(f"  Analyzing video for keyframes...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Store for later processing
            if current_frame_count % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                all_frames.append((current_frame_count, frame))
                
                if prev_frame is not None:
                    # Calculate difference between current and previous frame
                    diff = cv2.absdiff(gray, prev_frame)
                    # Calculate mean difference as a measure of change
                    mean_diff = np.mean(diff)  # Using numpy's mean function
                    frame_diffs.append((current_frame_count, mean_diff))
                
                prev_frame = gray
            
            current_frame_count += 1
            
        # Reset video capture
        cap.release()
        cap = cv2.VideoCapture(video_path)
        
        # Find the frames with the most significant changes
        if frame_diffs:
            # Sort by difference (largest first)
            frame_diffs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top changes, but limit to max_frames
            keyframes = [f[0] for f in frame_diffs[:max_frames]]
            
            # Add the first frame if it's not already included
            if 0 not in keyframes and all_frames:
                keyframes.append(0)
                
            # Sort keyframes by their position in video
            keyframes.sort()
            
            print(f"  Selected {len(keyframes)} keyframes for processing")
        else:
            # Fallback to regular interval if no frame differences
            keyframes = None
    else:
        keyframes = None
        
    # Reset counters
    current_frame_count = 0
    
    # Second pass - extract the selected frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame if it's a keyframe or we're using regular intervals
        should_extract = False
        
        if keyframes and current_frame_count in keyframes:
            should_extract = True
        elif not keyframes and current_frame_count % frame_interval == 0:
            should_extract = True
            
        if should_extract and saved_frame_index < max_frames:
            # Frame filename within the video-specific directory
            frame_filename = f"frame_{saved_frame_index:04d}.jpg" # Simple incremental name
            frame_filepath = os.path.join(video_specific_output_dir, frame_filename)
            
            try:
                cv2.imwrite(frame_filepath, frame)
                extracted_frames_info.append((frame_filepath, current_frame_count))
                saved_frame_index += 1
            except Exception as e:
                print(f"Error saving frame {frame_filepath}: {e}")
        
        current_frame_count += 1

    cap.release()
    if saved_frame_index > 0:
        print(f"  Extracted {saved_frame_index} frames from '{video_name_stem}' into '{video_specific_output_dir}'")
    return extracted_frames_info