import argparse
import cv2
import os
import torch
import numpy as np
import matplotlib
from pathlib import Path
from depth_anything_v2.dpt import DepthAnythingV2


def extract_keyframes_and_depth(video_path, output_dir='./keyframes_with_depth', interval=1, threshold=30):
    """
    Extract keyframes from video and run depth inference on them.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save keyframes and depth maps
        interval: Extract frames every N seconds (default: 1 second)
        threshold: Threshold for scene change detection (0-100, lower = more sensitive)
    """
    
    # Create output directories
    keyframes_dir = os.path.join(output_dir, 'keyframes')
    depth_dir = os.path.join(output_dir, 'depth_maps')
    combined_dir = os.path.join(output_dir, 'combined')
    
    os.makedirs(keyframes_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval)  # Convert seconds to frame count
    
    print(f"Video info: {fps} FPS, {total_frames} total frames")
    print(f"Extracting frames every {interval} second(s) ({frame_interval} frames)")
    
    # Initialize Depth Anything V2 model
    print("Loading Depth Anything V2 model...")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    # Use vits (small model) by default
    depth_anything = DepthAnythingV2(**model_configs['vits'])
    depth_anything.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    frame_count = 0
    keyframe_count = 0
    prev_gray = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract frames at regular intervals
        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Check for scene change (if prev frame exists)
            is_keyframe = True
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                mean_diff = np.mean(diff)
                is_keyframe = mean_diff > threshold
            
            if is_keyframe:
                # Save original keyframe
                timestamp = frame_count / fps
                keyframe_filename = f'keyframe_{keyframe_count:04d}_t{timestamp:.2f}s.jpg'
                keyframe_path = os.path.join(keyframes_dir, keyframe_filename)
                cv2.imwrite(keyframe_path, frame)
                
                print(f"\nKeyframe {keyframe_count}: {keyframe_filename}")
                
                # Run depth inference
                print(f"  Running depth inference...")
                with torch.no_grad():
                    depth = depth_anything.infer_image(frame, 518)
                
                # Normalize depth to 0-255
                depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth_normalized = depth_normalized.astype(np.uint8)
                
                # Save grayscale depth map
                depth_filename = f'depth_{keyframe_count:04d}_t{timestamp:.2f}s.png'
                depth_path = os.path.join(depth_dir, depth_filename)
                cv2.imwrite(depth_path, depth_normalized)
                
                # Save colored depth map
                depth_colored = (cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                depth_colored_path = os.path.join(depth_dir, f'depth_colored_{keyframe_count:04d}_t{timestamp:.2f}s.png')
                cv2.imwrite(depth_colored_path, depth_colored)
                
                # Save combined side-by-side
                split_region = np.ones((frame.shape[0], 50, 3), dtype=np.uint8) * 255
                depth_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
                combined = cv2.hconcat([frame, split_region, depth_bgr])
                combined_path = os.path.join(combined_dir, f'combined_{keyframe_count:04d}_t{timestamp:.2f}s.png')
                cv2.imwrite(combined_path, combined)
                print(f"  Saved: {keyframe_filename} and depth maps")
                
                keyframe_count += 1
                prev_gray = gray
        
        frame_count += 1
    
    cap.release()
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Total keyframes extracted: {keyframe_count}")
    print(f"Output directory: {output_dir}")
    print(f"  - Keyframes: {keyframes_dir}")
    print(f"  - Depth maps (grayscale): {depth_dir}")
    print(f"  - Depth maps (colored): {depth_dir}")
    print(f"  - Combined (original + depth): {combined_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract keyframes and compute depth from video')
    
    parser.add_argument('--video-path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output-dir', type=str, default='./keyframes_with_depth', help='Output directory for keyframes and depth maps')
    parser.add_argument('--interval', type=float, default=1, help='Extract frames every N seconds (default: 1)')
    parser.add_argument('--threshold', type=int, default=30, help='Scene change detection threshold (0-100, lower = more sensitive)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        exit(1)
    
    extract_keyframes_and_depth(
        video_path=args.video_path,
        output_dir=args.output_dir,
        interval=args.interval,
        threshold=args.threshold
    )
