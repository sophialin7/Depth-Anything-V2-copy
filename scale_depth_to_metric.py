"""
Scale relative depth maps to metric depth using either:
1. Reference Object Method: Place a known-size object (ruler, paper) in the video
2. Camera Intrinsics Method: Use focal length and sensor size

For a typical iPhone video:
- Focal length: ~28-35mm equivalent
- Sensor width: ~4-5mm
- Resolution: 1920x1080 or similar
"""

import cv2
import numpy as np
import os
from pathlib import Path


class DepthScaler:
    def __init__(self, depth_dir, output_dir='./depth_scaled'):
        """
        Args:
            depth_dir: Directory containing grayscale depth maps
            output_dir: Directory to save scaled depth maps
        """
        self.depth_dir = depth_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def method_a_reference_object(self, reference_pixel_count, reference_real_size_cm):
        """
        OPTION A: Reference Object Method
        
        Steps:
        1. Place a reference object (ruler, paper, known-size object) on the wall
        2. In the depth map, count pixels occupied by the reference object
        3. Know the real-world size of the object (e.g., 11 inches = 27.94 cm)
        4. Calculate scaling factor
        
        Args:
            reference_pixel_count: Number of pixels the reference object occupies in depth map
            reference_real_size_cm: Real-world size of reference object in centimeters
            
        Returns:
            scaling_factor: cm per depth unit
        """
        # The depth map has values 0-255 (after normalization)
        # We need to find how many depth units the reference object spans
        
        scaling_factor = reference_real_size_cm / reference_pixel_count
        print(f"\n{'='*60}")
        print(f"OPTION A: Reference Object Calibration")
        print(f"{'='*60}")
        print(f"Reference object real size: {reference_real_size_cm} cm")
        print(f"Reference object pixel width: {reference_pixel_count} pixels")
        print(f"Scaling factor: {scaling_factor:.4f} cm/pixel")
        print(f"{'='*60}\n")
        
        return scaling_factor
    
    def method_b_camera_intrinsics(self, focal_length_mm, sensor_width_mm, image_width_px):
        """
        OPTION B: Camera Intrinsics Method
        
        Uses the pinhole camera model:
        depth_in_world = (real_world_size * focal_length) / image_width
        
        For iPhone (typical values):
        - Focal length: 28-35mm (equivalent)
        - Sensor width: 4-5mm
        - Typical video resolution: 1920x1080
        
        Args:
            focal_length_mm: Focal length in millimeters (e.g., 28)
            sensor_width_mm: Sensor width in millimeters (e.g., 4.5)
            image_width_px: Image width in pixels (e.g., 1920)
            
        Returns:
            pixel_to_mm: Conversion factor from pixel width to mm at distance
        """
        # Field of view calculation
        fov_rad = 2 * np.arctan(sensor_width_mm / (2 * focal_length_mm))
        fov_deg = np.degrees(fov_rad)
        
        # At 1 meter distance, the mm per pixel
        pixels_per_mm_at_1m = image_width_px / (2 * 1000 * np.tan(fov_rad / 2))
        
        print(f"\n{'='*60}")
        print(f"OPTION B: Camera Intrinsics Calibration")
        print(f"{'='*60}")
        print(f"Focal length: {focal_length_mm}mm")
        print(f"Sensor width: {sensor_width_mm}mm")
        print(f"Image width: {image_width_px}px")
        print(f"Field of view: {fov_deg:.2f}°")
        print(f"Pixels per mm at 1m: {pixels_per_mm_at_1m:.4f}")
        print(f"{'='*60}\n")
        
        return pixels_per_mm_at_1m
    
    def apply_scaling_to_directory(self, scaling_factor, depth_filenames=None):
        """
        Apply scaling factor to all depth maps in directory
        
        Args:
            scaling_factor: Scaling factor (cm/pixel or derived from method)
            depth_filenames: List of specific files to process (None = all .png files)
        """
        # Get list of grayscale depth maps (exclude colored ones)
        if depth_filenames is None:
            all_files = sorted([f for f in os.listdir(self.depth_dir) if f.endswith('.png')])
            depth_filenames = [f for f in all_files if 'colored' not in f]
        
        print(f"Scaling {len(depth_filenames)} depth maps...")
        print(f"Scaling factor: {scaling_factor}\n")
        
        for filename in depth_filenames:
            filepath = os.path.join(self.depth_dir, filename)
            depth_map = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            if depth_map is None:
                print(f"Warning: Could not read {filename}")
                continue
            
            # Convert to metric (assuming depth_map is 0-255 normalized)
            depth_metric = depth_map.astype(np.float32) * scaling_factor
            
            # Save as float32 for precision
            output_path = os.path.join(self.output_dir, f'metric_{filename}')
            # Save as 16-bit PNG (can store wider range than 8-bit)
            depth_metric_uint16 = np.clip(depth_metric * 100, 0, 65535).astype(np.uint16)
            cv2.imwrite(output_path, depth_metric_uint16)
            
            print(f"✓ {filename}")
            print(f"  Range: {depth_metric.min():.2f} - {depth_metric.max():.2f} cm")
        
        print(f"\nScaled depth maps saved to: {self.output_dir}")
    
    def interactive_reference_object_picker(self, keyframe_path, depth_map_path):
        """
        Interactive tool to select reference object region in image
        and measure its depth
        
        Args:
            keyframe_path: Path to original keyframe image
            depth_map_path: Path to corresponding depth map
        """
        print(f"\n{'='*60}")
        print(f"INTERACTIVE REFERENCE OBJECT PICKER")
        print(f"{'='*60}")
        print(f"Keyframe: {keyframe_path}")
        print(f"Depth map: {depth_map_path}\n")
        
        img = cv2.imread(keyframe_path)
        depth = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or depth is None:
            print("Error: Could not read images")
            return
        
        print("Instructions:")
        print("1. Select the reference object in the image by clicking corners")
        print("2. Left-click to add points, right-click to undo last point")
        print("3. Press ENTER when done (need 2+ points)")
        print("4. The depth values in that region will be analyzed\n")
        
        roi_points = []
        original_img = img.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal roi_points, img
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_points.append((x, y))
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Select Reference Object', img)
            elif event == cv2.EVENT_RBUTTONDOWN and roi_points:
                roi_points.pop()
                img = original_img.copy()
                for pt in roi_points:
                    cv2.circle(img, pt, 5, (0, 255, 0), -1)
                cv2.imshow('Select Reference Object', img)
        
        cv2.namedWindow('Select Reference Object')
        cv2.setMouseCallback('Select Reference Object', mouse_callback)
        cv2.imshow('Select Reference Object', img)
        
        print("Click on the image to select reference object corners...")
        print("Press ENTER when done, or 'q' to cancel\n")
        
        while True:
            key = cv2.waitKey(0)
            if key == 13:  # ENTER
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return
        
        cv2.destroyAllWindows()
        
        if len(roi_points) < 2:
            print("Error: Need at least 2 points")
            return
        
        # Create mask for ROI
        mask = np.zeros(depth.shape, dtype=np.uint8)
        roi_points = np.array(roi_points, dtype=np.int32)
        cv2.fillPoly(mask, [roi_points], 255)
        
        # Get depth values in ROI
        roi_depth = depth[mask == 255]
        
        if len(roi_depth) == 0:
            print("Error: No depth values in selected region")
            return
        
        print(f"Analysis of selected reference object region:")
        print(f"  Pixels in region: {len(roi_depth)}")
        print(f"  Depth min: {roi_depth.min()}")
        print(f"  Depth max: {roi_depth.max()}")
        print(f"  Depth mean: {roi_depth.mean():.2f}")
        print(f"  Depth std: {roi_depth.std():.2f}\n")
        
        # Calculate bounding box width
        x_coords = roi_points[:, 0]
        y_coords = roi_points[:, 1]
        bbox_width = max(x_coords) - min(x_coords)
        bbox_height = max(y_coords) - min(y_coords)
        bbox_diagonal = np.sqrt(bbox_width**2 + bbox_height**2)
        
        print(f"Bounding box:")
        print(f"  Width: {bbox_width} pixels")
        print(f"  Height: {bbox_height} pixels")
        print(f"  Diagonal: {bbox_diagonal:.1f} pixels\n")
        
        return roi_points, roi_depth


def example_usage():
    """
    Example usage with typical values
    """
    print("\n" + "="*60)
    print("EXAMPLE: Scaling Depth Maps to Metric Units")
    print("="*60)
    
    scaler = DepthScaler(
        depth_dir='./keyframes_with_depth/depth_maps',
        output_dir='./keyframes_with_depth/depth_scaled'
    )
    
    # EXAMPLE 1: Using a reference object (e.g., A4 paper = 21cm wide)
    print("\n--- EXAMPLE 1: Reference Object Method ---")
    print("Scenario: You placed an A4 paper (21cm wide) on the wall")
    print("The paper appears 50 pixels wide in the depth map")
    
    scaling_factor_a = scaler.method_a_reference_object(
        reference_pixel_count=50,
        reference_real_size_cm=21
    )
    
    # EXAMPLE 2: Using camera intrinsics (typical iPhone)
    print("\n--- EXAMPLE 2: Camera Intrinsics Method ---")
    print("Scenario: iPhone camera with ~28mm equivalent focal length")
    
    scaling_factor_b = scaler.method_b_camera_intrinsics(
        focal_length_mm=28,
        sensor_width_mm=4.5,
        image_width_px=1920
    )
    
    # Apply scaling (uncomment one of these)
    print("\n--- Applying Scaling ---")
    # scaler.apply_scaling_to_directory(scaling_factor_a)
    # OR
    # scaler.apply_scaling_to_directory(scaling_factor_b)
    
    print("\nNote: Uncomment one of the apply_scaling_to_directory() calls above")
    print("to actually save scaled depth maps.")


def calibrate_with_user_input():
    """
    Interactive calibration with user input
    """
    print("\n" + "="*60)
    print("INTERACTIVE DEPTH SCALING CALIBRATION")
    print("="*60 + "\n")
    
    method = input("Choose calibration method:\n1. Reference Object\n2. Camera Intrinsics\nEnter 1 or 2: ").strip()
    
    depth_dir = input("\nEnter path to depth maps directory (default: ./keyframes_with_depth/depth_maps): ").strip()
    if not depth_dir:
        depth_dir = './keyframes_with_depth/depth_maps'
    
    scaler = DepthScaler(depth_dir=depth_dir)
    
    if method == '1':
        # Reference object method
        print("\n--- Reference Object Calibration ---")
        try:
            pixel_count = float(input("How many pixels wide is the reference object in the depth map? "))
            real_size = float(input("What is the real-world size of the reference object (in cm)? "))
            
            scaling_factor = scaler.method_a_reference_object(
                reference_pixel_count=pixel_count,
                reference_real_size_cm=real_size
            )
            
            # Apply scaling
            apply = input("\nApply this scaling to all depth maps? (y/n): ").strip().lower()
            if apply == 'y':
                scaler.apply_scaling_to_directory(scaling_factor)
        except ValueError:
            print("Invalid input!")
    
    elif method == '2':
        # Camera intrinsics method
        print("\n--- Camera Intrinsics Calibration ---")
        try:
            focal = float(input("Enter focal length (mm, e.g., 28): "))
            sensor = float(input("Enter sensor width (mm, e.g., 4.5): "))
            img_width = float(input("Enter image width (pixels, e.g., 1920): "))
            
            scaling_factor = scaler.method_b_camera_intrinsics(
                focal_length_mm=focal,
                sensor_width_mm=sensor,
                image_width_px=int(img_width)
            )
            
            apply = input("\nApply this scaling to all depth maps? (y/n): ").strip().lower()
            if apply == 'y':
                scaler.apply_scaling_to_directory(scaling_factor)
        except ValueError:
            print("Invalid input!")
    else:
        print("Invalid choice!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--example':
        example_usage()
    else:
        calibrate_with_user_input()
