"""
Interactive tool to measure reference object in depth maps
and calculate scaling factor
"""

import cv2
import numpy as np
import os
from pathlib import Path


def measure_reference_object():
    """
    Interactive measurement tool for reference objects in depth maps
    """
    print("\n" + "="*70)
    print("REFERENCE OBJECT MEASUREMENT TOOL")
    print("="*70)
    
    # Find available keyframes and depth maps
    keyframe_dir = './keyframes_with_depth/keyframes'
    depth_dir = './keyframes_with_depth/depth_maps'
    
    if not os.path.exists(keyframe_dir):
        print(f"Error: {keyframe_dir} not found!")
        return
    
    # List available frames
    keyframes = sorted([f for f in os.listdir(keyframe_dir) if f.endswith(('.jpg', '.png'))])
    
    if not keyframes:
        print("No keyframes found!")
        return
    
    print(f"\nFound {len(keyframes)} keyframes:\n")
    for i, frame in enumerate(keyframes):
        print(f"  {i}: {frame}")
    
    # Select which frame to measure
    while True:
        try:
            frame_idx = int(input(f"\nSelect frame (0-{len(keyframes)-1}): "))
            if 0 <= frame_idx < len(keyframes):
                break
            print("Invalid selection!")
        except ValueError:
            print("Enter a number!")
    
    selected_frame = keyframes[frame_idx]
    frame_name = os.path.splitext(selected_frame)[0]
    
    # Load images
    keyframe_path = os.path.join(keyframe_dir, selected_frame)
    
    # Find corresponding depth map (without '_colored')
    depth_maps = [f for f in os.listdir(depth_dir) 
                  if f.startswith(frame_name.replace('keyframe', 'depth')) 
                  and 'colored' not in f]
    
    if not depth_maps:
        print(f"Error: Could not find depth map for {selected_frame}")
        return
    
    depth_path = os.path.join(depth_dir, depth_maps[0])
    
    print(f"\nLoading:")
    print(f"  Keyframe: {selected_frame}")
    print(f"  Depth map: {depth_maps[0]}")
    
    # Read images
    keyframe = cv2.imread(keyframe_path)
    depth_gray = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    
    if keyframe is None or depth_gray is None:
        print("Error loading images!")
        return
    
    # Display keyframe
    print("\n" + "-"*70)
    print("STEP 1: View the KEYFRAME")
    print("-"*70)
    print("This is your original video frame with the reference object.")
    print("A window will open showing this image.")
    print("Close the window when ready. (Press any key to continue)\n")
    
    cv2.namedWindow('Keyframe (Original)', cv2.WINDOW_NORMAL)
    cv2.imshow('Keyframe (Original)', keyframe)
    print("Press any key in the image window to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Display depth map
    print("\n" + "-"*70)
    print("STEP 2: View the DEPTH MAP")
    print("-"*70)
    print("This is the depth map where:")
    print("  - BRIGHTER (255) = CLOSER to camera")
    print("  - DARKER (0) = FARTHER from camera")
    print("Close the window when ready.\n")
    
    cv2.namedWindow('Depth Map (Grayscale)', cv2.WINDOW_NORMAL)
    cv2.imshow('Depth Map (Grayscale)', depth_gray)
    print("Press any key in the image window to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Interactive measurement
    print("\n" + "-"*70)
    print("STEP 3: MEASURE REFERENCE OBJECT IN DEPTH MAP")
    print("-"*70)
    print("\nYou will now draw a line across your reference object in the depth map.")
    print("Click two points to define the width of your reference object.")
    print("  - Left-click to place first point")
    print("  - Left-click to place second point")
    print("  - The distance between them is the pixel width\n")
    
    # Copy for drawing
    depth_color = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, depth_color
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Draw circle at click point
            cv2.circle(depth_color, (x, y), 5, (0, 255, 0), -1)
            
            if len(points) == 1:
                print(f"Point 1 marked at ({x}, {y})")
            elif len(points) == 2:
                # Draw line between points
                cv2.line(depth_color, points[0], points[1], (0, 255, 0), 2)
                distance = np.sqrt((points[1][0] - points[0][0])**2 + 
                                  (points[1][1] - points[0][1])**2)
                print(f"Point 2 marked at ({x}, {y})")
                print(f"Distance: {distance:.1f} pixels")
            
            cv2.imshow('Measure Reference Object', depth_color)
    
    cv2.namedWindow('Measure Reference Object')
    cv2.setMouseCallback('Measure Reference Object', mouse_callback)
    cv2.imshow('Measure Reference Object', depth_color)
    
    print("Click on the image to mark the width of your reference object...")
    
    while len(points) < 2:
        key = cv2.waitKey(100)
        if key == ord('q'):
            cv2.destroyAllWindows()
            print("Cancelled!")
            return
    
    cv2.destroyAllWindows()
    
    # Calculate pixel width
    pixel_width = np.sqrt((points[1][0] - points[0][0])**2 + 
                         (points[1][1] - points[0][1])**2)
    
    print("\n" + "-"*70)
    print("STEP 4: ENTER REFERENCE OBJECT REAL-WORLD SIZE")
    print("-"*70)
    print(f"\nPixel width in depth map: {pixel_width:.1f} pixels")
    print("\nNow enter the REAL-WORLD size of your reference object.")
    print("Examples:")
    print("  - A4 paper width: 21 cm")
    print("  - US Letter width: 21.6 cm")
    print("  - Standard ruler: 30 cm")
    print("  - Credit card: 8.6 cm\n")
    
    while True:
        try:
            real_size_cm = float(input("Enter reference object size (in cm): "))
            if real_size_cm > 0:
                break
            print("Size must be positive!")
        except ValueError:
            print("Enter a number!")
    
    # Calculate scaling factor
    scaling_factor = real_size_cm / pixel_width
    
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE!")
    print("="*70)
    print(f"\nReference object:")
    print(f"  Real-world size: {real_size_cm} cm")
    print(f"  Pixel width in depth map: {pixel_width:.1f} pixels")
    print(f"\n✓ Scaling factor: {scaling_factor:.6f} cm/pixel")
    print(f"\nExample measurements:")
    print(f"  100 pixels × {scaling_factor:.6f} = {100 * scaling_factor:.2f} cm")
    print(f"  200 pixels × {scaling_factor:.6f} = {200 * scaling_factor:.2f} cm")
    print(f"  500 pixels × {scaling_factor:.6f} = {500 * scaling_factor:.2f} cm")
    print("="*70)
    
    # Ask to apply scaling
    apply_scaling = input("\nApply this scaling to all depth maps? (y/n): ").strip().lower()
    
    if apply_scaling == 'y':
        apply_scaling_to_all_depths(scaling_factor)
    else:
        print(f"\nScaling factor saved: {scaling_factor:.6f} cm/pixel")
        print("To apply later, run: python scale_depth_to_metric.py")
    
    return scaling_factor


def apply_scaling_to_all_depths(scaling_factor):
    """Apply scaling factor to all depth maps"""
    depth_dir = './keyframes_with_depth/depth_maps'
    output_dir = './keyframes_with_depth/depth_scaled'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get grayscale depth maps (exclude colored)
    depth_files = sorted([f for f in os.listdir(depth_dir) 
                         if f.endswith('.png') and 'colored' not in f])
    
    print(f"\nApplying scaling factor to {len(depth_files)} depth maps...\n")
    
    for filename in depth_files:
        filepath = os.path.join(depth_dir, filename)
        depth_map = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if depth_map is None:
            continue
        
        # Convert to metric (0-255 range to cm)
        depth_metric = depth_map.astype(np.float32) * scaling_factor
        
        # Save as 16-bit PNG
        output_path = os.path.join(output_dir, f'metric_{filename}')
        depth_metric_uint16 = np.clip(depth_metric * 100, 0, 65535).astype(np.uint16)
        cv2.imwrite(output_path, depth_metric_uint16)
        
        print(f"✓ {filename}")
        print(f"  Range: {depth_metric.min():.2f} - {depth_metric.max():.2f} cm")
    
    print(f"\n✓ Scaled depth maps saved to: {output_dir}")
    print("\nYou can now measure distances in these depth maps!")


if __name__ == '__main__':
    try:
        measure_reference_object()
    except KeyboardInterrupt:
        print("\n\nCancelled!")
    except Exception as e:
        print(f"\nError: {e}")
