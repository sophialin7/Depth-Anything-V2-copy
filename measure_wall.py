"""
Interactive wall measurement tool using scaled depth maps
Measure distances and dimensions on your wall
"""

import cv2
import numpy as np
import os
from pathlib import Path


def measure_wall():
    """
    Interactive tool to measure wall dimensions from scaled depth maps
    """
    print("\n" + "="*70)
    print("WALL MEASUREMENT TOOL")
    print("="*70)
    
    depth_dir = './keyframes_with_depth/depth_scaled'
    
    if not os.path.exists(depth_dir):
        print(f"Error: {depth_dir} not found!")
        print("Make sure you've run scale_with_book.py first")
        return
    
    # List available scaled depth maps
    depth_files = sorted([f for f in os.listdir(depth_dir) 
                         if f.endswith('.png') and 'metric' in f])
    
    if not depth_files:
        print("No scaled depth maps found!")
        return
    
    print(f"\nFound {len(depth_files)} scaled depth maps:\n")
    for i, f in enumerate(depth_files):
        print(f"  {i}: {f}")
    
    # Select which frame to measure
    while True:
        try:
            frame_idx = int(input(f"\nSelect frame to measure (0-{len(depth_files)-1}): "))
            if 0 <= frame_idx < len(depth_files):
                break
            print("Invalid selection!")
        except ValueError:
            print("Enter a number!")
    
    selected_file = depth_files[frame_idx]
    depth_path = os.path.join(depth_dir, selected_file)
    
    # Load depth map (16-bit, so divide by 100 to get cm)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    if depth_map is None:
        print(f"Error: Could not load {selected_file}")
        return
    
    depth_cm = depth_map.astype(np.float32) / 100
    
    print(f"\n{'='*70}")
    print(f"Loaded: {selected_file}")
    print(f"{'='*70}")
    print(f"\nDepth map statistics:")
    print(f"  Resolution: {depth_cm.shape[1]} x {depth_cm.shape[0]} pixels")
    print(f"  Min depth: {depth_cm.min():.2f} cm")
    print(f"  Max depth: {depth_cm.max():.2f} cm")
    print(f"  Mean depth: {depth_cm.mean():.2f} cm")
    print(f"\nScaling factor: 0.238125 cm/pixel")
    
    # Create visualization
    # Normalize for display
    depth_normalized = ((depth_cm - depth_cm.min()) / 
                       (depth_cm.max() - depth_cm.min() + 0.0001) * 255).astype(np.uint8)
    
    # Apply colormap for better visualization
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # Add text overlay
    text = f"Min: {depth_cm.min():.1f}cm  Max: {depth_cm.max():.1f}cm  Mean: {depth_cm.mean():.1f}cm"
    cv2.putText(depth_colored, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
    
    print(f"\n{'='*70}")
    print("MEASUREMENT MODE")
    print(f"{'='*70}")
    print("\nInstructions:")
    print("1. A colored depth map will appear (blue=far, red=close)")
    print("2. Click TWO POINTS on the wall to measure distance")
    print("3. The distance between them will be calculated in cm")
    print("4. You can make multiple measurements")
    print("5. Press 'q' or ESC to finish\n")
    
    points = []
    measurements = []
    display_image = depth_colored.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, display_image, measurements
        
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            
            # Get depth value at this point
            depth_value = depth_cm[y, x]
            
            # Draw circle
            cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
            
            if len(points) == 1:
                print(f"\nPoint 1: ({x}, {y}) - Depth: {depth_value:.2f} cm")
            
            elif len(points) == 2:
                # Calculate pixel distance
                pixel_dist = np.sqrt((points[1][0] - points[0][0])**2 + 
                                     (points[1][1] - points[0][1])**2)
                
                # Convert to cm using scaling factor
                cm_distance = pixel_dist * 0.238125
                
                print(f"Point 2: ({x}, {y}) - Depth: {depth_value:.2f} cm")
                print(f"\n{'─'*70}")
                print(f"MEASUREMENT #{len(measurements) + 1}")
                print(f"{'─'*70}")
                print(f"Point 1: ({points[0][0]}, {points[0][1]})")
                print(f"Point 2: ({points[1][0]}, {points[1][1]})")
                print(f"Pixel distance: {pixel_dist:.1f} pixels")
                print(f"Real distance: {cm_distance:.2f} cm ({cm_distance/100:.2f} m)")
                
                # Draw line between points
                cv2.line(display_image, points[0], points[1], (0, 255, 0), 2)
                
                # Add text label
                mid_point = ((points[0][0] + points[1][0]) // 2,
                            (points[0][1] + points[1][1]) // 2)
                label = f"{cm_distance:.1f}cm"
                cv2.putText(display_image, label, mid_point, cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 0), 2)
                
                measurements.append({
                    'point1': points[0],
                    'point2': points[1],
                    'pixel_dist': pixel_dist,
                    'cm_dist': cm_distance
                })
                
                points = []
            
            cv2.imshow('Depth Map - Click to Measure', display_image)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to undo last point
            if points:
                points.pop()
                display_image = depth_colored.copy()
                # Redraw all measurements
                for m in measurements:
                    cv2.line(display_image, m['point1'], m['point2'], (0, 255, 0), 2)
                    mid_point = ((m['point1'][0] + m['point2'][0]) // 2,
                                (m['point1'][1] + m['point2'][1]) // 2)
                    label = f"{m['cm_dist']:.1f}cm"
                    cv2.putText(display_image, label, mid_point, cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 255, 0), 2)
                cv2.imshow('Depth Map - Click to Measure', display_image)
                print("Point undone")
    
    cv2.namedWindow('Depth Map - Click to Measure')
    cv2.setMouseCallback('Depth Map - Click to Measure', mouse_callback)
    cv2.imshow('Depth Map - Click to Measure', display_image)
    
    print("Click on the image to start measuring...")
    print("(Right-click to undo last point)\n")
    
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
    
    cv2.destroyAllWindows()
    
    # Summary
    if measurements:
        print(f"\n{'='*70}")
        print("MEASUREMENT SUMMARY")
        print(f"{'='*70}\n")
        
        total_measurements = len(measurements)
        print(f"Total measurements: {total_measurements}\n")
        
        for i, m in enumerate(measurements, 1):
            print(f"Measurement {i}:")
            print(f"  From: ({m['point1'][0]}, {m['point1'][1]})")
            print(f"  To:   ({m['point2'][0]}, {m['point2'][1]})")
            print(f"  Distance: {m['cm_dist']:.2f} cm ({m['cm_dist']/100:.2f} m)\n")
        
        # Calculate average
        avg_dist = np.mean([m['cm_dist'] for m in measurements])
        print(f"Average distance: {avg_dist:.2f} cm ({avg_dist/100:.2f} m)")
        
        # Save measurements to file
        output_file = os.path.join(depth_dir, 'measurements.txt')
        with open(output_file, 'w') as f:
            f.write("WALL MEASUREMENTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Source: {selected_file}\n")
            f.write(f"Scaling factor: 0.238125 cm/pixel\n")
            f.write(f"Reference object: Red book (19.05 cm)\n\n")
            f.write(f"Total measurements: {total_measurements}\n\n")
            for i, m in enumerate(measurements, 1):
                f.write(f"Measurement {i}: {m['cm_dist']:.2f} cm\n")
            f.write(f"\nAverage: {avg_dist:.2f} cm\n")
        
        print(f"\n✓ Measurements saved to: {output_file}")
        print("="*70 + "\n")
    else:
        print("\nNo measurements taken.")


if __name__ == '__main__':
    try:
        measure_wall()
    except KeyboardInterrupt:
        print("\n\nCancelled!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
