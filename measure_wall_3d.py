"""
3D Back-projection Tool
Convert 2D pixel coordinates to 3D world coordinates using camera intrinsics
Then calculate real-world distances between points on the wall
"""

import cv2
import numpy as np
import os
from pathlib import Path


class CameraIntrinsics:
    """Store and use camera intrinsic calibration"""
    
    @staticmethod
    def get_iphone_intrinsics(image_width=1920, image_height=1080):
        """
        Typical iPhone camera intrinsics for video
        
        iPhone specs (typical):
        - Focal length: 28mm equivalent
        - Sensor width: 4.5mm
        - Sensor height: derived from aspect ratio
        """
        focal_length_mm = 28  # mm equivalent
        sensor_width_mm = 4.5  # mm
        
        # Calculate focal length in pixels
        fx = (image_width * focal_length_mm) / sensor_width_mm
        
        # Assume square pixels, so fy = fx
        fy = fx
        
        # Principal point (image center) - usually at center for most cameras
        cx = image_width / 2
        cy = image_height / 2
        
        return fx, fy, cx, cy
    
    @staticmethod
    def pixel_to_3d(u, v, depth_cm, fx, fy, cx, cy):
        """
        Back-project pixel (u, v) with depth to 3D world coordinates
        
        Formula:
        Z = depth (in cm)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        Args:
            u, v: Pixel coordinates
            depth_cm: Depth value in centimeters
            fx, fy: Focal length in pixels
            cx, cy: Principal point in pixels
            
        Returns:
            (X, Y, Z) in centimeters
        """
        Z = depth_cm
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        return X, Y, Z
    
    @staticmethod
    def distance_3d(p1, p2):
        """
        Calculate 3D Euclidean distance between two points
        
        Formula: √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)
        """
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return dist


class WallMeasurer3D:
    """Measure wall dimensions using 3D back-projection"""
    
    def __init__(self, depth_dir='./keyframes_with_depth/depth_scaled'):
        self.depth_dir = depth_dir
        self.image_width = 1920
        self.image_height = 1080
        
        # Get camera intrinsics
        self.fx, self.fy, self.cx, self.cy = CameraIntrinsics.get_iphone_intrinsics(
            self.image_width, self.image_height
        )
        
        print("\n" + "="*70)
        print("3D BACK-PROJECTION CAMERA CALIBRATION")
        print("="*70)
        print(f"\nAssuming iPhone camera intrinsics:")
        print(f"  Focal length (fx): {self.fx:.2f} pixels")
        print(f"  Focal length (fy): {self.fy:.2f} pixels")
        print(f"  Principal point (cx, cy): ({self.cx:.0f}, {self.cy:.0f})")
        print(f"  Image size: {self.image_width} x {self.image_height}")
        print("\nFormulas used:")
        print("  X = (u - cx) * Z / fx")
        print("  Y = (v - cy) * Z / fy")
        print("  Distance = √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)")
        print("="*70 + "\n")
    
    def measure_wall_3d(self):
        """Interactive 3D wall measurement tool"""
        
        # List available depth maps
        depth_files = sorted([f for f in os.listdir(self.depth_dir)
                             if f.endswith('.png') and 'metric' in f])
        
        if not depth_files:
            print("No scaled depth maps found!")
            return
        
        print(f"Found {len(depth_files)} depth maps:\n")
        for i, f in enumerate(depth_files):
            print(f"  {i}: {f}")
        
        # Select frame
        while True:
            try:
                frame_idx = int(input(f"\nSelect frame (0-{len(depth_files)-1}): "))
                if 0 <= frame_idx < len(depth_files):
                    break
            except ValueError:
                pass
        
        selected_file = depth_files[frame_idx]
        depth_path = os.path.join(self.depth_dir, selected_file)
        
        # Load depth map
        depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_map is None:
            print(f"Error loading {selected_file}")
            return
        
        depth_cm = depth_map.astype(np.float32) / 100
        
        print(f"\n{'='*70}")
        print(f"Loaded: {selected_file}")
        print(f"Depth range: {depth_cm.min():.2f} - {depth_cm.max():.2f} cm")
        print(f"{'='*70}\n")
        
        # Visualization
        depth_normalized = ((depth_cm - depth_cm.min()) / 
                           (depth_cm.max() - depth_cm.min() + 0.0001) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        print("INSTRUCTIONS:")
        print("1. A colored depth map will appear")
        print("2. Click 2 CORNER POINTS of your wall (e.g., top-left and bottom-right)")
        print("3. The script will calculate 3D coordinates")
        print("4. You can make multiple measurements")
        print("5. Press 'q' to finish\n")
        
        points_2d = []
        measurements_3d = []
        display = depth_colored.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal points_2d, display, measurements_3d
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Get depth at this pixel
                depth_value = depth_cm[y, x]
                
                # Back-project to 3D
                X, Y, Z = CameraIntrinsics.pixel_to_3d(
                    x, y, depth_value,
                    self.fx, self.fy, self.cx, self.cy
                )
                
                points_2d.append(((x, y), (X, Y, Z)))
                
                # Draw circle
                cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
                
                if len(points_2d) == 1:
                    print(f"\nPoint 1 (2D): pixel ({x}, {y})")
                    print(f"             depth: {depth_value:.2f} cm")
                    print(f"         (3D): X={X:.2f} cm, Y={Y:.2f} cm, Z={Z:.2f} cm")
                
                elif len(points_2d) == 2:
                    p1_3d = points_2d[0][1]
                    p2_3d = points_2d[1][1]
                    
                    dist_3d = CameraIntrinsics.distance_3d(p1_3d, p2_3d)
                    
                    print(f"\nPoint 2 (2D): pixel ({x}, {y})")
                    print(f"             depth: {depth_value:.2f} cm")
                    print(f"         (3D): X={X:.2f} cm, Y={Y:.2f} cm, Z={Z:.2f} cm")
                    
                    print(f"\n{'─'*70}")
                    print(f"MEASUREMENT #{len(measurements_3d) + 1} (3D)")
                    print(f"{'─'*70}")
                    print(f"Point 1: ({p1_3d[0]:.2f}, {p1_3d[1]:.2f}, {p1_3d[2]:.2f})")
                    print(f"Point 2: ({p2_3d[0]:.2f}, {p2_3d[1]:.2f}, {p2_3d[2]:.2f})")
                    print(f"3D Distance: {dist_3d:.2f} cm ({dist_3d/100:.2f} m)")
                    
                    # Calculate individual axis differences
                    dx = abs(p2_3d[0] - p1_3d[0])
                    dy = abs(p2_3d[1] - p1_3d[1])
                    dz = abs(p2_3d[2] - p1_3d[2])
                    
                    print(f"\nAxis breakdown:")
                    print(f"  ΔX: {dx:.2f} cm (horizontal in image)")
                    print(f"  ΔY: {dy:.2f} cm (vertical in image)")
                    print(f"  ΔZ: {dz:.2f} cm (depth/distance from camera)")
                    
                    # Draw line
                    cv2.line(display, points_2d[0][0], points_2d[1][0], (0, 255, 0), 2)
                    
                    # Add label
                    mid = ((points_2d[0][0][0] + points_2d[1][0][0]) // 2,
                           (points_2d[0][0][1] + points_2d[1][0][1]) // 2)
                    cv2.putText(display, f"{dist_3d:.0f}cm", mid,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    measurements_3d.append({
                        'point1_2d': points_2d[0][0],
                        'point2_2d': points_2d[1][0],
                        'point1_3d': p1_3d,
                        'point2_3d': p2_3d,
                        'distance_3d': dist_3d,
                        'dx': dx,
                        'dy': dy,
                        'dz': dz
                    })
                    
                    points_2d = []
                
                cv2.imshow('3D Wall Measurement (Click corners)', display)
        
        cv2.namedWindow('3D Wall Measurement (Click corners)')
        cv2.setMouseCallback('3D Wall Measurement (Click corners)', mouse_callback)
        cv2.imshow('3D Wall Measurement (Click corners)', display)
        
        print("Click on the image to select wall corners...\n")
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        cv2.destroyAllWindows()
        
        # Summary
        if measurements_3d:
            print(f"\n{'='*70}")
            print("3D MEASUREMENT SUMMARY")
            print(f"{'='*70}\n")
            
            for i, m in enumerate(measurements_3d, 1):
                print(f"Measurement {i}:")
                print(f"  Corner 1: pixel {m['point1_2d']}")
                print(f"            3D: ({m['point1_3d'][0]:.2f}, {m['point1_3d'][1]:.2f}, {m['point1_3d'][2]:.2f}) cm")
                print(f"  Corner 2: pixel {m['point2_2d']}")
                print(f"            3D: ({m['point2_3d'][0]:.2f}, {m['point2_3d'][1]:.2f}, {m['point2_3d'][2]:.2f}) cm")
                print(f"  3D Distance: {m['distance_3d']:.2f} cm ({m['distance_3d']/100:.2f} m)")
                print(f"  Axis deltas: ΔX={m['dx']:.2f}, ΔY={m['dy']:.2f}, ΔZ={m['dz']:.2f} cm\n")
            
            # Save to file
            output_file = os.path.join(self.depth_dir, 'measurements_3d.txt')
            with open(output_file, 'w') as f:
                f.write("3D WALL MEASUREMENTS (Back-projection)\n")
                f.write("="*60 + "\n\n")
                f.write(f"Source: {selected_file}\n")
                f.write(f"Camera intrinsics:\n")
                f.write(f"  fx (focal length X): {self.fx:.2f} pixels\n")
                f.write(f"  fy (focal length Y): {self.fy:.2f} pixels\n")
                f.write(f"  cx (principal X): {self.cx:.0f} pixels\n")
                f.write(f"  cy (principal Y): {self.cy:.0f} pixels\n\n")
                f.write(f"Back-projection formulas:\n")
                f.write(f"  X = (u - cx) * Z / fx\n")
                f.write(f"  Y = (v - cy) * Z / fy\n")
                f.write(f"  Distance = √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)\n\n")
                
                for i, m in enumerate(measurements_3d, 1):
                    f.write(f"Measurement {i}:\n")
                    f.write(f"  Pixel coords: {m['point1_2d']} to {m['point2_2d']}\n")
                    f.write(f"  3D coords: ({m['point1_3d'][0]:.2f}, {m['point1_3d'][1]:.2f}, {m['point1_3d'][2]:.2f})")
                    f.write(f" to ({m['point2_3d'][0]:.2f}, {m['point2_3d'][1]:.2f}, {m['point2_3d'][2]:.2f})\n")
                    f.write(f"  3D Distance: {m['distance_3d']:.2f} cm ({m['distance_3d']/100:.2f} m)\n")
                    f.write(f"  Axis breakdown: ΔX={m['dx']:.2f}, ΔY={m['dy']:.2f}, ΔZ={m['dz']:.2f} cm\n\n")
            
            print(f"✓ Measurements saved to: {output_file}")
            print("="*70 + "\n")
        else:
            print("\nNo measurements taken.")


if __name__ == '__main__':
    try:
        measurer = WallMeasurer3D()
        measurer.measure_wall_3d()
    except KeyboardInterrupt:
        print("\n\nCancelled!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
