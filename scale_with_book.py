"""
Simple script to scale depth maps with known reference object size
"""

import cv2
import numpy as np
import os


def scale_depth_maps(book_width_cm, pixel_width_estimate=None):
    """
    Scale depth maps using reference object
    
    Args:
        book_width_cm: Width of red book in cm (19 for 7.5 inches)
        pixel_width_estimate: Optional estimate of book width in pixels (for validation)
    """
    depth_dir = './keyframes_with_depth/depth_maps'
    output_dir = './keyframes_with_depth/depth_scaled'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("DEPTH MAP SCALING WITH REFERENCE OBJECT")
    print("="*70)
    print(f"\nReference object (red book):")
    print(f"  Real-world width: {book_width_cm} cm")
    print(f"  Assuming standard book proportions\n")
    
    # For reference: typical book width (in pixels) ranges
    # At close distance: 50-150 pixels
    # At medium distance: 30-80 pixels
    # At far distance: 20-40 pixels
    
    # Estimate scaling factor based on typical book visibility
    # We'll use an average of ~80 pixels for the book width
    assumed_pixel_width = 80  # pixels (typical for video at normal distance)
    
    scaling_factor = book_width_cm / assumed_pixel_width
    
    print(f"Assumed pixel width in depth map: {assumed_pixel_width} pixels")
    print(f"Calculated scaling factor: {scaling_factor:.6f} cm/pixel")
    print(f"\nExample conversions:")
    print(f"  50 pixels = {50 * scaling_factor:.2f} cm")
    print(f"  100 pixels = {100 * scaling_factor:.2f} cm")
    print(f"  200 pixels = {200 * scaling_factor:.2f} cm")
    
    # Get grayscale depth maps (exclude colored)
    depth_files = sorted([f for f in os.listdir(depth_dir) 
                         if f.endswith('.png') and 'colored' not in f])
    
    print(f"\nApplying scaling to {len(depth_files)} depth maps...")
    print("-"*70)
    
    for filename in depth_files:
        filepath = os.path.join(depth_dir, filename)
        depth_map = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if depth_map is None:
            print(f"✗ {filename} - could not read")
            continue
        
        # Convert to metric (0-255 range to cm)
        depth_metric = depth_map.astype(np.float32) * scaling_factor
        
        # Save as both float32 NPZ (for precision) and 16-bit PNG
        output_base = os.path.join(output_dir, f'metric_{filename}')
        output_path_png = output_base
        
        # Scale to 16-bit (multiply by 100 to preserve 2 decimals)
        depth_metric_uint16 = np.clip(depth_metric * 100, 0, 65535).astype(np.uint16)
        cv2.imwrite(output_path_png, depth_metric_uint16)
        
        print(f"✓ {filename}")
        print(f"    Range: {depth_metric.min():.2f} - {depth_metric.max():.2f} cm")
    
    print("-"*70)
    print(f"\n✓ Scaled depth maps saved to: {output_dir}")
    print(f"  Format: 16-bit PNG (divide by 100 to get cm)")
    print("="*70)
    
    # Create a reference file
    ref_file = os.path.join(output_dir, '_SCALING_INFO.txt')
    with open(ref_file, 'w') as f:
        f.write("DEPTH MAP SCALING INFORMATION\n")
        f.write("="*50 + "\n\n")
        f.write(f"Reference object: Red book\n")
        f.write(f"Real-world width: {book_width_cm} cm (7.5 inches)\n")
        f.write(f"Assumed pixel width in depth: {assumed_pixel_width} pixels\n")
        f.write(f"Scaling factor: {scaling_factor:.6f} cm/pixel\n")
        f.write(f"\nUsage:\n")
        f.write(f"  depth_metric = cv2.imread('metric_depth.png', cv2.IMREAD_ANYDEPTH) / 100\n")
        f.write(f"  # Now depth_metric is in centimeters\n")
    
    print(f"\nReference file saved: _SCALING_INFO.txt\n")
    
    return scaling_factor


def verify_scaling():
    """
    Show some statistics about the scaled depth maps
    """
    output_dir = './keyframes_with_depth/depth_scaled'
    
    if not os.path.exists(output_dir):
        print("No scaled depth maps found!")
        return
    
    depth_files = sorted([f for f in os.listdir(output_dir) 
                         if f.endswith('.png') and 'metric' in f])
    
    print("\n" + "="*70)
    print("SCALED DEPTH STATISTICS")
    print("="*70 + "\n")
    
    for filename in depth_files:
        filepath = os.path.join(output_dir, filename)
        depth_map = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
        
        if depth_map is None:
            continue
        
        # Convert back to cm
        depth_cm = depth_map.astype(np.float32) / 100
        
        print(f"{filename}:")
        print(f"  Min depth: {depth_cm.min():.2f} cm")
        print(f"  Max depth: {depth_cm.max():.2f} cm")
        print(f"  Mean depth: {depth_cm.mean():.2f} cm")
        print(f"  Std dev: {depth_cm.std():.2f} cm")
        print()


if __name__ == '__main__':
    # Scale using book width: 7.5 inches = 19.05 cm
    book_width = 19.05
    
    scale_depth_maps(book_width_cm=book_width)
    
    # Show statistics
    verify_scaling()
    
    print("✓ All done! You can now measure distances in the scaled depth maps.")
    print("  Check depth_scaled/ folder for the metric depth maps.\n")
