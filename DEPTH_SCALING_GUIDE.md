# Scaling Relative Depth to Metric Depth

Depth Anything V2 provides **relative depth** (which pixels are closer/farther) but lacks scale. To measure real-world distances, you need to calibrate using one of two methods.

---

## **Method A: Reference Object** (Simpler, No equipment needed)

### How it works:
1. Place an object of **known size** on the wall (ruler, paper, credit card, etc.)
2. In the depth map, measure how many pixels the object spans
3. Calculate a scaling factor
4. Apply to all depth maps

### Example:
- **Real object**: A4 paper (21 cm = 210 mm wide)
- **In depth map**: The paper appears 50 pixels wide
- **Scaling factor**: 21 cm ÷ 50 pixels = **0.42 cm/pixel**
- **Measurement**: If a wall region is 100 pixels wide → 100 × 0.42 = **42 cm** in real world

### Math:
```
scaling_factor = real_world_size_cm / pixel_width_in_depth_map
metric_depth = relative_depth_value × scaling_factor
```

### Best practices:
- Use a straight object (ruler, level) for accuracy
- Orient it parallel to the image plane
- Place it at roughly the same distance as what you want to measure
- Average multiple measurements for robustness

### Step-by-step:
1. Look at your keyframe image (original)
2. Identify where the reference object is
3. Look at the corresponding depth map
4. Count the pixels the object spans
5. Calculate: `scaling = object_real_size / pixel_count`
6. Apply to all depth maps

---

## **Method B: Camera Intrinsics** (More precise, requires camera specs)

### How it works:
Uses the **pinhole camera model** to relate pixel positions to real-world coordinates.

### Required information:
- **Focal length** (in mm, e.g., 28mm equivalent)
- **Sensor size** (in mm, e.g., 4.5mm width)
- **Image resolution** (in pixels, e.g., 1920 width)

### For common devices:

#### iPhone (typical):
- Focal length: 28mm equivalent
- Sensor width: ~4.5mm
- Video resolution: 1920 × 1080

#### Android phones (varies):
- Focal length: 25-35mm equivalent
- Sensor width: 4-6mm
- Video resolution: 1920 × 1080 or higher

### Math:
```
Field of View (radians) = 2 × arctan(sensor_width / (2 × focal_length))

At distance D meters:
Real_world_width_mm = (image_width_pixels × D × 1000) / (focal_length × image_width_pixels / (2 × tan(FOV/2)))
```

### Example:
- Focal length: 28mm
- Sensor: 4.5mm
- Resolution: 1920px
- At 1 meter distance, field of view covers: 2 × arctan(4.5 / (2 × 28)) ≈ 9.2 meters

---

## **How to Find Your Camera Specs:**

### iPhone:
- Open Settings → Camera → Check model
- Apple publishes focal lengths (usually 28mm equivalent for main camera)

### Android:
```bash
adb shell getprop ro.telephony.use_old_mnc_mcc_format
# Or check ExifData of photos taken
```

### From your video:
1. Open the MP4 in your computer
2. Right-click → Properties → Details
3. Look for "Camera Model" or metadata

---

## **Practical Workflow:**

### Option 1: Quick & Dirty (Reference Object)
1. Place a ruler (30cm) on your wall
2. Take a keyframe screenshot
3. Count pixels ruler spans (e.g., 60 pixels)
4. Scaling = 30cm / 60px = 0.5 cm/pixel
5. Done!

### Option 2: More Accurate (Camera Intrinsics)
1. Look up your phone's focal length specs
2. Find sensor width (usually in specs or EXIF data)
3. Note video resolution
4. Let the script calculate scaling automatically

---

## **Using the Script:**

```bash
# Run the script interactively
python scale_depth_to_metric.py

# Or use the example
python scale_depth_to_metric.py --example
```

### Interactive walk-through:
1. Choose calibration method (1 or 2)
2. Enter path to depth maps
3. Enter your measurements/camera specs
4. Script calculates scaling factor
5. Script applies to all depth maps
6. Outputs saved to `depth_scaled/` folder

---

## **Output:**

After scaling, you get:
- Depth maps in **centimeters** (or meters, depending on your input)
- Saved as 16-bit PNG (preserves precision)
- Can be read back with: `depth_m = cv2.imread('metric_depth.png', cv2.IMREAD_ANYDEPTH) / 100`

---

## **Important Notes:**

1. **Distance matters**: Depth accuracy depends on distance. Closer objects → more accurate.
2. **Perspective**: Objects at image edges appear distorted. Center is most accurate.
3. **Lighting**: Uniform lighting helps depth estimation. Shadows reduce accuracy.
4. **Surface texture**: Smooth walls may be harder to estimate than textured ones.
5. **Validation**: Always verify with a physical measurement!

---

## **Example calculation:**

Say you measure a wall:
- **Physical measurement**: 150 cm wide
- **Depth map measurement**: 71.4 pixels wide (after scaling)
- **Result**: Matches! ✓

If it doesn't match:
- Adjust scaling factor
- Re-measure from multiple locations
- Average the results

---

## **Troubleshooting:**

| Problem | Solution |
|---------|----------|
| Scaling seems off | Re-measure reference object pixel count |
| Values don't match physical | Camera might have different specs, recalibrate |
| Depth seems inverted | Check if 0=closer or 255=closer in your depth maps |
| Values too small/large | Check units (cm vs mm vs m) |

