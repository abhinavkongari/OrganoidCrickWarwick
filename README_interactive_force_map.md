# Interactive Magnetic Tweezers Force Map

## Overview

This tool provides an interactive visualization for magnetic tweezers force mapping experiments. It allows researchers to click anywhere in a 1400×1400 pixel frame to see real-time force estimates at that location based on calibration data from particle tracking experiments.

**Author**: Abhinav's Magnetic Tweezers Lab
**Date**: November 2025

## What It Does

The `interactive_force_map.py` script:

1. **Loads calibration data** from multiple CSV files containing particle tracking information
2. **Calculates forces** using Stokes drag equation based on particle velocities
3. **Fits a power law model** (F = A / r^n) to describe force as a function of distance from the magnetic tip
4. **Generates a 2D force map** across the entire imaging frame
5. **Provides interactive visualization** where users can explore forces at any position
6. **Saves marked positions** for later analysis

## Key Features

- **Real-time cursor tracking**: Hover over the map to see force estimates at any position
- **Click to mark positions**: Left-click to permanently mark positions of interest
- **Distance rings**: Visual guides showing distance from the magnetic tip (10, 20, 30, 40, 50, 60, 80, 100 μm)
- **Force model display**: Shows the fitted power law equation and R² value
- **Trajectory visualization**: Displays all calibration bead trajectories
- **Export functionality**: Saves all marked positions to CSV

## Physical Parameters

The analysis uses the following experimental parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Particle Radius | 1.4 μm | Radius of magnetic beads |
| Glycerol Viscosity | 1.412 Pa·s | Viscosity of surrounding medium |
| Frame Rate | 7.76 fps | Camera acquisition rate |
| Pixel Size | 0.11 μm | Spatial calibration |
| Frame Size | 1400 × 1400 pixels | Imaging area |

## Input Requirements

### CSV File Format

The script expects calibration CSV files with the following columns:

1. **Track_ID**: Identifier for the tracked particle
2. **Frame**: Frame number in the video sequence
3. **X_pixels**: X position in pixels
4. **Y_pixels**: Y position in pixels
5. **Distance_ImageJ**: Distance measurement from ImageJ (optional)
6. **Velocity_ImageJ**: Velocity from ImageJ (optional)
7. **Pixel_Value**: Intensity value (optional)

### Example Data Structure

```
Track_ID,Frame,X_pixels,Y_pixels,Distance_ImageJ,Velocity_ImageJ,Pixel_Value
1,0,500,600,0,0,255
1,1,502,598,2.8,0.31,250
1,2,505,595,4.2,0.46,248
...
```

## How to Use

### Step 1: Prepare Your Data

Ensure you have:
- Multiple calibration CSV files from particle tracking experiments
- Files should contain trajectories of beads moving toward/from the magnetic tip
- The last position in each trajectory should be closest to the magnetic tip

### Step 2: Run the Script

```bash
python interactive_force_map.py
```

### Step 3: Select Calibration Files

- A file dialog will appear
- Select **ALL** calibration CSV files you want to include
- You can select multiple files at once (Ctrl+Click or Shift+Click)
- Click "Open" to proceed

### Step 4: Interact with the Force Map

The interactive window will display:

**Left Panel (Force Map)**:
- Color-coded force map (hot colormap: dark = low force, bright = high force)
- Calibration bead trajectories (colored lines)
- Magnetic tip position (cyan star)
- Distance rings at regular intervals
- Your marked positions (yellow X markers)

**Right Panel (Information)**:
- Force model equation: F(r) = A / r^n
- Model fit quality (R²)
- Magnetic tip coordinates
- Current cursor position and force (when hovering)
- List of marked positions

**Controls**:
- **Hover**: Move mouse over the map to see real-time force estimates
- **Left Click**: Mark a position to save it for later
- **Right Click**: Clear all marked positions
- **Close Window**: Exit and save marked positions

### Step 5: Review Output

When you close the window, the script will save:

**File**: `marked_positions_forces.csv`
**Location**: Same directory as your input files
**Contents**:
- X_pixels: X coordinate of marked position
- Y_pixels: Y coordinate of marked position
- Distance_from_tip_um: Distance from magnetic tip (μm)
- Force_pN: Estimated force at that position (pN)

## Understanding the Output

### Force Model

The script fits a power law model to your calibration data:

```
F(r) = A / r^n
```

Where:
- F = Force in piconewtons (pN)
- r = Distance from magnetic tip in micrometers (μm)
- A = Amplitude parameter (fitted)
- n = Decay exponent (fitted)

**Typical values**: n is usually between 2-4 for magnetic tweezers, depending on tip geometry.

### Force Calculation Method

Forces are calculated using Stokes drag:

```
F = 6πηrv × 10^12
```

Where:
- η = Viscosity (1.412 Pa·s for glycerol)
- r = Particle radius (1.4 × 10^-6 m)
- v = Particle velocity (m/s)
- Result is in piconewtons (pN)

### Velocity Calculation

```
v = √[(ΔX² + ΔY²) × (pixel_size)²] / Δt
```

Where:
- ΔX, ΔY = Displacement between frames (pixels)
- pixel_size = 0.11 μm/pixel
- Δt = 1 / frame_rate = 1/7.76 s

## Troubleshooting

### Issue: "No files selected. Exiting..."
**Solution**: Make sure to select at least one CSV file in the file dialog.

### Issue: Poor model fit (low R²)
**Causes**:
- Insufficient data points
- Non-uniform bead trajectories
- Incorrect tip position estimation
**Solution**: Collect more calibration data with beads at various distances from the tip.

### Issue: Force values seem too high/low
**Check**:
- Particle radius is correct (currently 1.4 μm)
- Viscosity matches your medium (currently 1.412 Pa·s for glycerol)
- Frame rate is accurate (currently 7.76 fps)
- Pixel size calibration (currently 0.11 μm/pixel)

### Issue: Tip position is incorrect
**Note**: The script estimates tip position as the mean of the last positions in all trajectories. For better accuracy:
- Ensure calibration trajectories end near the tip
- Use multiple beads approaching the tip from different directions

## Example Workflow

1. **Record calibration videos**: Image magnetic beads being pulled toward the tip
2. **Track particles**: Use ImageJ or similar software to generate tracking data
3. **Export to CSV**: Save tracking results in the required format
4. **Run this script**: Load all calibration files and generate the force map
5. **Explore forces**: Click around to measure forces at positions of interest
6. **Plan experiments**: Use marked positions to plan where to position cells/organoids
7. **Export data**: Save marked positions for documentation

## Technical Details

### Dependencies

```python
pandas          # Data manipulation
numpy           # Numerical operations
matplotlib      # Visualization
tkinter         # File dialog (included with Python)
```

Install with:
```bash
pip install pandas numpy matplotlib
```

### Performance

- **Processing time**: ~1-2 seconds per calibration file
- **Map resolution**: 10 pixels (adjustable in code, line 139)
- **Display limit**: Last 8 marked positions shown in panel (all are saved)

### Customization

Key parameters that can be modified in the code:

- **Line 44-48**: Physical parameters (particle size, viscosity, etc.)
- **Line 50-51**: Frame dimensions
- **Line 139**: Force map resolution
- **Line 197**: Distance ring spacing

## Citation

If you use this tool in your research, please cite:

```
Interactive Magnetic Tweezers Force Map Tool
Abhinav's Magnetic Tweezers Lab, November 2025
```

## Support

For questions or issues:
- Check that your CSV files match the required format
- Verify physical parameters match your experimental setup
- Review the console output for diagnostic messages

## Related Scripts

This tool is part of a magnetic tweezers analysis suite:
- `comprehensive_force_map.py`: Batch processing and comprehensive analysis
- `magnetic_tweezers_force_analysis.py`: Detailed force calculations

---

**Version**: 1.0
**Last Updated**: November 2025
