# Interactive Magnetic Tweezers Force Map - User Guide

## 📊 Overview

You now have TWO interactive force map tools that let you click anywhere in the 1400×1400 frame and get real-time force estimates!

---

## 🎯 Option 1: Python Interactive Map (Recommended for Research)

### File: `interactive_force_map.py`

### Features:
- **Real-time hover tracking** - Force updates as you move cursor
- **Click to mark positions** - Left-click marks positions permanently
- **Clear markers** - Right-click clears all markers
- **Automatic saving** - Marked positions saved to CSV on exit
- **High-quality visualization** - Publication-ready graphics
- **Distance circles** - Visual guides at 10, 20, 30, 40, 50, 60, 80, 100 μm

### How to Use:
1. Run the script:
   ```bash
   python interactive_force_map.py
   ```

2. Select your calibration CSV files (select all 3 at once with Ctrl+Click)

3. The interactive window opens with:
   - **LEFT PANEL**: Force map with trajectories
   - **RIGHT PANEL**: Real-time force information

4. **Controls**:
   - **HOVER**: Move mouse over map → see live force estimate
   - **LEFT CLICK**: Mark a position → adds yellow X marker
   - **RIGHT CLICK**: Clear all markers
   - **CLOSE WINDOW**: Saves marked positions and exits

5. Output file created:
   - `marked_positions_forces.csv` - All clicked positions with forces

### What You See:
- Calibration bead trajectories (colored lines)
- Magnetic tip (cyan star)
- Distance circles (cyan dashed)
- Your marked positions (yellow X markers)
- Live force readout in right panel

---

## 🌐 Option 2: HTML Interactive Map (Easy, No Installation)

### File: `interactive_force_map.html`

### Features:
- **No Python needed** - Opens in any web browser
- **Beautiful interface** - Modern, colour-coded design
- **Export function** - Download marked positions as CSV
- **Instant feedback** - Real-time force display
- **Mobile-friendly** - Works on tablets/phones

### How to Use:
1. Double-click `interactive_force_map.html`
   - Opens in your default web browser
   - Or right-click → Open With → Chrome/Firefox/Edge

2. **Interface**:
   - **LEFT**: Colour-coded force map (1400×1400 visualization)
   - **RIGHT**: Information panel with live updates

3. **Controls**:
   - **HOVER**: See force at cursor position
   - **CLICK**: Mark position (numbered yellow X)
   - **Clear Button**: Remove all markers
   - **Export Button**: Download CSV of marked positions

4. **Colour Legend**:
   - 🔴 Dark Red: >800 pN (very high)
   - 🟠 Orange: 400-800 pN (high)
   - 🟡 Yellow: 200-400 pN (medium)
   - 🟢 Green: <200 pN (low)

---

## 📐 Force Model Used

Both tools use your calibrated force model:

```
F(r) = 25,441 / r^1.36
```

Where:
- F = force in picoNewtons (pN)
- r = distance from magnetic tip in micrometres (μm)
- Magnetic tip location: (459, 596) pixels
- R² = 0.848 (good fit quality)

---

## 💡 Use Cases

### Research Applications:
1. **Pre-experiment planning**
   - Check force at proposed bead positions
   - Ensure forces within target range
   - Optimize starting positions

2. **Post-experiment analysis**
   - Verify forces at observed positions
   - Compare theoretical vs actual forces
   - Validate calibration accuracy

3. **Method development**
   - Map force gradients across field of view
   - Identify optimal working regions
   - Design experiments with specific force targets

### Practical Examples:

**Example 1: Planning cell adhesion experiment**
- Target force: 50-100 pN
- Click around map to find positions
- Use positions 500-600 pixels from tip (45-55 μm)
- Export positions for microscope stage coordinates

**Example 2: Validating bead behaviour**
- Bead observed at (700, 700)
- Click position → shows 374 pN
- Compare with measured force
- Assess calibration quality

---

## 📊 Output Files

### From Python version:
**marked_positions_forces.csv**
```csv
X_pixels,Y_pixels,Distance_from_tip_um,Force_pN
700,700,30.9,374.2
540,593,11.0,883.6
640,593,22.0,496.7
```

### From HTML version:
**marked_positions_forces.csv** (same format)
- Downloaded to your Downloads folder
- Open in Excel/Origin/Python/R

---

## 🎓 Tips & Tricks

### For Python Version:
1. **Multiple sessions**: Marked positions append if file exists
2. **High precision**: Use zoom controls (matplotlib toolbar)
3. **Screenshots**: Use matplotlib save button for figures
4. **Batch analysis**: Script remembers last directory

### For HTML Version:
1. **Bookmark it**: Save to favourites for quick access
2. **Print to PDF**: Use browser print function
3. **Share easily**: Send HTML file to colleagues
4. **No internet needed**: Works completely offline

### General Tips:
1. **Distance reference**: Circles show 10 μm increments
2. **Tip location**: Cyan star marks magnetic tip
3. **Force ranges**: 
   - Close to tip (<10 μm): >1000 pN
   - Medium distance (20-30 μm): 250-500 pN
   - Far from tip (>50 μm): <200 pN
4. **Accuracy**: Model most accurate within calibration range (0-60 μm)

---

## ⚙️ Customization

### Changing Force Model Parameters:
Both files contain the model parameters at the top.

**In Python (`interactive_force_map.py`)**: Lines ~14-18
**In HTML (`interactive_force_map.html`)**: Lines ~217-221

```python
A = 25440.8      # Force coefficient
n = 1.36         # Power law exponent
TIP_X = 459      # Tip X position (pixels)
TIP_Y = 596      # Tip Y position (pixels)
```

To update:
1. Run comprehensive force map script with new calibrations
2. Note the new A and n values
3. Update in interactive map files
4. Re-run/reload

---

## 🔧 Troubleshooting

**Problem**: Python version won't start
- **Solution**: Check matplotlib installed: `pip install matplotlib pandas numpy`

**Problem**: HTML map shows wrong forces
- **Solution**: Check calibration parameters match your latest model

**Problem**: Can't export from HTML
- **Solution**: Check browser allows downloads (popup blocker)

**Problem**: Markers not visible
- **Solution**: Click closer to centre of map, away from edges

---

## 📞 Questions?

If you need help:
1. Check that calibration files are loaded correctly
2. Verify tip position makes sense (near bead endpoints)
3. Ensure force model parameters match your analysis
4. Test with known positions first

---

## 🎉 Quick Start Summary

**For immediate use**:
1. Open `interactive_force_map.html` in browser
2. Hover to see forces anywhere
3. Click to mark interesting positions
4. Export and use in your analysis

**For research workflow**:
1. Run `python interactive_force_map.py`
2. Select all calibration CSVs
3. Mark experimental positions
4. Use saved CSV for experiment planning

---

## 📚 Related Files

- `comprehensive_force_map.py` - Generates initial force map
- `force_estimates.csv` - Pre-calculated forces at key distances
- `force_map_data.npz` - Full force grid for advanced analysis
- `calibration_summary.csv` - Details of each bead calibration

Enjoy your interactive force mapping! 🧲
