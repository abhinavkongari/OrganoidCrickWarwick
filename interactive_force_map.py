"""
Interactive Magnetic Tweezers Force Map
Click anywhere in the 1400x1400 frame to see force estimates at that location
Features: Real-time cursor tracking, click to mark positions, distance rings
Author: Abhinav's Magnetic Tweezers Lab
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import sys

# ============================================
# CURVE FITTING
# ============================================

def fit_power_law(x, y):
    """Fit y = A / x^n using log-log linear regression"""
    mask = (x > 0) & (y > 0)
    x_fit, y_fit = x[mask], y[mask]
    if len(x_fit) < 3:
        return None, None, 0
    coeffs = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
    n = -coeffs[0]
    A = np.exp(coeffs[1])
    y_pred = A / (x_fit ** n)
    r_squared = 1 - (np.sum((y_fit - y_pred)**2) / np.sum((y_fit - np.mean(y_fit))**2))
    return A, n, r_squared

def power_law(r, A, n):
    """F = A / r^n"""
    return A / (r ** n)

# ============================================
# PARAMETERS
# ============================================

PARTICLE_RADIUS = 1.4e-6
GLYCEROL_VISCOSITY = 1.412
FRAME_RATE = 7.76
PIXEL_SIZE_MICRONS = 0.11
DRAG_COEFFICIENT = 6 * np.pi * GLYCEROL_VISCOSITY * PARTICLE_RADIUS

FRAME_WIDTH = 1400
FRAME_HEIGHT = 1400

print("="*80)
print("INTERACTIVE MAGNETIC TWEEZERS FORCE MAP")
print("="*80)
print("\nFeatures:")
print("  • Click anywhere to see force at that position")
print("  • Hover to see real-time force estimates")
print("  • Right-click to clear markers")
print("  • Distance and force displayed dynamically")

# ============================================
# FILE SELECTION
# ============================================

root = tk.Tk()
root.withdraw()

print("\n" + "="*80)
print("FILE SELECTION")
print("="*80)

file_paths = filedialog.askopenfilenames(
    title="Select ALL Calibration CSV files",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if not file_paths:
    print("No files selected. Exiting...")
    sys.exit(0)

print(f"\n✓ Selected {len(file_paths)} file(s)")
output_dir = Path(file_paths[0]).parent

# ============================================
# LOAD DATA
# ============================================

print("\n" + "="*80)
print("LOADING & PROCESSING")
print("="*80)

all_calibration_data = []
dt = 1.0 / FRAME_RATE

for i, data_file in enumerate(file_paths, 1):
    df_raw = pd.read_csv(data_file, encoding='latin-1')
    df_raw.columns = ['Track_ID', 'Frame', 'X_pixels', 'Y_pixels', 
                      'Distance_ImageJ', 'Velocity_ImageJ', 'Pixel_Value']
    
    df_raw['dX_pixels'] = df_raw['X_pixels'].diff()
    df_raw['dY_pixels'] = df_raw['Y_pixels'].diff()
    df_raw['Displacement_2D_microns'] = np.sqrt(
        (df_raw['dX_pixels'] * PIXEL_SIZE_MICRONS)**2 + 
        (df_raw['dY_pixels'] * PIXEL_SIZE_MICRONS)**2
    )
    df_raw['Velocity_2D_m_per_sec'] = (df_raw['Displacement_2D_microns'] * 1e-6) / dt
    df_raw['Force_2D_pN'] = DRAG_COEFFICIENT * df_raw['Velocity_2D_m_per_sec'] * 1e12
    
    df = df_raw.dropna().copy()
    
    tip_x_est = df['X_pixels'].iloc[-1]
    tip_y_est = df['Y_pixels'].iloc[-1]
    
    df.loc[:, 'Distance_from_tip_microns'] = np.sqrt(
        (df['X_pixels'] - tip_x_est)**2 + (df['Y_pixels'] - tip_y_est)**2
    ) * PIXEL_SIZE_MICRONS
    
    all_calibration_data.append({
        'id': i, 'data': df, 'tip_x': tip_x_est, 'tip_y': tip_y_est
    })

# Determine tip
TIP_X = np.mean([c['tip_x'] for c in all_calibration_data])
TIP_Y = np.mean([c['tip_y'] for c in all_calibration_data])

# Combine and fit
all_distances = np.concatenate([c['data']['Distance_from_tip_microns'].values for c in all_calibration_data])
all_forces = np.concatenate([c['data']['Force_2D_pN'].values for c in all_calibration_data])

valid = (all_distances > 0.1) & (all_forces > 0.1)
A, n, r2 = fit_power_law(all_distances[valid], all_forces[valid])

print(f"✓ Loaded {len(all_calibration_data)} calibrations ({len(all_distances)} points)")
print(f"✓ Tip location: ({TIP_X:.0f}, {TIP_Y:.0f})")
print(f"✓ Force model: F(r) = {A:.1f} / r^{n:.2f}  (R² = {r2:.3f})")

# Generate force map
resolution = 10
x_grid = np.arange(0, FRAME_WIDTH + resolution, resolution)
y_grid = np.arange(0, FRAME_HEIGHT + resolution, resolution)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

Distance_grid = np.maximum(
    np.sqrt((X_grid - TIP_X)**2 + (Y_grid - TIP_Y)**2) * PIXEL_SIZE_MICRONS, 0.1
)
Force_grid = np.clip(power_law(Distance_grid, A, n), 0, np.percentile(all_forces, 99))

print(f"✓ Generated {X_grid.shape[0]}×{X_grid.shape[1]} force map")

# ============================================
# FORCE LOOKUP FUNCTION
# ============================================

def get_force_at_position(x_pixel, y_pixel):
    """Calculate force at given pixel position"""
    distance_um = np.sqrt((x_pixel - TIP_X)**2 + (y_pixel - TIP_Y)**2) * PIXEL_SIZE_MICRONS
    distance_um = max(distance_um, 0.1)
    force_pN = power_law(distance_um, A, n)
    return distance_um, force_pN

# ============================================
# INTERACTIVE VISUALIZATION
# ============================================

print("\n" + "="*80)
print("LAUNCHING INTERACTIVE MAP")
print("="*80)
print("\nControls:")
print("  • LEFT CLICK: Mark position and show force")
print("  • HOVER: See real-time force estimate")
print("  • RIGHT CLICK: Clear all markers")
print("  • Close window to save and exit")
print("\nLaunching...")

# Create figure
fig, (ax_main, ax_info) = plt.subplots(1, 2, figsize=(22, 11), 
                                        gridspec_kw={'width_ratios': [3, 1]})

# Plot force map
im = ax_main.contourf(X_grid, Y_grid, Force_grid, levels=60, cmap='hot', alpha=0.85)
colors_beads = plt.cm.Set2(np.linspace(0, 1, len(all_calibration_data)))

# Plot trajectories
for i, cal in enumerate(all_calibration_data):
    df = cal['data']
    ax_main.plot(df['X_pixels'], df['Y_pixels'], '-', color=colors_beads[i], 
                linewidth=2, alpha=0.7, label=f"Bead {cal['id']}")
    ax_main.plot(df['X_pixels'].iloc[0], df['Y_pixels'].iloc[0], 'o',
                color=colors_beads[i], markersize=10, markeredgecolor='white', markeredgewidth=2)

# Mark tip
ax_main.plot(TIP_X, TIP_Y, '*', color='cyan', markersize=30, 
            markeredgecolor='black', markeredgewidth=3, label='Magnetic Tip', zorder=10)

# Distance circles
for dist_um in [10, 20, 30, 40, 50, 60, 80, 100]:
    dist_px = dist_um / PIXEL_SIZE_MICRONS
    circle = Circle((TIP_X, TIP_Y), dist_px, fill=False, 
                   color='cyan', linewidth=1, alpha=0.3, linestyle='--')
    ax_main.add_patch(circle)
    angle = np.pi / 4  # 45 degrees
    label_x = TIP_X + dist_px * np.cos(angle)
    label_y = TIP_Y + dist_px * np.sin(angle)
    ax_main.text(label_x, label_y, f'{dist_um}μm', color='cyan', fontsize=8, 
                fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

ax_main.set_xlabel('X Position (pixels)', fontsize=13, fontweight='bold')
ax_main.set_ylabel('Y Position (pixels)', fontsize=13, fontweight='bold')
ax_main.set_title('Interactive Force Map - Click anywhere to see force', 
                 fontsize=15, fontweight='bold', pad=15)
ax_main.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax_main.set_xlim(0, FRAME_WIDTH)
ax_main.set_ylim(0, FRAME_HEIGHT)
ax_main.invert_yaxis()
ax_main.grid(True, alpha=0.2, color='white', linewidth=0.5)

cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
cbar.set_label('Force (pN)', fontsize=12, fontweight='bold')

# Info panel
ax_info.axis('off')
ax_info.set_xlim(0, 1)
ax_info.set_ylim(0, 1)

# Title
ax_info.text(0.5, 0.95, 'FORCE INFORMATION', ha='center', fontsize=14, 
            fontweight='bold', transform=ax_info.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=2))

# Model info
ax_info.text(0.05, 0.88, 'Force Model:', fontsize=11, fontweight='bold', 
            transform=ax_info.transAxes)
ax_info.text(0.05, 0.84, f'F(r) = {A:.1f} / r^{n:.2f}', fontsize=10, 
            transform=ax_info.transAxes, family='monospace')
ax_info.text(0.05, 0.80, f'R² = {r2:.3f}', fontsize=10, 
            transform=ax_info.transAxes, family='monospace')

# Tip info
ax_info.text(0.05, 0.73, 'Magnetic Tip:', fontsize=11, fontweight='bold', 
            transform=ax_info.transAxes)
ax_info.text(0.05, 0.69, f'X = {TIP_X:.0f} px', fontsize=10, 
            transform=ax_info.transAxes, family='monospace')
ax_info.text(0.05, 0.65, f'Y = {TIP_Y:.0f} px', fontsize=10, 
            transform=ax_info.transAxes, family='monospace')

# Current cursor position (will be updated)
cursor_text = ax_info.text(0.05, 0.55, 'HOVER OVER MAP', ha='left', fontsize=12, 
                           fontweight='bold', color='blue', transform=ax_info.transAxes)
cursor_pos_text = ax_info.text(0.05, 0.50, '', ha='left', fontsize=10, 
                               transform=ax_info.transAxes, family='monospace')
cursor_dist_text = ax_info.text(0.05, 0.46, '', ha='left', fontsize=10, 
                                transform=ax_info.transAxes, family='monospace')
cursor_force_text = ax_info.text(0.05, 0.42, '', ha='left', fontsize=11, 
                                 transform=ax_info.transAxes, family='monospace',
                                 fontweight='bold', color='red')

# Separator
ax_info.plot([0.05, 0.95], [0.38, 0.38], 'k-', linewidth=2, transform=ax_info.transAxes)

# Marked positions header
ax_info.text(0.05, 0.34, 'MARKED POSITIONS:', fontsize=11, fontweight='bold', 
            transform=ax_info.transAxes)
marked_positions_text = ax_info.text(0.05, 0.05, '', ha='left', va='bottom', fontsize=9, 
                                    transform=ax_info.transAxes, family='monospace')

# Instructions
instructions = (
    "CONTROLS:\n"
    "• Click: Mark position\n"
    "• Hover: Live force\n"
    "• Right-click: Clear all"
)
ax_info.text(0.5, 0.01, instructions, ha='center', va='bottom', fontsize=9, 
            transform=ax_info.transAxes, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

# Store marked positions
marked_positions = []
marked_markers = []

# ============================================
# INTERACTIVE CALLBACKS
# ============================================

def on_mouse_move(event):
    """Update force display when hovering"""
    if event.inaxes == ax_main and event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        if 0 <= x <= FRAME_WIDTH and 0 <= y <= FRAME_HEIGHT:
            dist_um, force_pN = get_force_at_position(x, y)
            
            cursor_text.set_text('CURSOR POSITION:')
            cursor_pos_text.set_text(f'X={x:.0f}, Y={y:.0f} px')
            cursor_dist_text.set_text(f'Distance: {dist_um:.1f} μm')
            cursor_force_text.set_text(f'Force: {force_pN:.1f} pN')
            
            fig.canvas.draw_idle()

def on_click(event):
    """Mark position on left click, clear on right click"""
    if event.inaxes == ax_main and event.xdata is not None and event.ydata is not None:
        
        if event.button == 1:  # Left click - mark position
            x, y = event.xdata, event.ydata
            if 0 <= x <= FRAME_WIDTH and 0 <= y <= FRAME_HEIGHT:
                dist_um, force_pN = get_force_at_position(x, y)
                
                # Add marker
                marker, = ax_main.plot(x, y, 'X', color='yellow', markersize=15, 
                                      markeredgecolor='black', markeredgewidth=2, zorder=20)
                marked_markers.append(marker)
                
                # Store position
                marked_positions.append({
                    'x': x, 'y': y, 'dist': dist_um, 'force': force_pN
                })
                
                # Update display
                update_marked_positions_display()
                
                print(f"\n✓ MARKED: Position ({x:.0f}, {y:.0f}) → {dist_um:.1f} μm → {force_pN:.1f} pN")
                
        elif event.button == 3:  # Right click - clear all
            clear_all_markers()

def clear_all_markers():
    """Clear all marked positions"""
    global marked_positions, marked_markers
    
    # Remove markers from plot
    for marker in marked_markers:
        marker.remove()
    
    marked_markers = []
    marked_positions = []
    
    # Update display
    update_marked_positions_display()
    
    print("\n✓ Cleared all markers")
    fig.canvas.draw_idle()

def update_marked_positions_display():
    """Update the marked positions list in info panel"""
    if not marked_positions:
        marked_positions_text.set_text('(none - click to mark)')
    else:
        lines = []
        for i, pos in enumerate(marked_positions[-8:], 1):  # Show last 8
            lines.append(f"{i}. ({pos['x']:.0f},{pos['y']:.0f})")
            lines.append(f"   {pos['dist']:.1f}μm → {pos['force']:.0f}pN")
        
        if len(marked_positions) > 8:
            lines.append(f"\n... +{len(marked_positions)-8} more")
        
        marked_positions_text.set_text('\n'.join(lines))
    
    fig.canvas.draw_idle()

# Connect events
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_press_event', on_click)

plt.tight_layout()

# Show interactive plot
print("\n✓ Interactive map is now active!")
print("  Move your mouse over the map to see forces")
print("  Click to mark positions for later reference")
plt.show()

# ============================================
# SAVE MARKED POSITIONS
# ============================================

if marked_positions:
    print("\n" + "="*80)
    print("SAVING MARKED POSITIONS")
    print("="*80)
    
    marked_df = pd.DataFrame(marked_positions)
    marked_df.columns = ['X_pixels', 'Y_pixels', 'Distance_from_tip_um', 'Force_pN']
    
    marked_file = output_dir / "marked_positions_forces.csv"
    marked_df.to_csv(marked_file, index=False)
    
    print(f"\n✓ Saved {len(marked_positions)} marked positions to:")
    print(f"  {marked_file}")
    
    print("\nMarked positions summary:")
    for i, pos in enumerate(marked_positions, 1):
        print(f"  {i}. ({pos['x']:.0f}, {pos['y']:.0f}) → {pos['dist']:.1f} μm → {pos['force']:.1f} pN")

print("\n" + "="*80)
print("SESSION COMPLETE")
print("="*80)
print(f"\nForce model: F(r) = {A:.1f} / r^{n:.2f}")
print(f"Files saved to: {output_dir}")
