"""
Magnetic Tweezers Multi-Calibration Force Map Generator
Combines multiple bead calibrations to create comprehensive force field map
Generates a single comprehensive figure with distance-based force estimates
Author: Abhinav's Magnetic Tweezers Lab
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import sys

# ============================================
# CURVE FITTING FUNCTIONS
# ============================================

def fit_power_law(x, y):
    """Fit y = A / x^n using log-log linear regression"""
    mask = (x > 0) & (y > 0)
    x_fit = x[mask]
    y_fit = y[mask]
    
    if len(x_fit) < 3:
        return None, None, 0
    
    log_x = np.log(x_fit)
    log_y = np.log(y_fit)
    coeffs = np.polyfit(log_x, log_y, 1)
    n = -coeffs[0]
    log_A = coeffs[1]
    A = np.exp(log_A)
    
    y_pred = A / (x_fit ** n)
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return A, n, r_squared

def power_law(r, A, n):
    """F = A / r^n"""
    return A / (r ** n)

# ============================================
# PARAMETERS
# ============================================

PARTICLE_RADIUS = 1.4e-6  # metres
GLYCEROL_VISCOSITY = 1.412  # Pa·s at 20°C
FRAME_RATE = 7.76  # fps
PIXEL_SIZE_MICRONS = 0.11  # microns per pixel
DRAG_COEFFICIENT = 6 * np.pi * GLYCEROL_VISCOSITY * PARTICLE_RADIUS

FRAME_WIDTH = 1400
FRAME_HEIGHT = 1400

print("="*80)
print(" MAGNETIC TWEEZERS - COMPREHENSIVE MULTI-CALIBRATION FORCE MAP")
print("="*80)
print(f"\nPhysical Parameters:")
print(f"  Bead radius: {PARTICLE_RADIUS*1e6:.2f} μm")
print(f"  Glycerol viscosity: {GLYCEROL_VISCOSITY:.3f} Pa·s")
print(f"  Frame rate: {FRAME_RATE:.2f} fps")
print(f"  Pixel size: {PIXEL_SIZE_MICRONS:.3f} μm/pixel")
print(f"  Frame dimensions: {FRAME_WIDTH}×{FRAME_HEIGHT} pixels")

# ============================================
# FILE SELECTION
# ============================================

root = tk.Tk()
root.withdraw()

print("\n" + "="*80)
print("FILE SELECTION")
print("="*80)
print("\nSelect ALL calibration CSV files (hold Ctrl/Cmd to select multiple)")

file_paths = filedialog.askopenfilenames(
    title="Select ALL ImageJ Calibration CSV files",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if not file_paths:
    print("No files selected. Exiting...")
    sys.exit(0)

print(f"\n✓ Selected {len(file_paths)} calibration file(s):")
for i, fp in enumerate(file_paths, 1):
    print(f"  {i}. {Path(fp).name}")

output_dir = Path(file_paths[0]).parent

# ============================================
# MAGNETIC TIP LOCATION
# ============================================

print("\n" + "="*80)
print("MAGNETIC TIP LOCATION")
print("="*80)

use_auto = input("\nAuto-detect tip location from data? (y/n): ").strip().lower()

if use_auto == 'y':
    TIP_X = None
    TIP_Y = None
    print("✓ Will auto-detect from bead end positions")
else:
    try:
        TIP_X = float(input("  Enter tip X position (pixels): "))
        TIP_Y = float(input("  Enter tip Y position (pixels): "))
        print(f"✓ Using manual tip location: ({TIP_X:.0f}, {TIP_Y:.0f})")
    except:
        print("Invalid input. Will auto-detect.")
        TIP_X = None
        TIP_Y = None

# ============================================
# LOAD ALL CALIBRATION DATA
# ============================================

print("\n" + "="*80)
print("LOADING CALIBRATION DATA")
print("="*80)

all_calibration_data = []
dt = 1.0 / FRAME_RATE

for i, data_file in enumerate(file_paths, 1):
    print(f"\n[Calibration {i}] Processing: {Path(data_file).name}")
    
    try:
        # Load
        df_raw = pd.read_csv(data_file, encoding='latin-1')
        df_raw.columns = ['Track_ID', 'Frame', 'X_pixels', 'Y_pixels', 
                          'Distance_ImageJ', 'Velocity_ImageJ', 'Pixel_Value']
        
        # Calculate displacements and velocities
        df_raw['Time_sec'] = (df_raw['Frame'] - df_raw['Frame'].min()) * dt
        df_raw['dX_pixels'] = df_raw['X_pixels'].diff()
        df_raw['dY_pixels'] = df_raw['Y_pixels'].diff()
        df_raw['dX_microns'] = df_raw['dX_pixels'] * PIXEL_SIZE_MICRONS
        df_raw['dY_microns'] = df_raw['dY_pixels'] * PIXEL_SIZE_MICRONS
        df_raw['Displacement_2D_microns'] = np.sqrt(df_raw['dX_microns']**2 + df_raw['dY_microns']**2)
        df_raw['Velocity_2D_um_per_sec'] = df_raw['Displacement_2D_microns'] / dt
        df_raw['Velocity_2D_m_per_sec'] = df_raw['Velocity_2D_um_per_sec'] * 1e-6
        df_raw['Force_2D_pN'] = DRAG_COEFFICIENT * df_raw['Velocity_2D_m_per_sec'] * 1e12
        
        df = df_raw.dropna().copy()
        
        # Estimate tip location
        tip_x_est = df['X_pixels'].iloc[-1]
        tip_y_est = df['Y_pixels'].iloc[-1]
        
        if TIP_X is not None:
            tip_x_est = TIP_X
            tip_y_est = TIP_Y
        
        # Calculate distances from tip
        df['Distance_from_tip_pixels'] = np.sqrt(
            (df['X_pixels'] - tip_x_est)**2 + (df['Y_pixels'] - tip_y_est)**2
        )
        df['Distance_from_tip_microns'] = df['Distance_from_tip_pixels'] * PIXEL_SIZE_MICRONS
        
        all_calibration_data.append({
            'id': i,
            'filename': Path(data_file).name,
            'data': df,
            'tip_x': tip_x_est,
            'tip_y': tip_y_est,
            'start': (df['X_pixels'].iloc[0], df['Y_pixels'].iloc[0]),
            'end': (df['X_pixels'].iloc[-1], df['Y_pixels'].iloc[-1]),
            'n_points': len(df),
            'distance_range': (df['Distance_from_tip_microns'].min(), df['Distance_from_tip_microns'].max()),
            'force_range': (df['Force_2D_pN'].min(), df['Force_2D_pN'].max()),
            'mean_force': df['Force_2D_pN'].mean()
        })
        
        print(f"  ✓ Points: {len(df)}")
        print(f"  ✓ Path: ({df['X_pixels'].iloc[0]:.0f},{df['Y_pixels'].iloc[0]:.0f}) → "
              f"({df['X_pixels'].iloc[-1]:.0f},{df['Y_pixels'].iloc[-1]:.0f})")
        print(f"  ✓ Distance: {df['Distance_from_tip_microns'].min():.1f} - {df['Distance_from_tip_microns'].max():.1f} μm")
        print(f"  ✓ Force: {df['Force_2D_pN'].min():.1f} - {df['Force_2D_pN'].max():.1f} pN (mean: {df['Force_2D_pN'].mean():.1f})")
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")

if not all_calibration_data:
    print("\n✗ No valid calibration data. Exiting...")
    sys.exit(1)

# Determine tip location
if TIP_X is None:
    TIP_X = np.mean([cal['tip_x'] for cal in all_calibration_data])
    TIP_Y = np.mean([cal['tip_y'] for cal in all_calibration_data])
    print(f"\n✓ Auto-detected tip location: ({TIP_X:.0f}, {TIP_Y:.0f})")

# ============================================
# COMBINE AND FIT
# ============================================

print("\n" + "="*80)
print("COMBINING CALIBRATIONS & FITTING FORCE MODEL")
print("="*80)

all_distances = []
all_forces = []
all_x = []
all_y = []

for cal in all_calibration_data:
    df = cal['data']
    all_distances.extend(df['Distance_from_tip_microns'].values)
    all_forces.extend(df['Force_2D_pN'].values)
    all_x.extend(df['X_pixels'].values)
    all_y.extend(df['Y_pixels'].values)

all_distances = np.array(all_distances)
all_forces = np.array(all_forces)

print(f"\nCombined Dataset:")
print(f"  Total measurements: {len(all_distances)}")
print(f"  Distance range: {all_distances.min():.1f} - {all_distances.max():.1f} μm")
print(f"  Force range: {all_forces.min():.1f} - {all_forces.max():.1f} pN")
print(f"  Mean force: {all_forces.mean():.1f} ± {all_forces.std():.1f} pN")

# Fit power law
valid = (all_distances > 0.1) & (all_forces > 0.1)
A, n, r2 = fit_power_law(all_distances[valid], all_forces[valid])

print(f"\nFitted Force Model:")
print(f"  F(r) = {A:.1f} / r^{n:.2f}")
print(f"  R² = {r2:.3f}")
print(f"  {'Excellent' if r2 > 0.9 else 'Good' if r2 > 0.8 else 'Moderate'} fit quality")

# ============================================
# GENERATE 2D FORCE MAP
# ============================================

print("\n" + "="*80)
print("GENERATING 2D FORCE FIELD")
print("="*80)

resolution = 10
x_grid = np.arange(0, FRAME_WIDTH + resolution, resolution)
y_grid = np.arange(0, FRAME_HEIGHT + resolution, resolution)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

Distance_grid = np.sqrt((X_grid - TIP_X)**2 + (Y_grid - TIP_Y)**2) * PIXEL_SIZE_MICRONS
Distance_grid = np.maximum(Distance_grid, 0.1)
Force_grid = power_law(Distance_grid, A, n)
Force_grid = np.clip(Force_grid, 0, np.percentile(all_forces, 99))

print(f"✓ Generated {X_grid.shape[0]}×{X_grid.shape[1]} force map")
print(f"  Resolution: {resolution} pixels ({resolution*PIXEL_SIZE_MICRONS:.2f} μm)")
print(f"  Force range: {Force_grid.min():.1f} - {Force_grid.max():.1f} pN")

# ============================================
# CREATE COMPREHENSIVE FIGURE
# ============================================

print("\n" + "="*80)
print("CREATING COMPREHENSIVE VISUALIZATION")
print("="*80)

fig = plt.figure(figsize=(24, 14))
colors = plt.cm.Set2(np.linspace(0, 1, len(all_calibration_data)))

# ============================================
# MAIN PLOT: 2D Force Map with All Trajectories
# ============================================
ax_main = plt.subplot(2, 4, (1, 6))
im = ax_main.contourf(X_grid, Y_grid, Force_grid, levels=60, cmap='hot', alpha=0.85)

# Plot all trajectories
for i, cal in enumerate(all_calibration_data):
    df = cal['data']
    ax_main.plot(df['X_pixels'], df['Y_pixels'], '-', 
                color=colors[i], linewidth=3, alpha=0.8, label=f"Bead {cal['id']}")
    ax_main.plot(df['X_pixels'].iloc[0], df['Y_pixels'].iloc[0], 'o',
                color=colors[i], markersize=12, markeredgecolor='white', 
                markeredgewidth=2.5, zorder=5)
    ax_main.plot(df['X_pixels'].iloc[-1], df['Y_pixels'].iloc[-1], 's',
                color=colors[i], markersize=10, markeredgecolor='white', 
                markeredgewidth=2, zorder=5)

# Mark magnetic tip
ax_main.plot(TIP_X, TIP_Y, '*', color='cyan', markersize=30, 
            markeredgecolor='black', markeredgewidth=3, label='Magnetic Tip', zorder=10)

# Distance circles
distances_to_show = [10, 20, 30, 40, 50, 60]  # μm
for dist_um in distances_to_show:
    dist_px = dist_um / PIXEL_SIZE_MICRONS
    circle = plt.Circle((TIP_X, TIP_Y), dist_px, fill=False, 
                       color='cyan', linewidth=1.5, alpha=0.4, linestyle='--')
    ax_main.add_patch(circle)
    # Label the distance
    ax_main.text(TIP_X + dist_px, TIP_Y, f'{dist_um}μm', 
                color='cyan', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

ax_main.set_xlabel('X Position (pixels)', fontsize=14, fontweight='bold')
ax_main.set_ylabel('Y Position (pixels)', fontsize=14, fontweight='bold')
ax_main.set_title(f'2D Magnetic Force Field Map\nCombined from {len(all_calibration_data)} Bead Calibrations', 
                 fontsize=16, fontweight='bold', pad=20)
ax_main.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax_main.set_xlim(0, FRAME_WIDTH)
ax_main.set_ylim(0, FRAME_HEIGHT)
ax_main.invert_yaxis()
ax_main.grid(True, alpha=0.2, color='white', linewidth=0.5)

cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
cbar.set_label('Magnetic Force (pN)', fontsize=13, fontweight='bold')
cbar.ax.tick_params(labelsize=11)

# ============================================
# FORCE vs DISTANCE - ALL DATA
# ============================================
ax1 = plt.subplot(2, 4, 3)
for i, cal in enumerate(all_calibration_data):
    df = cal['data']
    ax1.scatter(df['Distance_from_tip_microns'], df['Force_2D_pN'],
               alpha=0.6, s=40, color=colors[i], label=f"Bead {cal['id']}", edgecolors='black', linewidth=0.5)

r_plot = np.linspace(0.1, max(all_distances), 300)
ax1.plot(r_plot, power_law(r_plot, A, n), 'r-', linewidth=4, 
        label=f'Fit: F = {A:.0f}/r$^{{{n:.2f}}}$\nR² = {r2:.3f}', zorder=10)

ax1.set_xlabel('Distance from Tip (μm)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Force (pN)', fontsize=12, fontweight='bold')
ax1.set_title('Force vs Distance\n(All Calibrations)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')

# ============================================
# FORCE ESTIMATES AT SPECIFIC DISTANCES
# ============================================
ax2 = plt.subplot(2, 4, 4)

# Calculate force at key distances
key_distances = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100])  # μm
key_forces = power_law(key_distances, A, n)

bars = ax2.barh(range(len(key_distances)), key_forces, color='steelblue', 
                edgecolor='black', linewidth=1.5, alpha=0.8)

# Color code by force magnitude
for i, (bar, force) in enumerate(zip(bars, key_forces)):
    if force > 500:
        bar.set_color('darkred')
    elif force > 300:
        bar.set_color('orangered')
    elif force > 200:
        bar.set_color('orange')
    else:
        bar.set_color('steelblue')

ax2.set_yticks(range(len(key_distances)))
ax2.set_yticklabels([f'{d} μm' for d in key_distances], fontsize=11)
ax2.set_xlabel('Estimated Force (pN)', fontsize=12, fontweight='bold')
ax2.set_title('Force Estimates at\nKey Distances from Tip', fontsize=13, fontweight='bold')
ax2.grid(True, axis='x', alpha=0.3)

# Add force values as text
for i, (dist, force) in enumerate(zip(key_distances, key_forces)):
    ax2.text(force + 20, i, f'{force:.0f} pN', 
            va='center', fontsize=10, fontweight='bold')

# ============================================
# RADIAL FORCE PROFILE
# ============================================
ax3 = plt.subplot(2, 4, 7)

max_dist = np.sqrt((FRAME_WIDTH/2)**2 + (FRAME_HEIGHT/2)**2) * PIXEL_SIZE_MICRONS
r_bins = np.linspace(0, max_dist, 60)
r_centers = (r_bins[:-1] + r_bins[1:]) / 2
force_radial = []

for i in range(len(r_bins)-1):
    mask = (Distance_grid >= r_bins[i]) & (Distance_grid < r_bins[i+1])
    if mask.any():
        force_radial.append(np.mean(Force_grid[mask]))
    else:
        force_radial.append(np.nan)

ax3.fill_between(r_centers, 0, force_radial, alpha=0.3, color='steelblue')
ax3.plot(r_centers, force_radial, 'o-', linewidth=2.5, markersize=5, 
        color='darkblue', markerfacecolor='steelblue', markeredgecolor='darkblue')

ax3.set_xlabel('Distance from Tip (μm)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Average Force (pN)', fontsize=12, fontweight='bold')
ax3.set_title('Radial Force Profile', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# ============================================
# CALIBRATION SUMMARY TABLE
# ============================================
ax4 = plt.subplot(2, 4, 8)
ax4.axis('off')

table_data = []
table_data.append(['Bead', 'Points', 'Dist Range (μm)', 'Force Range (pN)', 'Mean Force (pN)'])

for cal in all_calibration_data:
    table_data.append([
        f"{cal['id']}",
        f"{cal['n_points']}",
        f"{cal['distance_range'][0]:.1f}-{cal['distance_range'][1]:.1f}",
        f"{cal['force_range'][0]:.0f}-{cal['force_range'][1]:.0f}",
        f"{cal['mean_force']:.0f}"
    ])

table_data.append(['', '', '', '', ''])
table_data.append(['COMBINED', f"{len(all_distances)}", 
                  f"{all_distances.min():.1f}-{all_distances.max():.1f}",
                  f"{all_forces.min():.0f}-{all_forces.max():.0f}",
                  f"{all_forces.mean():.0f}"])

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.12, 0.12, 0.25, 0.25, 0.26])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(table_data)):
    for j in range(5):
        if i == len(table_data) - 1:  # Last row (COMBINED)
            table[(i, j)].set_facecolor('#90EE90')
            table[(i, j)].set_text_props(weight='bold')
        elif i == len(table_data) - 2:  # Empty row
            table[(i, j)].set_facecolor('#FFFFFF')
        else:
            table[(i, j)].set_facecolor('#E7E6E6' if i % 2 == 0 else '#F5F5F5')

ax4.set_title('Calibration Summary', fontsize=13, fontweight='bold', pad=20)

# ============================================
# Add overall title and info
# ============================================
fig.suptitle(f'Comprehensive Magnetic Tweezers Force Map Analysis\n' + 
             f'Model: F(r) = {A:.1f} / r^{n:.2f}  |  R² = {r2:.3f}  |  ' +
             f'Tip Location: ({TIP_X:.0f}, {TIP_Y:.0f})',
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
plot_file = output_dir / "comprehensive_force_map.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved comprehensive figure to:")
print(f"  {plot_file}")

# ============================================
# SAVE DATA
# ============================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save force map
np.savez(output_dir / "force_map_data.npz",
         X_grid=X_grid, Y_grid=Y_grid, Force_grid=Force_grid,
         Distance_grid=Distance_grid, tip_x=TIP_X, tip_y=TIP_Y)
print(f"✓ force_map_data.npz")

# Save parameters
params_df = pd.DataFrame({
    'Parameter': ['Tip_X', 'Tip_Y', 'Model_A', 'Model_n', 'R_squared',
                  'Num_Calibrations', 'Total_Points'],
    'Value': [TIP_X, TIP_Y, A, n, r2, len(all_calibration_data), len(all_distances)]
})
params_df.to_csv(output_dir / "force_model_parameters.csv", index=False)
print(f"✓ force_model_parameters.csv")

# Save force estimates
force_estimates = pd.DataFrame({
    'Distance_um': key_distances,
    'Estimated_Force_pN': key_forces
})
force_estimates.to_csv(output_dir / "force_estimates_by_distance.csv", index=False)
print(f"✓ force_estimates_by_distance.csv")

# Save calibration summary
cal_summary = []
for cal in all_calibration_data:
    cal_summary.append({
        'Bead_ID': cal['id'],
        'Filename': cal['filename'],
        'N_Points': cal['n_points'],
        'Start_X': cal['start'][0],
        'Start_Y': cal['start'][1],
        'End_X': cal['end'][0],
        'End_Y': cal['end'][1],
        'Min_Distance_um': cal['distance_range'][0],
        'Max_Distance_um': cal['distance_range'][1],
        'Min_Force_pN': cal['force_range'][0],
        'Max_Force_pN': cal['force_range'][1],
        'Mean_Force_pN': cal['mean_force']
    })
cal_summary_df = pd.DataFrame(cal_summary)
cal_summary_df.to_csv(output_dir / "calibration_summary.csv", index=False)
print(f"✓ calibration_summary.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\n✓ Combined {len(all_calibration_data)} calibrations")
print(f"✓ Total measurements: {len(all_distances)}")
print(f"✓ Force model: F = {A:.1f} / r^{n:.2f}  (R² = {r2:.3f})")
print(f"✓ All files saved to: {output_dir}")
print("\nKey force estimates:")
for dist, force in zip([10, 20, 30, 50], power_law(np.array([10, 20, 30, 50]), A, n)):
    print(f"  At {dist} μm: {force:.1f} pN")

plt.show()
