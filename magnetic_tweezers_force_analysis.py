"""
Magnetic Tweezers Force Analysis - INTERACTIVE VERSION
Calculates forces on magnetic beads from ImageJ manual tracking data
Features: File selection, outlier removal, comprehensive analysis
Author: Abhinav's Magnetic Tweezers Lab
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import sys

# ============================================
# FILE SELECTION
# ============================================

print("="*70)
print("MAGNETIC TWEEZERS FORCE ANALYSIS - ImageJ Manual Tracking")
print("="*70)

# Create file dialog
root = tk.Tk()
root.withdraw()  # Hide the main window

print("\nPlease select your ImageJ tracking CSV file (Calibration2.csv or similar)...")
data_file = filedialog.askopenfilename(
    title="Select ImageJ Tracking CSV file",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if not data_file:
    print("No file selected. Exiting...")
    sys.exit(0)

print(f"✓ Selected: {Path(data_file).name}")

# Set output directory to same as input file
output_dir = Path(data_file).parent

# ============================================
# PARAMETERS
# ============================================

# Physical parameters
PARTICLE_RADIUS = 1.4e-6  # metres (1.4 microns radius = 2.8 microns diameter)
GLYCEROL_VISCOSITY = 1.412  # Pa·s at 20°C
FRAME_RATE = 7.76  # fps (Hz)

# CRITICAL: Pixel to micron conversion
PIXEL_SIZE_MICRONS = 0.11  # microns per pixel (from microscope calibration)

# Stokes drag coefficient: F = 6πηrv
DRAG_COEFFICIENT = 6 * np.pi * GLYCEROL_VISCOSITY * PARTICLE_RADIUS

print(f"\nPhysical Parameters:")
print(f"  Particle radius: {PARTICLE_RADIUS*1e6:.2f} μm")
print(f"  Glycerol viscosity: {GLYCEROL_VISCOSITY:.3f} Pa·s")
print(f"  Drag coefficient: {DRAG_COEFFICIENT:.3e} N·s/m")
print(f"  Frame rate: {FRAME_RATE:.2f} fps")
print(f"  Pixel size: {PIXEL_SIZE_MICRONS:.3f} μm/pixel")

# ============================================
# LOAD AND PROCESS ImageJ DATA
# ============================================

print(f"\n{'='*70}")
print("LOADING DATA")
print("="*70)

try:
    # Read ImageJ CSV (handle encoding issues with special characters)
    df_raw = pd.read_csv(data_file, encoding='latin-1')
    
    # Clean column names (ImageJ uses special degree symbol)
    df_raw.columns = ['Track_ID', 'Frame', 'X_pixels', 'Y_pixels', 
                      'Distance_ImageJ', 'Velocity_ImageJ', 'Pixel_Value']
    
    print(f"✓ Loaded {len(df_raw)} tracking points")
    print(f"  Frames: {df_raw['Frame'].min()} to {df_raw['Frame'].max()}")
    print(f"  X range: {df_raw['X_pixels'].min():.1f} to {df_raw['X_pixels'].max():.1f} pixels")
    print(f"  Y range: {df_raw['Y_pixels'].min():.1f} to {df_raw['Y_pixels'].max():.1f} pixels")
    
except Exception as e:
    print(f"ERROR loading file: {e}")
    print("\nExpected CSV format:")
    print("Track n°,Slice n°,X,Y,Distance,Velocity,Pixel Value")
    sys.exit(1)

# ============================================
# CALCULATE DISPLACEMENTS AND VELOCITIES
# ============================================

print(f"\n{'='*70}")
print("CALCULATING DISPLACEMENTS")
print("="*70)

# Time between frames
dt = 1.0 / FRAME_RATE  # seconds
df_raw['Time_sec'] = (df_raw['Frame'] - df_raw['Frame'].min()) * dt

# Calculate frame-to-frame displacements in pixels
df_raw['dX_pixels'] = df_raw['X_pixels'].diff()
df_raw['dY_pixels'] = df_raw['Y_pixels'].diff()

# Convert to microns
df_raw['dX_microns'] = df_raw['dX_pixels'] * PIXEL_SIZE_MICRONS
df_raw['dY_microns'] = df_raw['dY_pixels'] * PIXEL_SIZE_MICRONS

# Calculate 2D displacement magnitude
df_raw['Displacement_2D_microns'] = np.sqrt(df_raw['dX_microns']**2 + 
                                             df_raw['dY_microns']**2)

# Calculate velocities (microns/sec)
df_raw['Velocity_2D_um_per_sec'] = df_raw['Displacement_2D_microns'] / dt
df_raw['Velocity_X_um_per_sec'] = df_raw['dX_microns'] / dt
df_raw['Velocity_Y_um_per_sec'] = df_raw['dY_microns'] / dt

# Calculate cumulative displacement from start position
start_x = df_raw['X_pixels'].iloc[0]
start_y = df_raw['Y_pixels'].iloc[0]
df_raw['Cumulative_dX_pixels'] = df_raw['X_pixels'] - start_x
df_raw['Cumulative_dY_pixels'] = df_raw['Y_pixels'] - start_y
df_raw['Cumulative_Distance_pixels'] = np.sqrt(df_raw['Cumulative_dX_pixels']**2 + 
                                                df_raw['Cumulative_dY_pixels']**2)
df_raw['Cumulative_Distance_microns'] = df_raw['Cumulative_Distance_pixels'] * PIXEL_SIZE_MICRONS

print(f"✓ Calculated displacements and velocities")
print(f"  Total distance travelled: {df_raw['Cumulative_Distance_microns'].iloc[-1]:.2f} μm")
print(f"  Net displacement in X: {df_raw['Cumulative_dX_pixels'].iloc[-1]:.1f} pixels "
      f"({df_raw['Cumulative_dX_pixels'].iloc[-1] * PIXEL_SIZE_MICRONS:.2f} μm)")
print(f"  Net displacement in Y: {df_raw['Cumulative_dY_pixels'].iloc[-1]:.1f} pixels "
      f"({df_raw['Cumulative_dY_pixels'].iloc[-1] * PIXEL_SIZE_MICRONS:.2f} μm)")

# ============================================
# CALCULATE FORCES
# ============================================

print(f"\n{'='*70}")
print("CALCULATING FORCES")
print("="*70)
print("Using Stokes drag law: F = 6πηrv")

# Convert velocities to m/s
df_raw['Velocity_2D_m_per_sec'] = df_raw['Velocity_2D_um_per_sec'] * 1e-6
df_raw['Velocity_X_m_per_sec'] = df_raw['Velocity_X_um_per_sec'] * 1e-6
df_raw['Velocity_Y_m_per_sec'] = df_raw['Velocity_Y_um_per_sec'] * 1e-6

# Calculate forces using Stokes drag: F = 6πηrv
df_raw['Force_2D_N'] = DRAG_COEFFICIENT * df_raw['Velocity_2D_m_per_sec']
df_raw['Force_X_N'] = DRAG_COEFFICIENT * df_raw['Velocity_X_m_per_sec']
df_raw['Force_Y_N'] = DRAG_COEFFICIENT * df_raw['Velocity_Y_m_per_sec']

# Convert to picoNewtons (pN)
df_raw['Force_2D_pN'] = df_raw['Force_2D_N'] * 1e12
df_raw['Force_X_pN'] = df_raw['Force_X_N'] * 1e12
df_raw['Force_Y_pN'] = df_raw['Force_Y_N'] * 1e12

# Remove first row (has NaN due to diff operation)
df = df_raw.dropna().copy()

print(f"✓ Force calculations complete")
print(f"  Valid data points: {len(df)}")
print(f"  Peak instantaneous velocity: {df['Velocity_2D_um_per_sec'].max():.2f} μm/s")
print(f"  Peak instantaneous force: {df['Force_2D_pN'].max():.2f} pN")

# ============================================
# OUTLIER REMOVAL OPTIONS
# ============================================

print("\n" + "="*70)
print("OUTLIER REMOVAL")
print("="*70)
print("\nLarge force spikes can occur due to:")
print("  - Detection errors")
print("  - Sudden jumps between frames")
print("  - Edge effects")
print("  - Final acceleration near magnetic tip")

# Calculate statistics for outlier detection
Q1 = df['Force_2D_pN'].quantile(0.25)
Q3 = df['Force_2D_pN'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 3 * IQR
lower_bound = Q1 - 3 * IQR

outliers = df[(df['Force_2D_pN'] > upper_bound) | (df['Force_2D_pN'] < lower_bound)]
n_outliers = len(outliers)

print(f"\nAutomatic outlier detection (IQR method, 3×IQR):")
print(f"  Q1 = {Q1:.2f} pN, Q3 = {Q3:.2f} pN, IQR = {IQR:.2f} pN")
print(f"  Upper threshold = {upper_bound:.2f} pN")
print(f"  Detected {n_outliers} potential outliers ({n_outliers/len(df)*100:.1f}%)")

if n_outliers > 0:
    print(f"  Outlier frames: {outliers['Frame'].tolist()}")
    print(f"  Outlier forces: {[f'{f:.2f} pN' for f in outliers['Force_2D_pN'].tolist()]}")

print("\nOptions:")
print("  1. Keep ALL data points")
print("  2. Remove outliers automatically (recommended)")
print("  3. Manual force threshold")

outlier_choice = input("\nEnter your choice (1, 2, or 3): ").strip()

# Store original data
df_original = df.copy()

if outlier_choice == "2":
    print(f"\nRemoving {n_outliers} outliers...")
    df = df[(df['Force_2D_pN'] <= upper_bound) & (df['Force_2D_pN'] >= lower_bound)]
    print(f"✓ Filtered data: {len(df)} points remaining")
    
elif outlier_choice == "3":
    print("\nEnter maximum force threshold in pN")
    print(f"(Current max: {df['Force_2D_pN'].max():.1f} pN)")
    try:
        threshold = float(input("Max force (pN): "))
        n_removed = (df['Force_2D_pN'] > threshold).sum()
        df = df[df['Force_2D_pN'] <= threshold]
        print(f"✓ Removed {n_removed} points above {threshold} pN")
    except:
        print("Invalid input. Keeping all data.")
else:
    print("✓ Keeping all data points")

# ============================================
# STATISTICS
# ============================================

print(f"\n{'='*70}")
print("FORCE STATISTICS")
print("="*70)

# Calculate average velocity over entire trajectory
total_time = df_original['Time_sec'].iloc[-1] - df_original['Time_sec'].iloc[0]
total_distance = df_original['Cumulative_Distance_microns'].iloc[-1]
avg_velocity = total_distance / total_time
steady_force = DRAG_COEFFICIENT * avg_velocity * 1e-6 * 1e12

print(f"\nTrajectory statistics:")
print(f"  Duration: {total_time:.2f} seconds")
print(f"  Total distance: {total_distance:.2f} μm")
print(f"  Average velocity: {avg_velocity:.2f} μm/s")
print(f"  Corresponding steady force: {steady_force:.2f} pN")

print(f"\nAll data (n={len(df_original)}):")
print(f"  2D Force (pN):")
print(f"    Mean:   {df_original['Force_2D_pN'].mean():.3f} ± {df_original['Force_2D_pN'].std():.3f}")
print(f"    Median: {df_original['Force_2D_pN'].median():.3f}")
print(f"    Range:  {df_original['Force_2D_pN'].min():.3f} to {df_original['Force_2D_pN'].max():.3f}")
print(f"  X Force (pN): {df_original['Force_X_pN'].mean():.3f} ± {df_original['Force_X_pN'].std():.3f}")
print(f"  Y Force (pN): {df_original['Force_Y_pN'].mean():.3f} ± {df_original['Force_Y_pN'].std():.3f}")

if len(df) != len(df_original):
    print(f"\nFiltered data (outliers removed, n={len(df)}):")
    print(f"  2D Force (pN):")
    print(f"    Mean:   {df['Force_2D_pN'].mean():.3f} ± {df['Force_2D_pN'].std():.3f}")
    print(f"    Median: {df['Force_2D_pN'].median():.3f}")
    print(f"    Range:  {df['Force_2D_pN'].min():.3f} to {df['Force_2D_pN'].max():.3f}")

print("\nContext (typical biological forces):")
print("  Single molecular motor: ~1-10 pN")
print("  DNA unzipping: ~10-20 pN")
print("  Cell adhesion: ~50-100 pN")
print("  Optical tweezers range: ~0.1-100 pN")
print("  Magnetic tweezers range: ~0.1-1000 pN")

# ============================================
# SAVE RESULTS
# ============================================

print(f"\n{'='*70}")
print("SAVING RESULTS")
print("="*70)

# Save detailed data (all points)
output_file = output_dir / "force_analysis_detailed.csv"
df_original.to_csv(output_file, index=False)
print(f"\n✓ Saved detailed analysis (all data) to:")
print(f"  {output_file}")

# Save filtered data if different
if len(df) != len(df_original):
    filtered_file = output_dir / "force_analysis_filtered.csv"
    df.to_csv(filtered_file, index=False)
    print(f"✓ Saved filtered data (outliers removed) to:")
    print(f"  {filtered_file}")

# Create summary
summary_data = {
    'Parameter': [
        'Total_Frames',
        'Duration_sec',
        'Total_Distance_microns',
        'Net_Displacement_X_microns',
        'Net_Displacement_Y_microns',
        'Average_Velocity_um_per_sec',
        'Steady_State_Force_pN',
        'Mean_Force_2D_pN',
        'Median_Force_2D_pN',
        'Std_Force_2D_pN',
        'Max_Force_2D_pN',
        'Min_Force_2D_pN',
        'Mean_Force_X_pN',
        'Mean_Force_Y_pN',
        'Outliers_Detected',
        'Data_Points_Used',
    ],
    'Value': [
        len(df_original),
        total_time,
        total_distance,
        df_original['Cumulative_dX_pixels'].iloc[-1] * PIXEL_SIZE_MICRONS,
        df_original['Cumulative_dY_pixels'].iloc[-1] * PIXEL_SIZE_MICRONS,
        avg_velocity,
        steady_force,
        df_original['Force_2D_pN'].mean(),
        df_original['Force_2D_pN'].median(),
        df_original['Force_2D_pN'].std(),
        df_original['Force_2D_pN'].max(),
        df_original['Force_2D_pN'].min(),
        df_original['Force_X_pN'].mean(),
        df_original['Force_Y_pN'].mean(),
        n_outliers,
        len(df),
    ]
}

if len(df) != len(df_original):
    summary_data['Parameter'].extend([
        'Filtered_Mean_Force_pN',
        'Filtered_Median_Force_pN',
        'Filtered_Std_Force_pN'
    ])
    summary_data['Value'].extend([
        df['Force_2D_pN'].mean(),
        df['Force_2D_pN'].median(),
        df['Force_2D_pN'].std()
    ])

summary_df = pd.DataFrame(summary_data)
summary_file = output_dir / "force_analysis_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"✓ Saved summary to:")
print(f"  {summary_file}")

# ============================================
# PLOTTING
# ============================================

print(f"\n{'='*70}")
print("GENERATING PLOTS")
print("="*70)

fig = plt.figure(figsize=(16, 12))

# Plot 1: Bead trajectory
ax1 = plt.subplot(3, 3, 1)
ax1.plot(df_original['X_pixels'], df_original['Y_pixels'], 'o-', color='steelblue', 
         markersize=6, linewidth=2, alpha=0.7)
ax1.plot(df_original['X_pixels'].iloc[0], df_original['Y_pixels'].iloc[0], 'go', 
         markersize=12, label='Start', zorder=5)
ax1.plot(df_original['X_pixels'].iloc[-1], df_original['Y_pixels'].iloc[-1], 'ro', 
         markersize=12, label='End', zorder=5)
ax1.set_xlabel('X Position (pixels)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Y Position (pixels)', fontsize=11, fontweight='bold')
ax1.set_title('Bead Trajectory', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.invert_yaxis()  # ImageJ has Y increasing downwards

# Plot 2: Position vs Time
ax2 = plt.subplot(3, 3, 2)
ax2.plot(df_original['Time_sec'], df_original['X_pixels'], 'r-', label='X', linewidth=2, alpha=0.7)
ax2.plot(df_original['Time_sec'], df_original['Y_pixels'], 'b-', label='Y', linewidth=2, alpha=0.7)
ax2.set_xlabel('Time (sec)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Position (pixels)', fontsize=11, fontweight='bold')
ax2.set_title('Position vs Time', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Cumulative distance
ax3 = plt.subplot(3, 3, 3)
ax3.plot(df_original['Time_sec'], df_original['Cumulative_Distance_microns'], 
         'o-', color='purple', markersize=4, linewidth=2, alpha=0.7)
ax3.set_xlabel('Time (sec)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Cumulative Distance (μm)', fontsize=11, fontweight='bold')
ax3.set_title('Total Distance Travelled', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Instantaneous velocity vs time
ax4 = plt.subplot(3, 3, 4)
ax4.plot(df_original['Time_sec'], df_original['Velocity_2D_um_per_sec'], 
         'o-', color='darkgreen', markersize=4, linewidth=2, alpha=0.7)
ax4.axhline(avg_velocity, color='red', linestyle='--', linewidth=2, 
            label=f'Average: {avg_velocity:.2f} μm/s')
ax4.set_xlabel('Time (sec)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Velocity (μm/s)', fontsize=11, fontweight='bold')
ax4.set_title('Instantaneous Velocity vs Time', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Force vs Time (2D)
ax5 = plt.subplot(3, 3, 5)
ax5.plot(df_original['Time_sec'], df_original['Force_2D_pN'], 
         'o-', color='steelblue', markersize=4, linewidth=2, alpha=0.7)
if n_outliers > 0 and outlier_choice == "2":
    ax5.plot(outliers['Time_sec'], outliers['Force_2D_pN'], 
             'ro', markersize=8, label='Outliers (removed)', zorder=5)
ax5.axhline(df_original['Force_2D_pN'].mean(), color='orange', linestyle='--', 
            linewidth=2, label=f'Mean: {df_original["Force_2D_pN"].mean():.2f} pN')
ax5.axhline(steady_force, color='green', linestyle='--', 
            linewidth=2, label=f'Steady: {steady_force:.2f} pN')
ax5.set_xlabel('Time (sec)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Force (pN)', fontsize=11, fontweight='bold')
ax5.set_title('2D Force vs Time', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Plot 6: Force components
ax6 = plt.subplot(3, 3, 6)
ax6.plot(df_original['Time_sec'], df_original['Force_X_pN'], 
         'r-', label='Fx', linewidth=2, alpha=0.7)
ax6.plot(df_original['Time_sec'], df_original['Force_Y_pN'], 
         'b-', label='Fy', linewidth=2, alpha=0.7)
ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax6.set_xlabel('Time (sec)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Force (pN)', fontsize=11, fontweight='bold')
ax6.set_title('Force Components vs Time', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Force histogram (all data)
ax7 = plt.subplot(3, 3, 7)
ax7.hist(df_original['Force_2D_pN'], bins=30, color='steelblue', 
         alpha=0.7, edgecolor='black')
ax7.axvline(df_original['Force_2D_pN'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {df_original["Force_2D_pN"].mean():.2f} pN')
ax7.axvline(df_original['Force_2D_pN'].median(), color='orange', linestyle='--', 
            linewidth=2, label=f'Median: {df_original["Force_2D_pN"].median():.2f} pN')
ax7.set_xlabel('Force (pN)', fontsize=11, fontweight='bold')
ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax7.set_title('Force Distribution (All Data)', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3, axis='y')

# Plot 8: Force histogram (filtered) or Velocity components
if len(df) != len(df_original):
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(df['Force_2D_pN'], bins=30, color='darkgreen', 
             alpha=0.7, edgecolor='black')
    ax8.axvline(df['Force_2D_pN'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {df["Force_2D_pN"].mean():.2f} pN')
    ax8.axvline(df['Force_2D_pN'].median(), color='orange', linestyle='--', 
                linewidth=2, label=f'Median: {df["Force_2D_pN"].median():.2f} pN')
    ax8.set_xlabel('Force (pN)', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax8.set_title('Force Distribution (Filtered)', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')
else:
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(df_original['Time_sec'], df_original['Velocity_X_um_per_sec'], 
             'r-', label='Vx', linewidth=2, alpha=0.7)
    ax8.plot(df_original['Time_sec'], df_original['Velocity_Y_um_per_sec'], 
             'b-', label='Vy', linewidth=2, alpha=0.7)
    ax8.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax8.set_xlabel('Time (sec)', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Velocity (μm/s)', fontsize=11, fontweight='bold')
    ax8.set_title('Velocity Components vs Time', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

# Plot 9: Force vs Distance
ax9 = plt.subplot(3, 3, 9)
ax9.plot(df_original['Cumulative_Distance_microns'], df_original['Force_2D_pN'], 
         'o-', color='purple', markersize=4, linewidth=2, alpha=0.7)
ax9.set_xlabel('Distance from Start (μm)', fontsize=11, fontweight='bold')
ax9.set_ylabel('Force (pN)', fontsize=11, fontweight='bold')
ax9.set_title('Force vs Distance Travelled', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3)

plt.suptitle('Magnetic Tweezers Force Analysis - ImageJ Manual Tracking', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

plot_file = output_dir / "force_analysis_plots.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plots to:")
print(f"  {plot_file}")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nKey Results:")
print(f"  Mean force: {df_original['Force_2D_pN'].mean():.2f} ± {df_original['Force_2D_pN'].std():.2f} pN")
print(f"  Median force: {df_original['Force_2D_pN'].median():.2f} pN")
print(f"  Steady-state force: {steady_force:.2f} pN")
print(f"  Average velocity: {avg_velocity:.2f} μm/s")
print(f"  Total distance: {total_distance:.2f} μm")
print(f"\nAll files saved to: {output_dir}")

plt.show()
