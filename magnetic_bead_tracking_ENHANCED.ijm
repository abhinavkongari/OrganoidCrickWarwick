// Magnetic Bead Tracking Macro - ENHANCED VERSION
// Filters stationary beads and handles gaps in tracking
// Author: Abhinav's Magnetic Tweezers Lab
// Date: November 2025

// ============================================
// PARAMETERS
// ============================================

z_step = 2;
particle_radius = 0.5;
glycerol_viscosity = 1.412;
frame_rate = 7.76;

min_particle_size = 50;
max_particle_size = 500;
circularity_min = 0.7;

max_displacement = 200;
min_track_length = 3;
min_total_displacement = 50; // NEW: Minimum total movement in pixels to keep track
max_gap_frames = 3; // NEW: Allow up to 3 frames of missing detection

// ============================================
// STEP 1: SELECT FILES
// ============================================

print("\\Clear");
print("=== Magnetic Bead Tracking - ENHANCED VERSION ===");
print("\nStep 1: File Selection\n");

lookup_path = File.openDialog("Select the Z-calibration stack (stackofbeads.tif)");
if (!File.exists(lookup_path)) {
    exit("ERROR: Lookup table file not found!");
}

open(lookup_path);
lookup_id = getImageID();
lookup_title = getTitle();
print("✓ Loaded Z-lookup table: " + lookup_title);

Stack.getDimensions(lut_width, lut_height, lut_channels, slices_lookup, lut_frames);
print("  Z-slices: " + slices_lookup);

print("\nPrecomputing Z-calibration profiles...");
setBatchMode(true);
lookup_variances = newArray(slices_lookup);

for (z = 1; z <= slices_lookup; z++) {
    selectImage(lookup_id);
    Stack.setSlice(z);
    makeRectangle(lut_width/2 - 20, lut_height/2 - 20, 40, 40);
    getStatistics(area, mean, min, max, std);
    lookup_variances[z-1] = std * std;
}
selectImage(lookup_id);
run("Select None");
setBatchMode(false);
print("✓ Calibration complete");
rename("Z_Lookup_Table");

print("\nSelect the time-lapse movie...");
wait(500);
timelapse_path = File.openDialog("Select the time-lapse movie");
if (!File.exists(timelapse_path)) {
    exit("ERROR: Time-lapse file not found!");
}

open(timelapse_path);
main_id = getImageID();
main_title = getTitle();
print("✓ Loaded time-lapse: " + main_title);

Stack.getDimensions(width, height, channels, slices, frames);
print("  Initial - Slices: " + slices + ", Frames: " + frames);

if (frames == 1 && slices > 1) {
    print("  Converting Z-stack to time series...");
    temp_slices = slices;
    run("Properties...", "channels=1 slices=1 frames=" + temp_slices + " unit=pixel pixel_width=1 pixel_height=1 voxel_depth=1");
    Stack.getDimensions(width, height, channels, slices, frames);
    print("  ✓ Converted - Frames: " + frames);
}

getPixelSize(unit, pixelWidth, pixelHeight);
print("  Pixel size: " + pixelWidth + " " + unit);
print("  Total frames: " + frames);

// ============================================
// STEP 2: PARAMETERS
// ============================================

Dialog.create("Enhanced Tracking Parameters");
Dialog.addNumber("Frame rate (fps):", frame_rate);
Dialog.addNumber("Max displacement (pixels):", max_displacement);
Dialog.addNumber("Max gap frames (allow missing detections):", max_gap_frames);
Dialog.addNumber("Min track length (frames):", min_track_length);
Dialog.addNumber("Min total displacement (pixels, filters stationary):", min_total_displacement);
Dialog.show();

frame_rate = Dialog.getNumber();
max_displacement = Dialog.getNumber();
max_gap_frames = Dialog.getNumber();
min_track_length = Dialog.getNumber();
min_total_displacement = Dialog.getNumber();

print("\nParameters:");
print("  Frame rate: " + frame_rate + " fps");
print("  Max displacement: " + max_displacement + " px");
print("  Max gap frames: " + max_gap_frames);
print("  Min track length: " + min_track_length + " frames");
print("  Min total displacement: " + min_total_displacement + " px");

// ============================================
// STEP 3: DETECT PARTICLES - VERY LENIENT
// ============================================

print("\nStep 3: Detecting particles (lenient for tracking gaps)...\n");

setBatchMode(true);

all_detections_frame = newArray(frames * 50);
all_detections_x = newArray(frames * 50);
all_detections_y = newArray(frames * 50);
all_detections_area = newArray(frames * 50);
detection_count = 0;

for (f = 1; f <= frames; f++) {
    selectImage(main_id);
    Stack.setFrame(f);
    
    run("Duplicate...", "title=temp");
    temp_id = getImageID();
    
    // Even more aggressive processing for difficult-to-detect beads
    run("Enhance Contrast", "saturated=0.6");
    run("Gaussian Blur...", "sigma=1.5");
    run("Subtract Background...", "rolling=25");
    setAutoThreshold("Moments dark");
    run("Convert to Mask");
    
    // VERY lenient detection
    actual_min_size = min_particle_size / 3;  // Even smaller
    actual_max_size = max_particle_size * 3;  // Even larger
    actual_circularity = 0.3;  // Very lenient circularity
    
    run("Set Measurements...", "area mean centroid center redirect=None decimal=3");
    run("Analyze Particles...", "size=" + actual_min_size + "-" + actual_max_size + 
        " circularity=" + actual_circularity + "-1.00 show=Nothing display clear");
    
    n_particles = nResults;
    
    for (i = 0; i < n_particles; i++) {
        x_coord = getResult("XM", i);
        y_coord = getResult("YM", i);
        
        if (isNaN(x_coord)) x_coord = getResult("X", i);
        if (isNaN(y_coord)) y_coord = getResult("Y", i);
        
        if (isNaN(x_coord)) {
            bx = getResult("BX", i);
            bwidth = getResult("Width", i);
            x_coord = bx + bwidth/2;
        }
        if (isNaN(y_coord)) {
            by = getResult("BY", i);
            bheight = getResult("Height", i);
            y_coord = by + bheight/2;
        }
        
        area_val = getResult("Area", i);
        
        if (!isNaN(x_coord) && !isNaN(y_coord)) {
            all_detections_frame[detection_count] = f;
            all_detections_x[detection_count] = x_coord;
            all_detections_y[detection_count] = y_coord;
            all_detections_area[detection_count] = area_val;
            detection_count++;
        }
    }
    
    selectImage(temp_id);
    close();
    
    if (f % 10 == 0 || f == frames) {
        print("  Frame " + f + " / " + frames + ": detected " + n_particles + " beads");
    }
}

setBatchMode(false);

all_detections_frame = Array.trim(all_detections_frame, detection_count);
all_detections_x = Array.trim(all_detections_x, detection_count);
all_detections_y = Array.trim(all_detections_y, detection_count);
all_detections_area = Array.trim(all_detections_area, detection_count);

print("\n✓ Detection complete: " + detection_count + " total detections");

if (detection_count == 0) {
    exit("ERROR: No valid particles detected!");
}

// ============================================
// STEP 4: LINK WITH GAP CLOSING
// ============================================

print("\nStep 4: Linking detections (with gap closing)...\n");

max_tracks = 100;
tracks_detections = newArray(max_tracks);
tracks_length = newArray(max_tracks);
tracks_start_frame = newArray(max_tracks);
tracks_end_frame = newArray(max_tracks);
tracks_total_displacement = newArray(max_tracks);
num_tracks = 0;

detection_assigned = newArray(detection_count);
for (i = 0; i < detection_count; i++) {
    detection_assigned[i] = 0;
}

// Track from frame 1
for (start_det = 0; start_det < detection_count; start_det++) {
    if (detection_assigned[start_det] == 1) continue;
    if (all_detections_frame[start_det] != 1) continue;
    
    track_indices = "" + start_det;
    current_det = start_det;
    detection_assigned[start_det] = 1;
    track_frame = all_detections_frame[start_det];
    
    // Track forward with gap closing
    gap_count = 0;
    
    for (next_frame = track_frame + 1; next_frame <= frames; next_frame++) {
        best_distance = 999999;
        best_det = -1;
        
        for (det = 0; det < detection_count; det++) {
            if (all_detections_frame[det] != next_frame) continue;
            if (detection_assigned[det] == 1) continue;
            
            dx = all_detections_x[det] - all_detections_x[current_det];
            dy = all_detections_y[det] - all_detections_y[current_det];
            distance = sqrt(dx*dx + dy*dy);
            
            // Allow larger displacement for gap frames
            max_allowed = max_displacement * (1 + gap_count);
            
            if (distance < best_distance && distance < max_allowed) {
                best_distance = distance;
                best_det = det;
            }
        }
        
        if (best_det >= 0) {
            track_indices = track_indices + "," + best_det;
            detection_assigned[best_det] = 1;
            current_det = best_det;
            gap_count = 0; // Reset gap counter
        } else {
            // No detection found - allow gap
            gap_count++;
            if (gap_count > max_gap_frames) {
                break; // Too many consecutive gaps
            }
            // Continue to next frame
        }
    }
    
    indices_array = split(track_indices, ",");
    track_length = indices_array.length;
    
    // Calculate total displacement for this track
    if (track_length >= 2) {
        first_det = parseInt(indices_array[0]);
        last_det = parseInt(indices_array[track_length - 1]);
        
        dx_total = all_detections_x[last_det] - all_detections_x[first_det];
        dy_total = all_detections_y[last_det] - all_detections_y[first_det];
        total_disp = sqrt(dx_total*dx_total + dy_total*dy_total);
    } else {
        total_disp = 0;
    }
    
    // Keep only tracks that move enough AND are long enough
    if (track_length >= min_track_length && total_disp >= min_total_displacement) {
        tracks_detections[num_tracks] = track_indices;
        tracks_length[num_tracks] = track_length;
        tracks_start_frame[num_tracks] = all_detections_frame[start_det];
        tracks_end_frame[num_tracks] = all_detections_frame[current_det];
        tracks_total_displacement[num_tracks] = total_disp;
        num_tracks++;
        print("  Track " + num_tracks + ": " + track_length + " frames, displacement=" + d2s(total_disp, 1) + " px");
    } else {
        if (total_disp < min_total_displacement) {
            print("  Rejected stationary track: " + d2s(total_disp, 1) + " px movement (need " + min_total_displacement + ")");
        }
    }
}

if (num_tracks == 0) {
    print("\n✗ No moving tracks found!");
    print("  Try reducing min_total_displacement parameter");
    exit("No tracks created.");
}

tracks_detections = Array.trim(tracks_detections, num_tracks);
tracks_length = Array.trim(tracks_length, num_tracks);
tracks_start_frame = Array.trim(tracks_start_frame, num_tracks);
tracks_end_frame = Array.trim(tracks_end_frame, num_tracks);
tracks_total_displacement = Array.trim(tracks_total_displacement, num_tracks);

// Find min/max manually
min_length = tracks_length[0];
max_length = tracks_length[0];
for (t = 0; t < num_tracks; t++) {
    if (tracks_length[t] < min_length) min_length = tracks_length[t];
    if (tracks_length[t] > max_length) max_length = tracks_length[t];
}

print("\n✓ Created " + num_tracks + " moving tracks");
print("  Track lengths: " + min_length + " to " + max_length + " frames");

// ============================================
// STEP 5: ESTIMATE Z-POSITIONS
// ============================================

print("\nStep 5: Estimating Z-positions...\n");

all_detections_z = newArray(detection_count);

selectImage(main_id);
setBatchMode(true);

for (det = 0; det < detection_count; det++) {
    frame = all_detections_frame[det];
    x = all_detections_x[det];
    y = all_detections_y[det];
    
    Stack.setFrame(frame);
    
    roi_x = x - 10;
    roi_y = y - 10;
    roi_size = 20;
    
    if (roi_x < 0) roi_x = 0;
    if (roi_y < 0) roi_y = 0;
    if (roi_x + roi_size > width) roi_x = width - roi_size;
    if (roi_y + roi_size > height) roi_y = height - roi_size;
    
    makeRectangle(roi_x, roi_y, roi_size, roi_size);
    getStatistics(area, mean, min, max, std);
    local_variance = std * std;
    
    min_diff = 999999;
    best_z_index = 0;
    
    for (z = 0; z < slices_lookup; z++) {
        diff = abs(local_variance - lookup_variances[z]);
        if (diff < min_diff) {
            min_diff = diff;
            best_z_index = z;
        }
    }
    
    all_detections_z[det] = best_z_index * z_step;
    
    if ((det+1) % 100 == 0 || det == detection_count - 1) {
        print("  Z-estimated: " + (det+1) + " / " + detection_count);
    }
}

run("Select None");
setBatchMode(false);
print("✓ Z-estimation complete");

// ============================================
// STEP 6: EXPORT DATA
// ============================================

print("\nStep 6: Exporting data...\n");

// Position data
Table.create("Bead_Tracking_Results");
row_count = 0;

for (track_id = 0; track_id < num_tracks; track_id++) {
    indices_str = tracks_detections[track_id];
    indices = split(indices_str, ",");
    
    for (i = 0; i < indices.length; i++) {
        det_idx = parseInt(indices[i]);
        
        frame = all_detections_frame[det_idx];
        time_sec = (frame - 1) / frame_rate;
        x_microns = all_detections_x[det_idx] * pixelWidth;
        y_microns = all_detections_y[det_idx] * pixelHeight;
        z_microns = all_detections_z[det_idx];
        area = all_detections_area[det_idx];
        
        Table.set("Frame", row_count, frame, "Bead_Tracking_Results");
        Table.set("Time_sec", row_count, time_sec, "Bead_Tracking_Results");
        Table.set("Bead_ID", row_count, track_id, "Bead_Tracking_Results");
        Table.set("X_microns", row_count, x_microns, "Bead_Tracking_Results");
        Table.set("Y_microns", row_count, y_microns, "Bead_Tracking_Results");
        Table.set("Z_microns", row_count, z_microns, "Bead_Tracking_Results");
        Table.set("Area_pixels", row_count, area, "Bead_Tracking_Results");
        Table.set("Track_Length", row_count, tracks_length[track_id], "Bead_Tracking_Results");
        Table.set("Total_Displacement", row_count, tracks_total_displacement[track_id], "Bead_Tracking_Results");
        row_count++;
    }
}

Table.update("Bead_Tracking_Results");
print("✓ Position table: " + row_count + " entries");

// Displacement data
Table.create("Bead_Displacements");
disp_row = 0;

for (track_id = 0; track_id < num_tracks; track_id++) {
    indices_str = tracks_detections[track_id];
    indices = split(indices_str, ",");
    
    for (i = 1; i < indices.length; i++) {
        det_idx_prev = parseInt(indices[i-1]);
        det_idx_curr = parseInt(indices[i]);
        
        x_prev = all_detections_x[det_idx_prev] * pixelWidth;
        y_prev = all_detections_y[det_idx_prev] * pixelHeight;
        z_prev = all_detections_z[det_idx_prev];
        
        x_curr = all_detections_x[det_idx_curr] * pixelWidth;
        y_curr = all_detections_y[det_idx_curr] * pixelHeight;
        z_curr = all_detections_z[det_idx_curr];
        
        dx = x_curr - x_prev;
        dy = y_curr - y_prev;
        dz = z_curr - z_prev;
        
        displacement_2d = sqrt(dx*dx + dy*dy);
        displacement_3d = sqrt(dx*dx + dy*dy + dz*dz);
        
        frame_curr = all_detections_frame[det_idx_curr];
        frame_prev = all_detections_frame[det_idx_prev];
        time_sec = (frame_curr - 1) / frame_rate;
        
        dt = (frame_curr - frame_prev) / frame_rate;
        if (dt > 0) {
            velocity_2d = displacement_2d / dt;
            velocity_3d = displacement_3d / dt;
        } else {
            velocity_2d = 0;
            velocity_3d = 0;
        }
        
        Table.set("Frame", disp_row, frame_curr, "Bead_Displacements");
        Table.set("Time_sec", disp_row, time_sec, "Bead_Displacements");
        Table.set("Bead_ID", disp_row, track_id, "Bead_Displacements");
        Table.set("dX_microns", disp_row, dx, "Bead_Displacements");
        Table.set("dY_microns", disp_row, dy, "Bead_Displacements");
        Table.set("dZ_microns", disp_row, dz, "Bead_Displacements");
        Table.set("Displacement_2D_microns", disp_row, displacement_2d, "Bead_Displacements");
        Table.set("Displacement_3D_microns", disp_row, displacement_3d, "Bead_Displacements");
        Table.set("Velocity_2D_um_per_sec", disp_row, velocity_2d, "Bead_Displacements");
        Table.set("Velocity_3D_um_per_sec", disp_row, velocity_3d, "Bead_Displacements");
        disp_row++;
    }
}

Table.update("Bead_Displacements");
print("✓ Displacement table: " + disp_row + " entries");

// ============================================
// STEP 7: VISUALIZE
// ============================================

print("\nStep 7: Visualization...\n");

selectImage(main_id);
run("Select None");
Overlay.remove;

colors = newArray("red", "green", "blue", "yellow", "cyan", "magenta", "orange", "pink", "white");
num_show = num_tracks;
if (num_show > 9) num_show = 9;

for (track_id = 0; track_id < num_show; track_id++) {
    indices_str = tracks_detections[track_id];
    indices = split(indices_str, ",");
    
    setColor(colors[track_id]);
    setLineWidth(3);
    
    for (i = 0; i < indices.length - 1; i++) {
        det_idx1 = parseInt(indices[i]);
        det_idx2 = parseInt(indices[i+1]);
        
        x1 = all_detections_x[det_idx1];
        y1 = all_detections_y[det_idx1];
        x2 = all_detections_x[det_idx2];
        y2 = all_detections_y[det_idx2];
        
        Overlay.drawLine(x1, y1, x2, y2);
        
        if (i == 0) {
            Overlay.drawEllipse(x1 - 10, y1 - 10, 20, 20);
        }
    }
    
    // Mark end with X
    last_idx = parseInt(indices[indices.length - 1]);
    x_end = all_detections_x[last_idx];
    y_end = all_detections_y[last_idx];
    Overlay.drawLine(x_end - 8, y_end - 8, x_end + 8, y_end + 8);
    Overlay.drawLine(x_end - 8, y_end + 8, x_end + 8, y_end - 8);
}

Overlay.show;
print("✓ Overlay created");

// ============================================
// STEP 8: SAVE
// ============================================

print("\nStep 8: Saving...\n");

output_dir = getDirectory("Choose output directory");

Table.save(output_dir + "bead_tracking_data.csv", "Bead_Tracking_Results");
print("✓ Saved: bead_tracking_data.csv");

Table.save(output_dir + "bead_displacements.csv", "Bead_Displacements");
print("✓ Saved: bead_displacements.csv");

selectImage(main_id);
saveAs("Tiff", output_dir + "tracked_movie.tif");
print("✓ Saved: tracked_movie.tif");

// Save summary
summary_file = File.open(output_dir + "tracking_summary.txt");
print(summary_file, "=== MAGNETIC BEAD TRACKING SUMMARY ===\n");
print(summary_file, "Total tracks: " + num_tracks);
print(summary_file, "Track lengths: " + min_length + " to " + max_length + " frames\n");
print(summary_file, "Individual tracks:");
for (t = 0; t < num_tracks; t++) {
    print(summary_file, "  Track " + t + ": " + tracks_length[t] + " frames, " +
          d2s(tracks_total_displacement[t], 1) + " px displacement, " +
          "frames " + tracks_start_frame[t] + "-" + tracks_end_frame[t]);
}
File.close(summary_file);
print("✓ Saved: tracking_summary.txt");

// Find max manually
max_track_length = tracks_length[0];
for (t = 0; t < num_tracks; t++) {
    if (tracks_length[t] > max_track_length) max_track_length = tracks_length[t];
}

print("\n==================================================");
print("=== TRACKING COMPLETE ===");
print("==================================================");
print("\nResults:");
print("  " + num_tracks + " MOVING tracks (stationary filtered out)");
print("  Longest: " + max_track_length + " frames");
print("  Position data: " + row_count + " points");
print("  Displacement data: " + disp_row + " points");
print("\nFiles saved to: " + output_dir);
print("==================================================");

selectWindow("Bead_Tracking_Results");
selectWindow("Bead_Displacements");
