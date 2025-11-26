// DIAGNOSTIC VERSION - Magnetic Bead Tracking
// This version will help identify why tracks aren't being created

// ============================================
// PARAMETERS - ADJUST THESE FOR YOUR SETUP
// ============================================

// Z-calibration parameters
z_step = 2;
particle_radius = 0.5;
glycerol_viscosity = 1.412;
frame_rate = 7.76;

// Detection parameters
min_particle_size = 50;
max_particle_size = 500;
circularity_min = 0.7;

// Tracking parameters - MUCH MORE LENIENT
max_displacement = 200; // INCREASED from 100
min_track_length = 3; // REDUCED from 5

// ============================================
// STEP 1: SELECT FILES
// ============================================

print("\\Clear");
print("=== DIAGNOSTIC VERSION - Magnetic Bead Tracking ===");
print("\nStep 1: File Selection\n");

// Select the lookup table
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

// Precompute reference profiles
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

// Select the time-lapse movie
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

// Convert if needed
if (frames == 1 && slices > 1) {
    print("  Converting Z-stack to time series...");
    temp_slices = slices;
    run("Properties...", "channels=1 slices=1 frames=" + temp_slices + " unit=pixel pixel_width=1 pixel_height=1 voxel_depth=1");
    Stack.getDimensions(width, height, channels, slices, frames);
    print("  ✓ Converted - Frames: " + frames);
}

getPixelSize(unit, pixelWidth, pixelHeight);
print("  Pixel size: " + pixelWidth + " " + unit);

// ============================================
// SIMPLIFIED PARAMETER DIALOG
// ============================================

Dialog.create("Diagnostic Tracking Parameters");
Dialog.addMessage("=== Tracking Parameters (Lenient) ===");
Dialog.addNumber("Max displacement per frame (pixels):", max_displacement);
Dialog.addNumber("Min track length (frames):", min_track_length);
Dialog.addMessage("\nDiagnostic mode: Will show detailed tracking info");
Dialog.show();

max_displacement = Dialog.getNumber();
min_track_length = Dialog.getNumber();

print("\nDiagnostic Parameters:");
print("  Max displacement: " + max_displacement + " pixels");
print("  Min track length: " + min_track_length + " frames");

// ============================================
// STEP 3: DETECT PARTICLES
// ============================================

print("\nStep 3: Detecting particles...\n");

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
    
    run("Enhance Contrast", "saturated=0.5");
    run("Gaussian Blur...", "sigma=1");
    run("Subtract Background...", "rolling=30");
    setAutoThreshold("Moments dark");
    run("Convert to Mask");
    
    actual_min_size = min_particle_size / 2;
    actual_max_size = max_particle_size * 2;
    actual_circularity = circularity_min - 0.2;
    if (actual_circularity < 0.3) actual_circularity = 0.3;
    
    run("Analyze Particles...", "size=" + actual_min_size + "-" + actual_max_size + 
        " circularity=" + actual_circularity + "-1.00 show=Nothing display clear");
    
    n_particles = nResults;
    
    for (i = 0; i < n_particles; i++) {
        all_detections_frame[detection_count] = f;
        all_detections_x[detection_count] = getResult("X", i);
        all_detections_y[detection_count] = getResult("Y", i);
        all_detections_area[detection_count] = getResult("Area", i);
        detection_count++;
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
    exit("ERROR: No particles detected!");
}

// ============================================
// DIAGNOSTIC: Analyze frame-to-frame distances
// ============================================

print("\n=== DIAGNOSTIC: Frame-to-frame analysis ===\n");

// Sample first few frames to see typical distances
test_frames = 5;
if (test_frames > frames) test_frames = frames;

for (test_f = 1; test_f < test_frames; test_f++) {
    // Get detections in frame test_f
    frame1_x = newArray();
    frame1_y = newArray();
    count1 = 0;
    
    for (d = 0; d < detection_count; d++) {
        if (all_detections_frame[d] == test_f) {
            frame1_x = Array.concat(frame1_x, all_detections_x[d]);
            frame1_y = Array.concat(frame1_y, all_detections_y[d]);
            count1++;
        }
    }
    
    // Get detections in frame test_f+1
    frame2_x = newArray();
    frame2_y = newArray();
    count2 = 0;
    
    for (d = 0; d < detection_count; d++) {
        if (all_detections_frame[d] == (test_f + 1)) {
            frame2_x = Array.concat(frame2_x, all_detections_x[d]);
            frame2_y = Array.concat(frame2_y, all_detections_y[d]);
            count2++;
        }
    }
    
    print("Frame " + test_f + " → " + (test_f+1) + ":");
    print("  Beads: " + count1 + " → " + count2);
    
    if (count1 > 0 && count2 > 0) {
        // Find minimum distance between any bead in frame1 to any bead in frame2
        min_distance = 999999;
        max_distance = 0;
        total_distance = 0;
        match_count = 0;
        
        for (i = 0; i < count1; i++) {
            // Find closest bead in next frame
            closest = 999999;
            for (j = 0; j < count2; j++) {
                dx = frame2_x[j] - frame1_x[i];
                dy = frame2_y[j] - frame1_y[i];
                dist = sqrt(dx*dx + dy*dy);
                if (dist < closest) closest = dist;
            }
            
            if (closest < min_distance) min_distance = closest;
            if (closest > max_distance) max_distance = closest;
            total_distance += closest;
            match_count++;
        }
        
        avg_distance = total_distance / match_count;
        
        print("  Min displacement: " + d2s(min_distance, 1) + " px");
        print("  Avg displacement: " + d2s(avg_distance, 1) + " px");
        print("  Max displacement: " + d2s(max_distance, 1) + " px");
        
        if (min_distance > max_displacement) {
            print("  ⚠️ WARNING: Even the minimum displacement (" + d2s(min_distance, 1) + " px) exceeds max_displacement!");
        } else if (avg_distance > max_displacement) {
            print("  ⚠️ WARNING: Average displacement exceeds max_displacement!");
        }
    }
    print("");
}

print("RECOMMENDATION:");
print("  Based on the above, set max_displacement to at least: " + d2s(max_distance * 1.5, 0) + " pixels");
print("");

// ============================================
// STEP 4: LINK DETECTIONS (WITH DIAGNOSTICS)
// ============================================

print("Step 4: Linking detections into tracks...\n");

max_tracks = 100;
tracks_detections = newArray(max_tracks);
tracks_length = newArray(max_tracks);
tracks_start_frame = newArray(max_tracks);
tracks_end_frame = newArray(max_tracks);
num_tracks = 0;

detection_assigned = newArray(detection_count);
for (i = 0; i < detection_count; i++) {
    detection_assigned[i] = 0;
}

// Count beads in first frame
first_frame_count = 0;
for (d = 0; d < detection_count; d++) {
    if (all_detections_frame[d] == 1) first_frame_count++;
}
print("Beads detected in frame 1: " + first_frame_count);

// Try to track from EVERY detection in first frame (more lenient)
for (start_det = 0; start_det < detection_count; start_det++) {
    if (detection_assigned[start_det] == 1) continue;
    
    // Start from frame 1 only
    if (all_detections_frame[start_det] != 1) continue;
    
    track_indices = "" + start_det;
    current_det = start_det;
    detection_assigned[start_det] = 1;
    track_frame = all_detections_frame[start_det];
    
    // Track forward
    for (next_frame = track_frame + 1; next_frame <= frames; next_frame++) {
        best_distance = 999999;
        best_det = -1;
        
        for (det = 0; det < detection_count; det++) {
            if (all_detections_frame[det] != next_frame) continue;
            if (detection_assigned[det] == 1) continue;
            
            dx = all_detections_x[det] - all_detections_x[current_det];
            dy = all_detections_y[det] - all_detections_y[current_det];
            distance = sqrt(dx*dx + dy*dy);
            
            if (distance < best_distance && distance < max_displacement) {
                best_distance = distance;
                best_det = det;
            }
        }
        
        if (best_det >= 0) {
            track_indices = track_indices + "," + best_det;
            detection_assigned[best_det] = 1;
            current_det = best_det;
        } else {
            break;
        }
    }
    
    indices_array = split(track_indices, ",");
    track_length = indices_array.length;
    
    // Keep tracks that meet minimum length
    if (track_length >= min_track_length) {
        tracks_detections[num_tracks] = track_indices;
        tracks_length[num_tracks] = track_length;
        tracks_start_frame[num_tracks] = all_detections_frame[start_det];
        tracks_end_frame[num_tracks] = all_detections_frame[current_det];
        num_tracks++;
        print("  Track " + num_tracks + ": " + track_length + " frames");
    } else {
        print("  Rejected short track: " + track_length + " frames (need " + min_track_length + ")");
    }
}

if (num_tracks == 0) {
    print("\n✗ STILL NO TRACKS FOUND!");
    print("\nPossible issues:");
    print("  1. Beads moving too fast (>" + max_displacement + " pixels/frame)");
    print("  2. Beads appearing/disappearing (focus/detection issues)");
    print("  3. Too many beads - algorithm confused by crowding");
    print("\nTry these solutions:");
    print("  A. INCREASE max_displacement to " + d2s(max_distance * 2, 0) + " pixels");
    print("  B. DECREASE min_track_length to 2 frames");
    print("  C. Check if beads are consistently detected across frames");
    
    // Export detection data for manual inspection
    Table.create("All_Detections_Debug");
    for (d = 0; d < detection_count; d++) {
        Table.set("Detection_ID", d, d, "All_Detections_Debug");
        Table.set("Frame", d, all_detections_frame[d], "All_Detections_Debug");
        Table.set("X", d, all_detections_x[d], "All_Detections_Debug");
        Table.set("Y", d, all_detections_y[d], "All_Detections_Debug");
    }
    Table.update("All_Detections_Debug");
    
    output_dir = getDirectory("Choose directory to save debug data");
    Table.save(output_dir + "all_detections_debug.csv", "All_Detections_Debug");
    print("\n✓ Saved all_detections_debug.csv for inspection");
    
    exit("No tracks created. Check diagnostics above.");
}

// Trim arrays
tracks_detections = Array.trim(tracks_detections, num_tracks);
tracks_length = Array.trim(tracks_length, num_tracks);
tracks_start_frame = Array.trim(tracks_start_frame, num_tracks);
tracks_end_frame = Array.trim(tracks_end_frame, num_tracks);

print("\n✓ SUCCESS! Created " + num_tracks + " tracks");
print("  Track lengths: " + Array.findMinimum(tracks_length) + " to " + Array.findMaximum(tracks_length) + " frames");

// ============================================
// QUICK EXPORT AND VISUALIZATION
// ============================================

print("\nCreating visualization...");

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
}

Overlay.show;
print("✓ Visualization complete");

print("\n=== DIAGNOSTIC COMPLETE ===");
print("If tracks were found, the macro worked!");
print("Use the recommended max_displacement value above in the main macro.");
