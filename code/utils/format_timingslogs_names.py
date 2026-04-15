import os
import re

# Change to your stimuli directory
os.chdir("./data/stimuli")

for filename in os.listdir("."):
    # Look for files that match your pattern
    # e.g., "CONV_001_TimingsLog_20200108_123845_events.csv"
    if re.match(r"^CONV_\d+_TimingsLog_.*_events\.csv$", filename):
        # Extract the ID
        match = re.match(r"^CONV_(\d+)_TimingsLog_.*_events\.csv$", filename)
        if match:
            subject_id = match.group(1)  # e.g., "001"
            # Build the new directory "conv-001"
            new_dir = f"conv-{subject_id}"
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            
            # Construct new filename "conv-001_events.csv"
            new_filename = f"conv-{subject_id}_events.csv"
            old_path = os.path.join(".", filename)
            new_path = os.path.join(new_dir, new_filename)
            
            print(f"Moving {old_path} -> {new_path}")
            os.rename(old_path, new_path)