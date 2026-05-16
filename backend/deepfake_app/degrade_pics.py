import os
import random
from PIL import Image
from pathlib import Path

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Your exact input directories
REAL_INPUT_DIR = r"D:\Projects\Major Project\Deepfake Detection\datasets\New folder\real-vs-fake\hd"
#FAKE_INPUT_DIR = r"D:\Projects\Major Project\Deepfake Detection\datasets\test\real_and_fake_face\training_fake"

# Where the ruined images will be saved
OUTPUT_BASE_DIR = r"D:\Projects\Major Project\Deepfake Detection\datasets\New folder\real-vs-fake\degraded"
REAL_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "real")
FAKE_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "fake")

# Degradation settings
SAMPLE_SIZE = 500
JPEG_QUALITY = 30  # 30% quality = heavy compression/WhatsApp quality

def degrade_images(input_dir, output_dir, category_name):
    print(f"\n--- Processing {category_name.upper()} Images ---")
    
    # 1. Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Get all valid image files from the input directory
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    all_files = [f for f in os.listdir(input_dir) if Path(f).suffix.lower() in valid_extensions]
    
    if len(all_files) == 0:
        print(f"❌ No images found in {input_dir}!")
        return

    # 3. Randomly sample 500 images (or all of them if less than 500)
    num_to_sample = min(SAMPLE_SIZE, len(all_files))
    sampled_files = random.sample(all_files, num_to_sample)
    
    print(f"Found {len(all_files)} images. Randomly selected {num_to_sample} for degradation.")
    
    # 4. Compress and save
    success_count = 0
    for idx, filename in enumerate(sampled_files, 1):
        input_path = os.path.join(input_dir, filename)
        
        # Force the output to be .jpg since we are applying JPEG compression
        base_name = Path(filename).stem
        output_filename = f"{base_name}_degraded.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # Convert to RGB (drops alpha channel if it's a PNG) and compress
            with Image.open(input_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(output_path, "JPEG", quality=JPEG_QUALITY)
            success_count += 1
            
            # Print a progress update every 100 images
            if idx % 100 == 0 or idx == num_to_sample:
                print(f"  [{idx}/{num_to_sample}] Degraded and saved...")
                
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")

    print(f"✅ Successfully degraded {success_count} {category_name} images!")

if __name__ == "__main__":
    print("=====================================================")
    print("🔥 INITIATING THE DEGRADATION TEST (FINAL BOSS) 🔥")
    print("=====================================================")
    
    degrade_images(REAL_INPUT_DIR, REAL_OUTPUT_DIR, "Real")
    #degrade_images(FAKE_INPUT_DIR, FAKE_OUTPUT_DIR, "Fake")
    
    print("\n=====================================================")
    print(f"🎉 DONE! Your degraded dataset is ready at:")
    print(f"   {OUTPUT_BASE_DIR}")
    print("=====================================================")

'''
import os
import random
from PIL import Image, ImageFilter
from pathlib import Path

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
REAL_INPUT_DIR = r"D:\Projects\Major Project\Deepfake Detection\datasets\new_combined\real"
FAKE_INPUT_DIR = r"D:\Projects\Major Project\Deepfake Detection\datasets\new_combined\fake"

# Save to a NEW folder so we don't overwrite the 30% test
OUTPUT_BASE_DIR = r"D:\Projects\Major Project\Deepfake Detection\datasets\extreme_degraded_test"
REAL_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "real")
FAKE_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "fake")

SAMPLE_SIZE = 500
# EXTREME SETTINGS
JPEG_QUALITY = 10     # Down from 30 (this will look like a deep-fried meme)
BLUR_RADIUS = 1.5     # Simulates being out-of-focus or screenshotted

def extreme_degrade(input_dir, output_dir, category_name):
    print(f"\n--- Processing {category_name.upper()} Images ---")
    os.makedirs(output_dir, exist_ok=True)
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    all_files = [f for f in os.listdir(input_dir) if Path(f).suffix.lower() in valid_extensions]
    
    num_to_sample = min(SAMPLE_SIZE, len(all_files))
    sampled_files = random.sample(all_files, num_to_sample)
    
    success_count = 0
    for idx, filename in enumerate(sampled_files, 1):
        input_path = os.path.join(input_dir, filename)
        output_filename = f"{Path(filename).stem}_extreme.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            with Image.open(input_path) as img:
                rgb_img = img.convert("RGB")
                # 1. Apply Blur
                blurred_img = rgb_img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
                # 2. Apply Extreme Compression
                blurred_img.save(output_path, "JPEG", quality=JPEG_QUALITY)
                
            success_count += 1
            if idx % 100 == 0 or idx == num_to_sample:
                print(f"  [{idx}/{num_to_sample}] Destroyed and saved...")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    print("=====================================================")
    print("☢️ INITIATING LEVEL 100 TORTURE TEST ☢️")
    print("=====================================================")
    extreme_degrade(REAL_INPUT_DIR, REAL_OUTPUT_DIR, "Real")
    extreme_degrade(FAKE_INPUT_DIR, FAKE_OUTPUT_DIR, "Fake")
    print("\n✅ Extreme Degraded Dataset Ready!")
'''