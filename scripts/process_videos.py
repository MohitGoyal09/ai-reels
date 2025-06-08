import os
import glob
import json
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import tempfile
from pathlib import Path
import time
import re

# --- Add project root to sys.path if running locally from scripts/ ---
import sys
SCRIPT_DIR_PATH = Path(__file__).resolve().parent
PROJECT_ROOT_PATH = SCRIPT_DIR_PATH.parent
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_PATH))

# --- Import custom modules ---
try:
    from utils.video_processor import extract_frames
    from utils.text_cleaner import clean_text_for_nlp
    from models.fashion_detector_auto import FashionDetectorAuto
    from models.product_matcher import ProductMatcher
    from models.vibe_classifier_hf_zeroshot import VibeClassifierHFZeroShot
except ImportError as e:
    print(f"ERROR: Could not import all custom modules: {e}")
    exit(1)

# --- Global Configuration ---
BASE_DIR = PROJECT_ROOT_PATH
DATA_DIR = BASE_DIR / 'data'
INPUT_VIDEO_DIR = DATA_DIR / 'videos'
CAPTIONS_DIR = DATA_DIR / 'captions'
METADATA_JSON_DIR = DATA_DIR / 'metadata_json'
PRODUCT_METADATA_CSV_PATH = DATA_DIR / 'product_data.csv'
VIBE_TAXONOMY_JSON_PATH = DATA_DIR / 'vibeslist.json'

PRECOMPUTED_MODELS_DIR = BASE_DIR / 'models' / 'precomputed'
FAISS_INDEX_PATH = PRECOMPUTED_MODELS_DIR / 'catalog.index'
PRODUCT_IDS_MAP_PATH = PRECOMPUTED_MODELS_DIR / 'product_ids.csv'

OUTPUT_JSON_DIR = BASE_DIR / 'output'
DETECTED_FRAMES_DIR = BASE_DIR / 'detected_frames'  # New directory for detected frames

# --- Model Configuration ---
ZS_VIBE_MODEL_NAME = "facebook/bart-large-mnli"
ZS_VIBE_CONFIDENCE_THRESHOLD = 0.50
FASHION_DETECTOR_CONFIDENCE = 0.4

# --- Fashion Categories ---
MAIN_FASHION_CATEGORIES = [
    'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 
    'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 
    'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory',
    'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings',
    'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella'
]

SECONDARY_FASHION_CATEGORIES = [
    'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 
    'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 
    'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel'
]

# --- Helper Functions ---
def parse_product_tags_from_meta(tags_str):
    color = "N/A"
    print_pattern = "N/A"
    if pd.notna(tags_str) and isinstance(tags_str, str):
        tags_list = tags_str.split(',')
        for tag in tags_list:
            tag_cleaned = tag.strip().lower()
            if tag_cleaned.startswith('colour:'):
                color = tag.split(':', 1)[1].strip().title()
            elif tag_cleaned.startswith('print:'):
                print_pattern = tag.split(':', 1)[1].strip().title()
    return color, print_pattern

def extract_hashtags(text):
    if not text:
        return []
    hashtags = re.findall(r'#(\w+)', text)
    return [tag.lower() for tag in hashtags]

def enhance_vibe_detection(text, detected_vibes, vibe_keywords):
    """Enhance vibe detection by looking for specific keywords in hashtags and caption"""
    if not text:
        return detected_vibes
        
    hashtags = extract_hashtags(text)
    text_lower = text.lower()
    
    additional_vibes = set()
    for vibe, keywords in vibe_keywords.items():
        for keyword in keywords:
            if keyword in text_lower or keyword in hashtags:
                additional_vibes.add(vibe)
                break
    
    all_vibes = set(detected_vibes) | additional_vibes
    return list(all_vibes)

def create_product_output(yolos_class_name, yolos_confidence, match_info, original_frame_num, product_meta_df):
    """Helper function to create product output dictionary"""
    matched_prod_id = match_info['matched_product_id']
    
    p_name, p_actual_type, p_color, p_print = "N/A", yolos_class_name, "N/A", "N/A"
    if product_meta_df is not None and matched_prod_id is not None and matched_prod_id in product_meta_df.index:
        meta = product_meta_df.loc[matched_prod_id]
        p_name = meta.get('title', "N/A")
        if pd.notna(meta.get('product_type_alias')): 
            p_actual_type = meta['product_type_alias']
        p_color, p_print = parse_product_tags_from_meta(meta.get('product_tags'))
    
    # Use detected colors if available
    detected_colors = match_info.get('detected_colors', [])
    if detected_colors and detected_colors[0] != "unknown":
        p_color = ', '.join(detected_colors)
    
    return {
        "type": p_actual_type,
        "matched_product_id": str(matched_prod_id) if pd.notna(matched_prod_id) else None,
        "product_name": p_name,
        "color": p_color,
        "print_pattern": p_print,
        "match_type": match_info['match_type'],
        "detection_confidence": round(yolos_confidence, 4),
        "matching_similarity": round(match_info['similarity_score'], 4),
        "frame_number": original_frame_num
    }

def save_annotated_frame(frame_path, detections, video_id, frame_num, output_dir):
    """Save a frame with detection bounding boxes and labels"""
    try:
        # Load the image
        img = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw bounding boxes and labels for each detection
        for det in detections:
            class_name = det['class_name']
            bbox = det['bounding_box']
            conf = det['confidence_score']
            
            # Extract bounding box coordinates
            x, y, w, h = bbox
            
            # Colors based on category type
            color = "green" if class_name in MAIN_FASHION_CATEGORIES else "blue"
            
            # Draw rectangle
            draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
            
            # Draw label with confidence
            label = f"{class_name}: {conf:.2f}"
            text_bbox = draw.textbbox((x, y-15), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x, y-15), label, fill="white", font=font)
        
        # Save the annotated image
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_id}_frame_{frame_num}.jpg")
        img.save(output_path)
        
        return output_path
    except Exception as e:
        print(f"Error saving annotated frame: {e}")
        return None

# --- Main Video Processing Function ---
def process_single_video(
    video_path: Path, 
    captions_dir_path: Path,
    metadata_json_dir_path: Path,
    fashion_detector_instance: FashionDetectorAuto, 
    product_matcher_instance: ProductMatcher, 
    vibe_classifier_instance: VibeClassifierHFZeroShot, 
    product_meta_df: pd.DataFrame
    ):
    """
    Processes a single video end-to-end.
    """
    video_id = video_path.stem
    start_time_video = time.time()
    print(f"\n▶️ Starting processing for Video ID: {video_id}")

    # Create a directory for this video's detected frames
    video_frames_dir = DETECTED_FRAMES_DIR / video_id
    os.makedirs(video_frames_dir, exist_ok=True)

    # 1. Prepare Text for Vibe Classification
    raw_caption_from_txt = ""
    txt_caption_file_path = captions_dir_path / f"{video_id}.txt"
    if txt_caption_file_path.exists():
        with open(txt_caption_file_path, 'r', encoding='utf-8') as f: raw_caption_from_txt = f.read().strip()
    
    raw_caption_from_json, raw_user_bio = "", ""
    json_meta_file_path = metadata_json_dir_path / f"{video_id}.json"
    if json_meta_file_path.exists():
        try:
            with open(json_meta_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            if data.get("node",{}).get("edge_media_to_caption",{}).get("edges"):
                if data["node"]["edge_media_to_caption"]["edges"]: raw_caption_from_json = data["node"]["edge_media_to_caption"]["edges"][0]["node"].get("text","").strip()
            raw_user_bio = data.get("node",{}).get("owner",{}).get("biography","").strip()
        except Exception as e: print(f"  Error reading/parsing JSON for {video_id}: {e}")
            
    combined_text = raw_caption_from_txt
    if raw_caption_from_json and raw_caption_from_json.lower() != combined_text.lower(): combined_text += " " + raw_caption_from_json
    if raw_user_bio: combined_text += " " + raw_user_bio
    cleaned_text_for_vibe = clean_text_for_nlp(combined_text)
    
    # 2. Classify Vibes
    print(f"  🎤 Classifying vibes for '{video_id}'...")
    detected_vibes = []
    if cleaned_text_for_vibe:
        detected_vibes = vibe_classifier_instance.classify_vibes(cleaned_text_for_vibe, confidence_threshold=ZS_VIBE_CONFIDENCE_THRESHOLD)
        
        # Define vibe-specific keywords to enhance detection
        vibe_keywords = {
            "Coquette": ["coquette", "girly", "feminine", "lace", "bows", "pastel", "pink", "ribbons", "delicate", "romantic"],
            "Clean Girl": ["clean", "minimal", "classy", "sleek", "minimalist", "neutral", "gold", "slick", "dewy", "glossy"],
            "Cottagecore": ["cottagecore", "cottage", "rural", "vintage", "floral", "prairie", "garden", "cozy", "nature"],
            "Streetcore": ["street", "streetwear", "urban", "streetstyle", "oversized", "skate", "edgy", "hypebeast"],
            "Y2K": ["y2k", "2000s", "00s", "noughties", "retro", "vintage", "butterfly", "crop", "low", "rise"],
            "Boho": ["boho", "bohemian", "hippie", "gypsy", "flowy", "ethnic", "earthy", "layered", "relaxed"],
            "Party Glam": ["glam", "party", "glamour", "evening", "club", "sequin", "metallic", "sparkle", "shine"]
        }
        
        # Enhance vibe detection with hashtags and keywords
        detected_vibes = enhance_vibe_detection(combined_text, detected_vibes, vibe_keywords)
    
    print(f"    Vibes: {detected_vibes if detected_vibes else 'None Found'}")

    # 3. Frame Extraction, Object Detection, Product Matching
    processed_products_for_video = []
    unique_matched_products_tracker = set() # (yolos_class_name, matched_prod_id)
    frame_detections = {} # Store all detections for each frame

    with tempfile.TemporaryDirectory(prefix=f"flickd_frames_{video_id}_") as temp_frames_dir:
        print(f"  🖼️ Extracting frames to: {temp_frames_dir}...")
        # Extract frames with improved keyframe detection (reduced for speed)
        frame_paths_and_nums = extract_frames(
            str(video_path), 
            temp_frames_dir, 
            fps=0.5,
            use_keyframes=True,
            max_frames=20
        )
        print(f"    Extracted {len(frame_paths_and_nums)} frames.")
        
        if not frame_paths_and_nums:
            print(f"  No frames extracted for {video_id}. Skipping product detection.")
        else:
            print(f"  👀 Detecting & Matching products in frames...")
            
            # Track detected products per frame
            frame_products = {}
            
            for frame_idx, (frame_path_str, original_frame_num) in enumerate(frame_paths_and_nums):
                if frame_idx % 5 == 0: # Log progress every 5 frames
                    print(f"    Processing frame {frame_idx + 1}/{len(frame_paths_and_nums)} (original num: {original_frame_num})")

                yolos_detections = fashion_detector_instance.detect_fashion_items(
                    image_path=frame_path_str, 
                    frame_number=original_frame_num,
                    confidence_threshold=FASHION_DETECTOR_CONFIDENCE
                )
                
                if not yolos_detections: continue

                # Save the frame with detection bounding boxes
                annotated_frame_path = save_annotated_frame(
                    frame_path_str, 
                    yolos_detections,
                    video_id, 
                    original_frame_num,
                    video_frames_dir
                )
                
                # Store all detections for this frame
                frame_detections[original_frame_num] = {
                    "frame_path": frame_path_str,
                    "annotated_frame_path": annotated_frame_path,
                    "detections": yolos_detections
                }

                try:
                    full_frame_image_pil = Image.open(frame_path_str).convert("RGB")
                except Exception as e_img:
                    print(f"    Error opening frame image {frame_path_str}: {e_img}")
                    continue

                # Store products for this frame
                frame_products[original_frame_num] = []
                
                for det in yolos_detections:
                    yolos_class_name = det['class_name']
                    bbox = det['bounding_box']
                    yolos_confidence = det['confidence_score']
                    
                    # Prioritize main fashion categories over secondary components
                    is_main_category = yolos_class_name in MAIN_FASHION_CATEGORIES
                    is_secondary_category = yolos_class_name in SECONDARY_FASHION_CATEGORIES
                    
                    # Filter out irrelevant detections
                    if not (is_main_category or is_secondary_category):
                        continue
                        
                    # Skip secondary categories with low confidence
                    if is_secondary_category and yolos_confidence < 0.6:
                        continue
                    
                    x_tl, y_tl, w, h = bbox
                    img_w_pil, img_h_pil = full_frame_image_pil.size
                    x_br, y_br = min(x_tl + w, img_w_pil), min(y_tl + h, img_h_pil)
                    x_tl_safe, y_tl_safe = max(x_tl, 0), max(y_tl, 0)

                    # Skip tiny detections (likely false positives)
                    min_size = min(w, h)
                    if min_size < 30:  # Skip if either dimension is less than 30 pixels
                        continue
                        
                    # Skip if the bounding box has invalid dimensions
                    if w <= 0 or h <= 0 or x_tl_safe >= x_br or y_tl_safe >= y_br:
                        continue
                        
                    cropped_item_pil = full_frame_image_pil.crop((x_tl_safe, y_tl_safe, x_br, y_br))
                    
                    # Save the cropped detection
                    detection_output_path = os.path.join(
                        video_frames_dir, 
                        f"{video_id}_frame_{original_frame_num}_det_{hash(str(frame_products[original_frame_num]))}.jpg"
                    )
                    cropped_item_pil.save(detection_output_path)
                    
                    # Add detection path to the detection info
                    det['detection_image_path'] = detection_output_path
                    
                    match_info = product_matcher_instance.match_detected_item(cropped_item_pil)
                    
                    # Only process matches above threshold
                    if match_info['match_type'] != 'no_match':
                        matched_prod_id = match_info['matched_product_id']
                        
                        # Check if we've already processed this item
                        item_key = (yolos_class_name, matched_prod_id)
                        if item_key in unique_matched_products_tracker:
                            continue 
                        unique_matched_products_tracker.add(item_key)

                        output_product = create_product_output(
                            yolos_class_name, yolos_confidence, match_info, 
                            original_frame_num, product_meta_df
                        )
                        
                        # Add detection image path
                        output_product['detection_image_path'] = detection_output_path
                        
                        # Add bounding box information
                        output_product['bounding_box'] = {
                            'x': x_tl_safe,
                            'y': y_tl_safe,
                            'width': x_br - x_tl_safe,
                            'height': y_br - y_tl_safe
                        }
                        
                        # Add to frame products
                        frame_products[original_frame_num].append(output_product)
                        
                        # Add to overall products list
                        processed_products_for_video.append(output_product)
            
            # If we have no products, do a second pass with lower threshold
            if not processed_products_for_video:
                print("  No products found with standard settings, retrying with lower threshold...")
                
                for frame_idx, (frame_path_str, original_frame_num) in enumerate(frame_paths_and_nums):
                    # Only process every 3rd frame for efficiency in second pass
                    if frame_idx % 3 != 0:
                        continue
                        
                    yolos_detections = fashion_detector_instance.detect_fashion_items(
                        image_path=frame_path_str, 
                        frame_number=original_frame_num,
                        confidence_threshold=0.3  # Lower threshold
                    )
                    
                    if not yolos_detections: continue
                    
                    # Save the frame with detection bounding boxes (lower threshold pass)
                    annotated_frame_path = save_annotated_frame(
                        frame_path_str, 
                        yolos_detections,
                        video_id, 
                        original_frame_num,
                        video_frames_dir
                    )
                    
                    # Store all detections for this frame
                    frame_detections[original_frame_num] = {
                        "frame_path": frame_path_str,
                        "annotated_frame_path": annotated_frame_path,
                        "detections": yolos_detections
                    }
                    
                    full_frame_image_pil = Image.open(frame_path_str).convert("RGB")
                    
                    for det in yolos_detections:
                        yolos_class_name = det['class_name']
                        
                        # Only look for main categories in second pass
                        if yolos_class_name not in MAIN_FASHION_CATEGORIES:
                            continue
                            
                        bbox = det['bounding_box']
                        yolos_confidence = det['confidence_score']
                        
                        x_tl, y_tl, w, h = bbox
                        img_w_pil, img_h_pil = full_frame_image_pil.size
                        x_br, y_br = min(x_tl + w, img_w_pil), min(y_tl + h, img_h_pil)
                        x_tl_safe, y_tl_safe = max(x_tl, 0), max(y_tl, 0)
                        
                        if w > 0 and h > 0 and x_tl_safe < x_br and y_tl_safe < y_br:
                            cropped_item_pil = full_frame_image_pil.crop((x_tl_safe, y_tl_safe, x_br, y_br))
                            
                            # Save the cropped detection (second pass)
                            detection_output_path = os.path.join(
                                video_frames_dir, 
                                f"{video_id}_frame_{original_frame_num}_det_2nd_{len(processed_products_for_video)}.jpg"
                            )
                            cropped_item_pil.save(detection_output_path)
                            
                            # Add detection path to the detection info
                            det['detection_image_path'] = detection_output_path
                            
                            match_info = product_matcher_instance.match_detected_item(cropped_item_pil)
                            
                            if match_info['match_type'] != 'no_match':
                                matched_prod_id = match_info['matched_product_id']
                                item_key = (yolos_class_name, matched_prod_id)
                                if item_key in unique_matched_products_tracker:
                                    continue 
                                    
                                unique_matched_products_tracker.add(item_key)
                                
                                output_product = create_product_output(
                                    yolos_class_name, yolos_confidence, match_info, 
                                    original_frame_num, product_meta_df
                                )
                                
                                # Add detection image path
                                output_product['detection_image_path'] = detection_output_path
                                
                                # Add bounding box information
                                output_product['bounding_box'] = {
                                    'x': x_tl_safe,
                                    'y': y_tl_safe,
                                    'width': x_br - x_tl_safe,
                                    'height': y_br - y_tl_safe
                                }
                                
                                processed_products_for_video.append(output_product)
    
    # 4. Assemble final JSON
    final_video_output = {
        "video_id": video_id,
        "vibes": detected_vibes,
        "products": processed_products_for_video,
        "frames_dir": str(video_frames_dir),
        "frame_count": len(frame_detections),
        "frames": [
            {
                "frame_number": frame_num,
                "annotated_frame_path": info["annotated_frame_path"],
                "detection_count": len(info["detections"])
            }
            for frame_num, info in frame_detections.items()
        ]
    }
    elapsed_time_video = time.time() - start_time_video
    print(f"  ✅ Finished processing '{video_id}'. Duration: {elapsed_time_video:.2f}s. Found {len(detected_vibes)} vibes, {len(processed_products_for_video)} product matches.")
    print(f"  💾 Saved {len(frame_detections)} annotated frames to {video_frames_dir}")
    return final_video_output

# --- Main Execution Block ---
if __name__ == '__main__':
    overall_start_time = time.time()
    print("🚀 Flickd AI Full Pipeline - Initializing...")

    # Create output directories if they don't exist
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    os.makedirs(DETECTED_FRAMES_DIR, exist_ok=True)
    
    # Basic checks for essential input files/dirs
    essential_paths = [INPUT_VIDEO_DIR, CAPTIONS_DIR, METADATA_JSON_DIR, 
                       PRODUCT_METADATA_CSV_PATH, VIBE_TAXONOMY_JSON_PATH, 
                       FAISS_INDEX_PATH, PRODUCT_IDS_MAP_PATH]
    for p in essential_paths:
        if not p.exists():
            print(f"CRITICAL ERROR: Path not found: {p}. Please check your setup.")
            exit(1)

    # 1. Initialize All Models & Load Metadata (ONCE)
    print("\n[1/3] Initializing models and loading metadata...")
    try:
        detector = FashionDetectorAuto()
        matcher = ProductMatcher(index_path=str(FAISS_INDEX_PATH), product_id_map_path=str(PRODUCT_IDS_MAP_PATH))
        vibe_clf = VibeClassifierHFZeroShot(
            vibe_taxonomy_json_path=str(VIBE_TAXONOMY_JSON_PATH), 
            model_name=ZS_VIBE_MODEL_NAME
        )
        
        product_metadata_df = pd.read_csv(PRODUCT_METADATA_CSV_PATH)
        if 'id' in product_metadata_df.columns:
            product_metadata_df.set_index('id', inplace=True)
            print("  Product metadata loaded and indexed by 'id'.")
        else:
            print("  WARNING: 'id' column missing in product_data.csv. Metadata enrichment will be limited.")
            product_metadata_df = None
            
    except Exception as e_init:
        print(f"FATAL ERROR during model/metadata initialization: {e_init}")
        import traceback; traceback.print_exc()
        exit(1)
    print("  All models and metadata successfully initialized.")

    # 2. Find and Process Video Files
    print("\n[2/3] Finding video files to process...")
    video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
    all_video_file_paths = []
    for ext in video_extensions:
        all_video_file_paths.extend(list(INPUT_VIDEO_DIR.glob(ext)))

    if not all_video_file_paths:
        print(f"  No video files found in {INPUT_VIDEO_DIR} with extensions {video_extensions}. Exiting.")
    else:
        print(f"  Found {len(all_video_file_paths)} video(s) to process.")
        print("\n[3/3] Starting video processing loop...")
        for i, video_p_obj in enumerate(all_video_file_paths):
            print(f"\n--- Processing video {i+1}/{len(all_video_file_paths)}: {video_p_obj.name} ---")
            
            result_json_data = process_single_video(
                video_path=video_p_obj,
                captions_dir_path=CAPTIONS_DIR,
                metadata_json_dir_path=METADATA_JSON_DIR,
                fashion_detector_instance=detector,
                product_matcher_instance=matcher,
                vibe_classifier_instance=vibe_clf,
                product_meta_df=product_metadata_df
            )
            
            # Save the output JSON
            output_json_file = OUTPUT_JSON_DIR / f"{video_p_obj.stem}.json"
            try:
                with open(output_json_file, 'w', encoding='utf-8') as f_out:
                    json.dump(result_json_data, f_out, indent=4)
                print(f"  💾 Output saved to: {output_json_file}")
            except Exception as e_save:
                print(f"  ❌ ERROR saving JSON for {video_p_obj.stem}: {e_save}")
    
    overall_elapsed_time = time.time() - overall_start_time
    print(f"\n🏁 Flickd AI Full Pipeline Finished. Total processing time: {overall_elapsed_time:.2f}s")