# AI-Reels: Fashion Analysis Pipeline

A computer vision and NLP pipeline that analyzes fashion videos to detect fashion items, match them to product catalogs, and classify fashion vibes.

## 🌟 Features

- **Fashion Item Detection**: Uses YOLOS Fashionpedia to detect clothing items and accessories in videos
- **Product Matching**: Matches detected items to a product catalog using CLIP embeddings and FAISS
- **Fashion Vibe Classification**: Detects fashion vibes (Coquette, Clean Girl, Y2K, etc.) using zero-shot text classification
- **Color Recognition**: Identifies dominant colors in detected fashion items
- **Video Frame Analysis**: Extracts key frames from videos using scene change detection
- **JSON Output**: Generates structured JSON output for each video with detected vibes and matched products

## 📋 Requirements

- Python 3.9+
- PyTorch
- Transformers
- OpenCV
- FAISS
- PIL
- Pandas
- HuggingFace models

## Demo
https://www.loom.com/share/36d6f1befca84d62be7003119b13adf2?sid=bba8222e-5d43-4a09-8653-6495ab349ae5


## 🚀 Quick Start

1. **Clone the repository**

   ```
   git clone https://github.com/MohitGoyal09/ai-reels.git
   cd ai-reels
   ```

2. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

3. **Prepare your data structure**

   ```
   data/
   ├── videos/             # Place your .mp4 video files here
   ├── captions/           # Place [video_id].txt files with captions here
   ├── metadata_json/      # Place [video_id].json files with metadata here
   ├── product_data.csv    # Product catalog with metadata
   └── vibeslist.json      # List of fashion vibes to detect
   
   models/precomputed/
   ├── catalog.index       # FAISS index of product embeddings
   └── product_ids.csv     # Mapping of FAISS indices to product IDs
   ```

4. **Run the pipeline directly**

   ```
   python scripts/process_videos.py
   ```

5. **Check results**

   ```
   output/
   └── [video_id].json     # Results for each processed video
   ```

## 📊 Output Format

Each processed video generates a JSON file with the following structure:

```json
{
  "video_id": "2025-05-22_08-25-12_UTC",
  "vibes": ["Boho", "Clean Girl"],
  "products": [
    {
      "type": "dress",
      "matched_product_id": "12345",
      "product_name": "Floral Summer Dress",
      "color": "Blue, White",
      "print_pattern": "Floral",
      "match_type": "similar",
      "detection_confidence": 0.9231,
      "matching_similarity": 0.8765,
      "frame_number": 120,
      "detection_image_path": "/path/to/detection/image.jpg",
      "bounding_box": {
        "x": 100,
        "y": 150,
        "width": 200,
        "height": 300
      }
    },
    // More products...
  ],
  "frames_dir": "/path/to/detected_frames/video_id",
  "frame_count": 10,
  "frames": [
    {
      "frame_number": 120,
      "annotated_frame_path": "/path/to/annotated/frame.jpg",
      "detection_count": 3
    },
    // More frames...
  ]
}
```

## 📸 Detected Frames

The system saves annotated frames and individual detected fashion items:

1. **Annotated Frames**: Each frame with detected objects is saved with bounding boxes and labels.
2. **Individual Detections**: Each detected fashion item is cropped and saved separately.

These images can be accessed:

- Directly from the filesystem in the `detected_frames/{video_id}/` directory


## 🧠 Architecture

The system consists of several interconnected components:

1. **Video Processor**: Extracts keyframes from videos based on scene change detection
2. **Fashion Detector**: Detects fashion items using YOLOS Fashionpedia model
3. **Product Matcher**: Matches detected items to a catalog using CLIP embeddings and FAISS
4. **Vibe Classifier**: Classifies fashion vibes using zero-shot classification
5. **Main Pipeline**: Orchestrates the entire process and generates JSON output

## 📁 Project Structure

```
ai-reels/
├── data/                  # Data directory
├── models/                # Model implementations
│   ├── fashion_detector_auto.py
│   ├── product_matcher.py
│   ├── vibe_classifier_hf_zeroshot.py
│   └── precomputed/       # Pre-computed model files
├── output/                # Output JSON files
├── detected_frames/       # Saved frames with detections
│   └── [video_id]/        # Subdirectories for each video
├── scripts/               # Main scripts
│   └── process_videos.py  # Main pipeline script
├── utils/                 # Utility functions
│   ├── text_cleaner.py
│   └── video_processor.py
└── README.md              # This file
```

## ⚙️ Configuration

The main configuration parameters are defined at the top of `scripts/process_videos.py`:

- `ZS_VIBE_CONFIDENCE_THRESHOLD`: Confidence threshold for vibe classification (default: 0.50)
- `FASHION_DETECTOR_CONFIDENCE`: Confidence threshold for fashion item detection (default: 0.40)
- Various file paths for input/output data

## 🔧 Advanced Usage

### Parallel Processing

For large batches of videos, you can implement parallel processing by modifying the main loop in `process_videos.py`:

```python
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_single_video, ...) for video_p_obj in all_video_file_paths]
    for future in concurrent.futures.as_completed(futures):
        result_json_data = future.result()
        # Save result...
```

### Custom Fashion Categories

You can modify the `MAIN_FASHION_CATEGORIES` and `SECONDARY_FASHION_CATEGORIES` lists in `process_videos.py` to focus on specific fashion items.

### Customizing Vibe Detection

Edit the vibe keywords in the `vibe_keywords` dictionary in `process_single_video()` to enhance vibe detection for specific fashion styles.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
