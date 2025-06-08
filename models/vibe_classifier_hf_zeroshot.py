import re
import string
import json
import os
from pathlib import Path
import emoji
import torch
from transformers import pipeline 


def clean_text_for_nlp(text: str) -> str:
    """
    Cleans text for NLP tasks.
    """
    if not text or not isinstance(text, str):
        return ""
    
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'http\S+|www\S+|https\S+', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'@\w+', '', cleaned_text)
    cleaned_text = cleaned_text.replace('#', '')
    
    cleaned_text = emoji.replace_emoji(cleaned_text, replace=' ')  
    
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    cleaned_text = cleaned_text.translate(translator)
    
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  
    cleaned_text = cleaned_text.strip()
    return cleaned_text


class VibeClassifierHFZeroShot:
    def __init__(self, vibe_taxonomy_json_path: str, 
                 model_name: str):
        print(f"Initializing Hugging Face Zero-Shot Vibe Classifier with model: {model_name}...")
        if not Path(vibe_taxonomy_json_path).exists():
            raise FileNotFoundError(f"Vibe taxonomy JSON file not found at: {vibe_taxonomy_json_path}")
        try:
            with open(vibe_taxonomy_json_path, 'r', encoding='utf-8') as f:
                self.official_vibe_list = json.load(f)
            if not isinstance(self.official_vibe_list, list) or not all(isinstance(v, str) for v in self.official_vibe_list):
                raise ValueError("Vibe taxonomy JSON must be a list of strings.")
            if not self.official_vibe_list: raise ValueError("Vibe taxonomy list empty.")
            print(f"  Loaded {len(self.official_vibe_list)} official vibes (candidate labels): {self.official_vibe_list}")
            
            self.vibe_descriptions = {
                "Coquette": "a delicate, feminine, and flirtatious style with bows, pastels, lace, and ribbons",
                "Clean Girl": "a minimal, polished aesthetic with slicked back hair, gold jewelry, and neutral colors",
                "Cottagecore": "a romanticized rural lifestyle style with floral prints, loose dresses, and natural elements",
                "Streetcore": "an urban, edgy style with oversized clothes, sneakers, and streetwear brands",
                "Y2K": "early 2000s inspired fashion with low-rise jeans, baby tees, mini skirts, and bright colors",
                "Boho": "a bohemian free-spirited style with flowy fabrics, earthy tones, and layered accessories",
                "Party Glam": "glamorous evening wear with sequins, metallics, bodycon dresses, and statement pieces"
            }
            
            self.enhanced_candidates = []
            for vibe in self.official_vibe_list:
                if vibe in self.vibe_descriptions:
                    self.enhanced_candidates.append(f"{vibe}: {self.vibe_descriptions[vibe]}")
                else:
                    self.enhanced_candidates.append(vibe)
            
        except Exception as e: 
            raise Exception(f"Error loading vibe taxonomy: {e}")

        try:
            print(f"  Loading zero-shot classification pipeline for model {model_name}...")
            device_to_use = 0 if torch.cuda.is_available() else -1
            self.classifier_pipeline = pipeline("zero-shot-classification", model=model_name, device=device_to_use)
            print(f"  Zero-shot pipeline loaded successfully onto device: {'cuda' if device_to_use == 0 else 'cpu'}.")
        except Exception as e: 
            raise Exception(f"Error loading zero-shot pipeline: {e}")
        print("Hugging Face Zero-Shot Vibe Classifier initialized.")

    def classify_vibes(self, text_to_classify: str, confidence_threshold: float = 0.50, multi_label: bool = True):
        """
        Classifies text into fashion vibes using zero-shot classification
        """
        if not text_to_classify or not isinstance(text_to_classify, str) or not text_to_classify.strip(): 
            return []
            
        try:
            hypothesis_template = "This outfit, style, or fashion content is {}"
            
            classification_results = self.classifier_pipeline(
                text_to_classify,
                candidate_labels=self.enhanced_candidates,
                multi_label=multi_label,
                hypothesis_template=hypothesis_template
            )
            
            labels_to_original = {}
            for i, label in enumerate(classification_results['labels']):
                original_vibe = label.split(':', 1)[0] if ':' in label else label
                labels_to_original[label] = original_vibe
            
            scored_labels = [
                {'label': labels_to_original[lbl], 'score': scr} 
                for lbl, scr in zip(classification_results['labels'], classification_results['scores']) 
                if scr >= confidence_threshold
            ]
            
            sorted_scored_labels = sorted(scored_labels, key=lambda x: x['score'], reverse=True)
            
            if len(sorted_scored_labels) > 1:
                top_score = sorted_scored_labels[0]['score']
                second_score = sorted_scored_labels[1]['score']
                
                if top_score > second_score * 1.15:
                    detected_vibes_final = [sorted_scored_labels[0]['label']]
                else:
                    detected_vibes_final = [item['label'] for item in sorted_scored_labels[:3]]
            else:
                detected_vibes_final = [item['label'] for item in sorted_scored_labels[:3]]
                
            return detected_vibes_final
            
        except Exception as e:
            print(f"    Error during zero-shot vibe classification for text '{text_to_classify[:50]}...': {e}")
            return []


# if __name__ == "__main__":
#     SCRIPT_DIR_PATH = Path(__file__).resolve().parent
#     PROJECT_ROOT_PATH = SCRIPT_DIR_PATH.parent
#     
#     PROJECT_DATA_DIR = PROJECT_ROOT_PATH / 'data'
#     CAPTIONS_DIR = PROJECT_DATA_DIR / 'captions'
#     METADATA_JSON_DIR = PROJECT_DATA_DIR / 'metadata_json'
#     VIBE_JSON_PATH = str(PROJECT_DATA_DIR / 'vibeslist.json')
#     
#     if not Path(VIBE_JSON_PATH).exists():
#         print(f"Warning: Vibe list JSON not found at {VIBE_JSON_PATH}. Creating a dummy for testing.")
#         os.makedirs(PROJECT_DATA_DIR, exist_ok=True)
#         vibelist_content = ["Coquette", "Clean Girl", "Cottagecore", "Streetcore", "Y2K", "Boho", "Party Glam"]
#         with open(VIBE_JSON_PATH, 'w') as f_json: json.dump(vibelist_content, f_json)
#     
#     ZS_MODEL_NAME = "facebook/bart-large-mnli"
#     
#     vibe_classifier_hf_instance = None
#     try:
#         vibe_classifier_hf_instance = VibeClassifierHFZeroShot(
#             vibe_taxonomy_json_path=VIBE_JSON_PATH,
#             model_name=ZS_MODEL_NAME
#         )
#     except Exception as e:
#         print(f"CRITICAL ERROR initializing ZeroShot VibeClassifier ({ZS_MODEL_NAME}): {e}")
#         import traceback
#         traceback.print_exc()
#     
#     video_ids_to_process = [
#         "2025-05-22_08-25-12_UTC",
#         "2025-05-27_13-46-16_UTC",
#         "2025-05-31_14-01-37_UTC",
#         "2025-06-02_11-31-19_UTC",
#         "2025-05-28_13-42-32_UTC",
#         "2025-05-28_13-40-09_UTC"
#     ]
#     
#     if vibe_classifier_hf_instance:
#         print(f"\n\n--- Processing Video Captions with {ZS_MODEL_NAME} ---")
#         
#         ZERO_SHOT_CONFIDENCE_THRESHOLD = 0.50 
#     
#         def _get_combined_text_for_video_id_test(video_id: str, captions_dir: Path, metadata_dir: Path) -> str:
#             raw_caption_from_txt = "" 
#             txt_caption_path = captions_dir / f"{video_id}.txt"
#             if txt_caption_path.exists():
#                 with open(txt_caption_path, 'r', encoding='utf-8') as f: raw_caption_from_txt = f.read().strip()
#             else: print(f"    Warning: .txt not found for {video_id}")
#             raw_caption_from_json_meta, raw_user_bio_from_json = "", ""
#             json_metadata_path = metadata_dir / f"{video_id}.json"
#             if json_metadata_path.exists():
#                 with open(json_metadata_path, 'r', encoding='utf-8') as f: video_json_data = json.load(f)
#                 if video_json_data.get("node",{}).get("edge_media_to_caption",{}).get("edges"):
#                     if video_json_data["node"]["edge_media_to_caption"]["edges"]: raw_caption_from_json_meta = video_json_data["node"]["edge_media_to_caption"]["edges"][0]["node"].get("text","").strip()
#                 raw_user_bio_from_json = video_json_data.get("node",{}).get("owner",{}).get("biography","").strip()
#             else: print(f"    Warning: .json not found for {video_id}")
#             combined_text = raw_caption_from_txt
#             if raw_caption_from_json_meta and raw_caption_from_json_meta.lower() != combined_text.lower(): combined_text += " " + raw_caption_from_json_meta
#             if raw_user_bio_from_json: combined_text += " " + raw_user_bio_from_json
#             return combined_text
#     
#         for video_id in video_ids_to_process:
#             print(f"\n---------------------------------------------------------")
#             print(f"Processing Video ID: {video_id} (Model: {ZS_MODEL_NAME})")
#             print(f"---------------------------------------------------------")
#     
#             test_txt_file = CAPTIONS_DIR / f"{video_id}.txt"
#             if not test_txt_file.exists():
#                 os.makedirs(CAPTIONS_DIR, exist_ok=True)
#                 with open(test_txt_file, 'w') as f: f.write(f"Dummy caption for {video_id}.")
#                 print(f"    Created dummy .txt for {video_id}")
#             test_json_file = METADATA_JSON_DIR / f"{video_id}.json"
#             if not test_json_file.exists():
#                 os.makedirs(METADATA_JSON_DIR, exist_ok=True)
#                 with open(test_json_file, 'w') as f: json.dump({"node":{"owner":{"biography":f"Dummy bio for {video_id}"}}},f)
#                 print(f"    Created dummy .json for {video_id}")
#                 
#             combined_text_from_files = _get_combined_text_for_video_id_test(
#                 video_id,
#                 CAPTIONS_DIR,
#                 METADATA_JSON_DIR
#             )
#             
#             cleaned_text_for_classification = clean_text_for_nlp(combined_text_from_files)
#             print(f"\n  Cleaned Text (Input to Zero-Shot VibeClassifier):\n    \"\"\"\n    {cleaned_text_for_classification}\n    \"\"\"")
#     
#             if cleaned_text_for_classification:
#                 detected_vibes = vibe_classifier_hf_instance.classify_vibes(
#                     cleaned_text_for_classification, 
#                     confidence_threshold=ZERO_SHOT_CONFIDENCE_THRESHOLD
#                 )
#                 print(f"\n  >>> DETECTED VIBES (Model: {ZS_MODEL_NAME}, Threshold: {ZERO_SHOT_CONFIDENCE_THRESHOLD}) for '{video_id}': {detected_vibes} <<<")
#             else:
#                 print(f"\n  >>> No text to classify for '{video_id}'. Detected Vibes: [] <<<")
#     
#         print(f"\n\n--- Zero-Shot Vibe Classification Test Run with {ZS_MODEL_NAME} Finished ---")
#     else:
#         print(f"ZeroShot VibeClassifier instance with {ZS_MODEL_NAME} could not be initialized. Test run aborted.")