from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch
import os

class FashionDetectorAuto:
    def __init__(self, model_name='valentinafeve/yolos-fashionpedia'):
        print(f"Loading Hugging Face AutoModel: {model_name}...")
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            # Get category names from model config
            self.categories = self.model.config.id2label
            if not self.categories: # Fallback if id2label is not populated as expected
                 self.categories = {0: 'shirt, blouse', 1: 'top, t-shirt, sweatshirt', 2: 'sweater', 3: 'cardigan', 4: 'jacket', 5: 'vest', 6: 'pants', 7: 'shorts', 8: 'skirt', 9: 'coat', 10: 'dress', 11: 'jumpsuit', 12: 'cape', 13: 'glasses', 14: 'hat', 15: 'headband, head covering, hair accessory', 16: 'tie', 17: 'glove', 18: 'watch', 19: 'belt', 20: 'leg warmer', 21: 'tights, stockings', 22: 'sock', 23: 'shoe', 24: 'bag, wallet', 25: 'scarf', 26: 'umbrella', 27: 'hood', 28: 'collar', 29: 'lapel', 30: 'epaulette', 31: 'sleeve', 32: 'pocket', 33: 'neckline', 34: 'buckle', 35: 'zipper', 36: 'applique', 37: 'bead', 38: 'bow', 39: 'flower', 40: 'fringe', 41: 'ribbon', 42: 'rivet', 43: 'ruffle', 44: 'sequin', 45: 'tassel'}


        except Exception as e:
            print(f"Error loading Hugging Face AutoModel for {model_name}: {e}")
            raise
        print("Hugging Face AutoModel loaded successfully.")

    def detect_fashion_items(self, image_path, frame_number, confidence_threshold=0.5):
        if isinstance(image_path, str) and not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return []

        detections_output = []
        try:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            elif isinstance(image_path, Image.Image):
                image = image_path.convert("RGB")
            else:
                print("Invalid image_path type. Must be str or PIL.Image")
                return []

            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.image_processor.post_process_object_detection(outputs, threshold=confidence_threshold, target_sizes=target_sizes)[0]

            for score, label_idx, box_coords in zip(results["scores"], results["labels"], results["boxes"]):
                # No need for extra score filtering if post_process_object_detection's threshold works

                class_name = self.categories[label_idx.item()] # map integer label to string

                xmin, ymin, xmax, ymax = box_coords.tolist()

                x_tl = int(xmin)
                y_tl = int(ymin)
                width = int(xmax - xmin)
                height = int(ymax - ymin)

                detections_output.append({
                    'class_name': class_name,
                    'bounding_box': [x_tl, y_tl, width, height],
                    'confidence_score': score.item(),
                    'frame_number': frame_number
                })
        except Exception as e:
            print(f"Error during AutoModel inference on {image_path}: {e}")
            # import traceback
            # traceback.print_exc()
            return []

        return detections_output

