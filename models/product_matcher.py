import faiss
import numpy as np
import pandas as pd
from PIL import Image # For handling cropped images
from sentence_transformers import SentenceTransformer # To load CLIP model for query embeddings
import os # For os.path.exists
import torch
from torchvision import transforms
import matplotlib.colors as mcolors

class ProductMatcher:
    def __init__(self, index_path, product_id_map_path, clip_model_name='clip-ViT-B-32'):
        """
        Initializes the ProductMatcher.

        Args:
            index_path (str): Path to the pre-computed FAISS index file.
            product_id_map_path (str): Path to the CSV file mapping FAISS indices to Product IDs.
            clip_model_name (str): Name of the CLIP model to use for generating query embeddings.
                                   This MUST be the same model architecture used to build the FAISS index.
        """
        print("Initializing Product Matcher...")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not os.path.exists(product_id_map_path):
            raise FileNotFoundError(f"Product ID map file not found: {product_id_map_path}")

        print(f"  Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(index_path)
        print(f"  FAISS index loaded. Contains {self.index.ntotal} items.")

        print(f"  Loading product ID map from: {product_id_map_path}")
        self.product_id_df = pd.read_csv(product_id_map_path)
        # Ensure the column name matches what was saved by build_faiss_index.py (should be 'Product ID')
        if 'Product ID' not in self.product_id_df.columns:
            raise ValueError(f"'Product ID' column not found in {product_id_map_path}. Found columns: {self.product_id_df.columns.tolist()}")
        if len(self.product_id_df) != self.index.ntotal:
            print(f"Warning: Number of entries in product_id_map ({len(self.product_id_df)}) "
                  f"does not match number of items in FAISS index ({self.index.ntotal}). "
                  "This might lead to errors if build_faiss_index.py had issues.")

        print(f"  Loading CLIP model for querying: {clip_model_name}...")
        try:
            # Using SentenceTransformer as it's consistent with how build_faiss_index.py created embeddings.
            self.clip_model = SentenceTransformer(clip_model_name)
        except Exception as e:
            print(f"Error loading CLIP model {clip_model_name}: {e}")
            raise # Re-raise the exception to stop initialization if model load fails
        
        # Add image processing for color extraction
        self.color_preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        # Define basic color mappings
        self.color_mappings = {
            'red': ['#FF0000', '#8B0000', '#FF4500', '#FF6347', '#DC143C', '#B22222'],
            'green': ['#008000', '#006400', '#228B22', '#32CD32', '#00FF00', '#7CFC00'],
            'blue': ['#0000FF', '#00008B', '#4169E1', '#1E90FF', '#00BFFF', '#87CEEB'],
            'yellow': ['#FFFF00', '#FFD700', '#FFA500', '#FFFFE0'],
            'orange': ['#FFA500', '#FF8C00', '#FF7F50', '#FF6347'],
            'purple': ['#800080', '#8B008B', '#9400D3', '#9932CC', '#BA55D3'],
            'pink': ['#FFC0CB', '#FF69B4', '#FF1493', '#C71585', '#DB7093'],
            'brown': ['#A52A2A', '#8B4513', '#D2691E', '#CD853F', '#DEB887'],
            'white': ['#FFFFFF', '#F8F8FF', '#F5F5F5', '#FFFAFA', '#F0F8FF'],
            'black': ['#000000', '#2F4F4F', '#696969', '#808080', '#A9A9A9'],
            'grey': ['#808080', '#A9A9A9', '#C0C0C0', '#D3D3D3', '#DCDCDC'],
            'beige': ['#F5F5DC', '#FAEBD7', '#FFE4C4', '#FFEBCD', '#FFF8DC']
        }
        
        print("Product Matcher initialized successfully.")

    def extract_dominant_colors(self, image, n_colors=3):
        """Extract the dominant colors from an image"""
        try:
            img_tensor = self.color_preprocess(image)
            img_np = img_tensor.numpy().transpose(1, 2, 0)  # Convert to HWC format
            
            # Reshape to pixels x 3 (RGB)
            pixels = img_np.reshape(-1, 3)
            
            # Simple clustering to find dominant colors (using pixel sampling for speed)
            sampled_pixels = pixels[::20]  # Sample every 20th pixel
            
            # Convert to 0-255 range for easier color naming
            sampled_pixels_255 = (sampled_pixels * 255).astype(np.uint8)
            
            # Count frequency of pixels
            unique_colors, counts = np.unique(sampled_pixels_255, axis=0, return_counts=True)
            
            # Get top n colors
            if len(unique_colors) == 0:
                return ["unknown"]
                
            top_indices = np.argsort(-counts)[:n_colors]
            top_colors = unique_colors[top_indices]
            
            # Convert to hex for color mapping
            color_names = []
            for color_rgb in top_colors:
                hex_color = '#%02x%02x%02x' % tuple(color_rgb)
                
                # Find closest named color
                min_dist = float('inf')
                closest_color = "unknown"
                
                for color_name, hex_values in self.color_mappings.items():
                    for hex_val in hex_values:
                        # Convert hex to RGB
                        hex_val = hex_val.lstrip('#')
                        rgb = tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))
                        
                        # Calculate Euclidean distance with overflow protection
                        try:
                            dist = sum((int(a)-int(b))**2 for a, b in zip(color_rgb, rgb))
                        except (OverflowError, ValueError):
                            dist = float('inf')
                        
                        if dist < min_dist:
                            min_dist = dist
                            closest_color = color_name
                
                if closest_color not in color_names:
                    color_names.append(closest_color)
            
            return color_names
            
        except Exception as e:
            print(f"Error extracting colors: {e}")
            return ["unknown"]

    def match_detected_item(self, cropped_pil_image: Image.Image, k_matches=1):
        """
        Finds the best product match(es) for a cropped image of a detected object.

        Args:
            cropped_pil_image (PIL.Image.Image): The cropped image of the detected object.
            k_matches (int): Number of top matches to return (currently only processes the top 1 for 'match_type').

        Returns:
            dict: A dictionary containing 'match_type', 'matched_product_id', 'similarity_score',
                  and 'detected_colors' for the best match.
        """
        if cropped_pil_image is None:
            return {
                'match_type': 'no_match',
                'matched_product_id': None,
                'similarity_score': 0.0,
                'detected_colors': []
            }

        try:
            # Extract dominant colors from the image
            detected_colors = self.extract_dominant_colors(cropped_pil_image)
            
            # 1. Generate embedding for the cropped query image
            # Convert PIL image to RGB if it has an alpha channel, as CLIP expects 3 channels.
            query_embedding = self.clip_model.encode([cropped_pil_image.convert("RGB")]) # Result is a numpy array (1, embedding_dim)

            # Normalize the query embedding (L2 normalization)
            # This is crucial for cosine similarity when using IndexFlatIP.
            faiss.normalize_L2(query_embedding)

            # 2. Search FAISS index
            # D = distances (or dot products for IndexFlatIP after normalization), I = indices
            # We search for k_matches, but current logic only processes the top one for 'match_type'.
            similarity_scores_faiss, faiss_indices = self.index.search(query_embedding.astype(np.float32), k=k_matches)

            # Process the top match (index 0)
            best_match_faiss_index = faiss_indices[0][0]
            # For IndexFlatIP with normalized vectors, the "distance" D is actually the dot product (cosine similarity).
            # No need to convert D if it's already similarity. If it were L2 distance, you'd convert.
            similarity_to_best_match = float(similarity_scores_faiss[0][0])

            # 3. Retrieve Product ID using the FAISS index
            # The self.product_id_df's own index (0 to N-1) corresponds to the FAISS index order.
            matched_product_id_value = self.product_id_df.iloc[best_match_faiss_index]['Product ID']

            # 4. Classify match type based on similarity score (as per hackathon doc)
            # Thresholds are: exact > 0.9, similar >= 0.75
            if similarity_to_best_match > 0.9:
                match_type = 'exact'
            elif similarity_to_best_match >= 0.75:
                match_type = 'similar'
            else:
                match_type = 'no_match'

            final_matched_product_id = None
            if match_type != 'no_match':
                final_matched_product_id = matched_product_id_value

            # For now, we return info for the single best match
            return {
                'match_type': match_type,
                'matched_product_id': final_matched_product_id,
                'similarity_score': similarity_to_best_match,
                'detected_colors': detected_colors
            }

        except IndexError:
            # This can happen if faiss_indices is empty or smaller than expected,
            # though with k=1 and a non-empty index, it's less likely for the top match.
            print(f"Error: Index out of bounds during FAISS result processing. FAISS indices: {faiss_indices}")
            return {'match_type': 'no_match', 'matched_product_id': None, 'similarity_score': 0.0, 'detected_colors': []}
        except Exception as e:
            print(f"Error during product matching: {e}")
            # import traceback
            # traceback.print_exc() # For detailed debugging
            return {'match_type': 'no_match', 'matched_product_id': None, 'similarity_score': 0.0, 'detected_colors': []}
