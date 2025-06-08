
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import os

print("Step 1: Load CLIP model")
model = SentenceTransformer('clip-ViT-B-32')


print("Step 2: Load Product Catalog from data/product_catalog.csv")
try:
    catalog_df_raw = pd.read_csv('data/product_catalog.csv')
except Exception as e:
    print(f"Error reading CSV. Make sure 'data/product_catalog.csv' exists and is a valid CSV. Error: {e}")
    print("Ensure you saved your 'images.xlsx' as 'product_catalog.csv' in the 'data' folder.")
    exit()


print("Step 2a: Processing product catalog to get unique products (first image per ID)")

if 'id' not in catalog_df_raw.columns or 'image_url' not in catalog_df_raw.columns:
    print("Error: CSV must contain 'id' and 'image_url' columns.")
    print(f"Columns found: {catalog_df_raw.columns.tolist()}")
    exit()

# Drop rows where image_url might be missing, if any
catalog_df_raw.dropna(subset=['image_url'], inplace=True)
# Get the first occurrence of each unique 'id'
unique_products_df = catalog_df_raw.drop_duplicates(subset=['id'], keep='first')

image_urls = unique_products_df['image_url'].tolist()
product_ids_list = unique_products_df['id'].tolist() 

if not image_urls:
    print("Error: No image URLs found after processing the catalog. Please check your CSV.")
    exit()

# Ensure the output directory for models exists
os.makedirs('models/precomputed', exist_ok=True)

print(f"Step 3: Generating embeddings for {len(image_urls)} unique catalog images...")
catalog_embeddings = []
processed_product_ids = []
for product_id, url in zip(product_ids_list, image_urls):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        embedding = model.encode([img])[0]
        catalog_embeddings.append(embedding)
        processed_product_ids.append(product_id)
    except requests.exceptions.RequestException as e:
        print(f"Warning: Request failed for Product ID {product_id}, URL {url}. Error: {e}. Skipping.")
    except Exception as e:
        print(f"Warning: Could not process image for Product ID {product_id}, URL {url}. Error: {e}. Skipping.")
       

if not catalog_embeddings:
    print("Error: No embeddings were generated. Halting.")
    exit()

catalog_embeddings_np = np.array(catalog_embeddings).astype('float32')

print("Step 4: Building and saving FAISS index...")
embedding_dimension = catalog_embeddings_np.shape[1]
faiss.normalize_L2(catalog_embeddings_np) 
index = faiss.IndexFlatIP(embedding_dimension)
index.add(catalog_embeddings_np)

faiss.write_index(index, 'models/precomputed/catalog.index')


pd.DataFrame({'Product ID': processed_product_ids}).to_csv('models/precomputed/product_ids.csv', index=False)

print(f"Preprocessing complete! Your product search index is ready with {len(processed_product_ids)} items.")