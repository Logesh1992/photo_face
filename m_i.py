import os
import shutil
import torch as t_1
import imageio
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps
import cv2
import rawpy
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import time

device = t_1.device('cuda' if t_1.cuda.is_available() else 'cpu')
print(device)

t_1.backends.cudnn.benchmark = True  # Optimizes for variable input sizes
t_1.set_grad_enabled(False)  # Disable gradients for inference
# Initialize the models
model = InceptionResnetV1(pretrained='vggface2', device=device,).eval()
mtcnn = MTCNN(image_size=145, keep_all=True, margin=20, min_face_size=20,
              thresholds=[0.8, 0.9, 0.9], factor=0.5, post_process=True,
              device=device)

start_time = time.time()

image_dir = "me vagamon"
sample_image_path = "me vagamon/IMG_3975.DNG"  # Path to the sample image for comparison
output_dir = "output"
supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.cr2', '.dng', '.nef', '.arw']

os.makedirs(output_dir, exist_ok=True)
print(f"Total image files in directory: {len([f for f in os.listdir(image_dir) if any(f.lower().endswith(ext) for ext in supported_formats)])}")
# Function to load image with handling for various formats
def load_image(image_path):
    try:
        if image_path.lower().endswith(('.cr2', '.nef', '.arw')):
            try:
                with rawpy.imread(image_path) as raw:
                    rgb_image = raw.postprocess()
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_image)
                    img = ImageOps.exif_transpose(img)  # Fix orientation
                    return img
            except rawpy.LibRawFileUnsupportedError:
                print(f"Error: Unsupported RAW file format {image_path}. Skipping.")
                return None
        elif image_path.lower().endswith('.dng'):
            try:
                reader = imageio.get_reader(image_path, format='DNG')
                image = reader.get_data(0)
                img = Image.fromarray(image)
                img = ImageOps.exif_transpose(img)  # Fix orientation
                return img
            except Exception as e:
                print(f"Error: Unable to process DNG file {image_path}: {e}")
                return None
        else:
            img = Image.open(image_path).convert('RGB')
            img = ImageOps.exif_transpose(img)  # Fix orientation
            return img
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file {image_path}. Skipping.")
        return None

# Function to get face embeddings from images
def get_face_embeddings(image):
    with t_1.amp.autocast(device_type='cuda'):
        img_cropped = mtcnn(image)  # Detect and crop faces
        if img_cropped is not None:
            embeddings = []
            if isinstance(img_cropped, t_1.Tensor):
                if img_cropped.dim() == 3:
                    img_cropped = img_cropped.unsqueeze(0)  # Add batch dimension if needed

                img_cropped = img_cropped.to(device)

                with t_1.no_grad():
                    embeddings = model(img_cropped).cpu().detach().numpy()  # Generate embeddings
            return embeddings
    return None

# Function to process a batch of images
def process_image_batch(image_batch, sample_face_embedding, best_threshold, output_dir):
    for image_path, img in image_batch:
        face_embeddings = get_face_embeddings(img)
        if face_embeddings is not None:
            for embedding in face_embeddings:
                similarity = cosine_similarity([sample_face_embedding], [embedding])[0][0]
                print(f"Checking similarity for {image_path}: {similarity}")
                if similarity > best_threshold:
                    print(f"Matching face found in {image_path} with similarity: {similarity:.4f}")
                    shutil.copy(image_path, output_dir)
                    break

# Batch image processing function
def batch_images(image_dir, batch_size=12):  # Batch size optimized for RTX 4050 with 6GB VRAM
    image_batch = []
    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in supported_formats):
            image_path = os.path.join(image_dir, filename)
            img = load_image(image_path)
            if img is not None:
                image_batch.append((image_path, img))
                if len(image_batch) == batch_size:
                    yield image_batch
                    image_batch = []
    if image_batch:
        yield image_batch

# Load and process sample image for comparison
sample_img = load_image(sample_image_path)
if sample_img is None:
    print("No valid sample image found.")
    exit()

sample_face_embeddings = get_face_embeddings(sample_img)
if sample_face_embeddings is None or len(sample_face_embeddings) == 0:
    print("No face detected in the sample image.")
    exit()

sample_face_embedding = sample_face_embeddings[0]  # Use the first face embedding for comparison
best_threshold = 0.50

# Parallel processing using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on your CPU cores
    print("Starting batch processing of images...")
    for image_batch in batch_images(image_dir, batch_size=12):  # Optimized batch size for 6GB VRAM
        executor.submit(process_image_batch, image_batch, sample_face_embedding, best_threshold, output_dir)

print(f"Matching images have been copied to {output_dir}.")
# End measuring time
end_time = time.time()
total_time = end_time - start_time
print(f"Process completed in {total_time:.2f} seconds.")
