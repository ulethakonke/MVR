# ============ utils.py ============
import os
import json
import hashlib
import base64
from PIL import Image
from typing import Dict, Any, List, Union
import numpy as np
import io

# Make pytesseract optional for cloud deployment
try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False

def save_json(data: Dict[str, Any], filepath: str):
    """Saves a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved data to {filepath}")

def load_json(filepath: str) -> Dict[str, Any]:
    """Loads data from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded data from {filepath}")
    return data

def generate_sha256_hash(data: Union[str, bytes]) -> str:
    """Generates a SHA256 hash for a given string or bytes."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def preprocess_image_for_ocr(img: Image) -> Image:
    """Converts image to grayscale and enhances contrast for better OCR stability."""
    img = img.convert('L')
    img = img.point(lambda x: 0 if x < 128 else 255)
    return img

def consistent_ocr(image_path: str) -> str:
    """Performs OCR on an image file using Tesseract with consistent configuration."""
    if not HAS_PYTESSERACT:
        return "OCR not available in this environment"
    
    try:
        img = Image.open(image_path)
        processed_img = preprocess_image_for_ocr(img)
        config = '--psm 6 --oem 1 -l eng' 
        text = pytesseract.image_to_string(processed_img, config=config)
        print(f"OCR completed for {image_path}")
        return text
    except pytesseract.TesseractNotFoundError:
        print("Tesseract is not installed or not in your PATH.")
        return "Tesseract not found"
    except Exception as e:
        print(f"Error during OCR for {image_path}: {e}")
        return f"OCR Error: {e}"

def extract_image_metadata(image_path: str) -> Dict[str, Any]:
    """Extracts basic metadata from an image file."""
    filename = os.path.basename(image_path)
    parts = filename.split('_')
    publication = parts[0] if len(parts) > 0 else "Unknown"
    date = parts[1] if len(parts) > 1 else "Unknown"
    page_num = "Unknown"
    for part in parts:
        if 'page' in part:
            page_num = part.replace('page', '').split('.')[0]
            break
    
    return {
        "filename": filename,
        "publication": publication,
        "date": date,
        "page_num": page_num,
        "original_path": image_path
    }

def safe_b64_decode(data: str) -> bytes:
    """Safely decodes a base64 string."""
    try:
        return base64.b64decode(data)
    except Exception as e:
        print(f"Base64 decode error: {e}")
        raise ValueError("Invalid base64 data in soulzip seed. Data might be corrupted.")

# ============ gtd.py ============
from typing import List, Dict, Any
import os
import json
from utils import load_json, save_json

class GenerativeTokenDictionary:
    """Manages the Generative Token Dictionary (GTD) for structural archetypes."""
    
    def __init__(self, gtd_filepath: str = 'data/gtd/gtd_dictionary.json'):
        self.gtd_filepath = gtd_filepath
        self.dictionary = self._load_or_initialize_gtd()

    def _load_or_initialize_gtd(self) -> Dict[str, Any]:
        """Loads the GTD from file or initializes an empty one."""
        if os.path.exists(self.gtd_filepath):
            print(f"Loading GTD from {self.gtd_filepath}")
            return load_json(self.gtd_filepath)
        else:
            print("GTD file not found, initializing new GTD.")
            return {
                "layout_archetypes": {},
                "element_archetypes": {}
            }

    def save_gtd(self):
        """Saves the current state of the GTD to file."""
        save_json(self.dictionary, self.gtd_filepath)
        print(f"GTD saved to {self.gtd_filepath}")

    def add_layout_archetype(self, archetype_id: str, description: str, structure: Dict[str, Any]):
        """Adds or updates a layout archetype in the GTD."""
        self.dictionary["layout_archetypes"][archetype_id] = {
            "description": description,
            "structure": structure
        }
        print(f"Added/updated layout archetype: {archetype_id}")

    def get_layout_archetype(self, archetype_id: str) -> Dict[str, Any]:
        """Retrieves a layout archetype by its ID."""
        return self.dictionary["layout_archetypes"].get(archetype_id)

    def add_element_archetype(self, archetype_id: str, description: str, properties: Dict[str, Any]):
        """Adds or updates an element archetype."""
        self.dictionary["element_archetypes"][archetype_id] = {
            "description": description,
            "properties": properties
        }
        print(f"Added/updated element archetype: {archetype_id}")

    def get_element_archetype(self, archetype_id: str) -> Dict[str, Any]:
        """Retrieves an element archetype by its ID."""
        return self.dictionary["element_archetypes"].get(archetype_id)

    def build_gtd_from_dataset(self, image_paths: List[str], ocr_text_paths: List[str]):
        """Analyzes dataset to identify and add structural GTD archetypes."""
        print("Starting GTD analysis (MVR simplified for structural archetypes).")

        # Initialize basic archetypes
        if not self.get_layout_archetype("LAYOUT_FRONT_PAGE_1920S_A"):
            self.add_layout_archetype(
                "LAYOUT_FRONT_PAGE_1920S_A",
                "Common 1920s front page layout with large masthead and multi-column articles.",
                {"masthead_area": [0,0,1,0.1], "main_article_area": [0,0.1,0.7,0.9]}
            )
        if not self.get_layout_archetype("LAYOUT_INNER_PAGE_TEXT_HEAVY_B"):
            self.add_layout_archetype(
                "LAYOUT_INNER_PAGE_TEXT_HEAVY_B",
                "Standard inner page with 4 text columns and small ads/images.",
                {"column_count": 4, "ad_slots": [[0.8,0.1,0.9,0.2]]}
            )

        if not self.get_element_archetype("ELEMENT_HEADLINE_LARGE_BOLD"):
            self.add_element_archetype(
                "ELEMENT_HEADLINE_LARGE_BOLD",
                "Large, bold headline style typical of front pages.",
                {"font_size_range": [36, 48], "font_weight": "bold"}
            )
        if not self.get_element_archetype("ELEMENT_ARTICLE_TEXT_BLOCK"):
            self.add_element_archetype(
                "ELEMENT_ARTICLE_TEXT_BLOCK",
                "Standard body text block for articles.",
                {"font_size_range": [9, 11], "line_height": 1.2}
            )
        if not self.get_element_archetype("ELEMENT_PHOTO_SQUARE_B&W"):
            self.add_element_archetype(
                "ELEMENT_PHOTO_SQUARE_B&W",
                "Typical square black and white news photo.",
                {"aspect_ratio": "1:1", "color_mode": "grayscale"}
            )

        self.save_gtd()
        print("GTD analysis complete. Basic structural archetypes added.")

# ============ layout_tools.py ============
from typing import Dict, Any, List
from PIL import Image

def analyze_page_layout(image_path: str, ocr_text_path: str, gtd_manager: Any) -> Dict[str, Any]:
    """Analyzes the layout of a newspaper page (simplified placeholder for MVR)."""
    print(f"Analyzing layout for {image_path} (MVR placeholder).")

    layout_archetype_id = "LAYOUT_FRONT_PAGE_1920S_A"
    
    elements_data = [
        {
            "type": "headline",
            "gtd_archetype_id": "ELEMENT_HEADLINE_LARGE_BOLD",
            "position": [0.1, 0.05, 0.9, 0.15],
            "content_ref": "headline_1"
        },
        {
            "type": "article",
            "gtd_archetype_id": "ELEMENT_ARTICLE_TEXT_BLOCK",
            "position": [0.05, 0.2, 0.45, 0.7],
            "content_ref": "article_1"
        },
        {
            "type": "image",
            "gtd_archetype_id": "ELEMENT_PHOTO_SQUARE_B&W",
            "position": [0.5, 0.25, 0.75, 0.45],
            "content_ref": "image_1"
        }
    ]

    if not gtd_manager.get_layout_archetype(layout_archetype_id):
        print(f"Warning: Layout archetype {layout_archetype_id} not found in GTD.")

    return {
        "layout_archetype_id": layout_archetype_id,
        "elements_data": elements_data
    }

# ============ encoder.py ============
import os
import json
import zstandard as zstd
import base64
import io
from typing import Dict, Any, Union
from utils import generate_sha256_hash, consistent_ocr, extract_image_metadata
from layout_tools import analyze_page_layout
from gtd import GenerativeTokenDictionary

CHUNK_SIZE = 1024 * 1024 

def pack_newspaper_page(image_path: str, ocr_text_path: str, gtd_manager: GenerativeTokenDictionary, output_filepath: str) -> str:
    """Packs newspaper page into .soulzip file using Zstandard streaming."""
    print(f"Starting packing for image: {image_path}, text: {ocr_text_path}")

    # Load original data
    with open(image_path, 'rb') as f:
        original_image_bytes = f.read()
    with open(ocr_text_path, 'r', encoding='utf-8') as f:
        original_text_content = f.read()

    # Extract structural metadata
    page_layout_info = analyze_page_layout(image_path, ocr_text_path, gtd_manager)

    # Compute SHA256 hashes for integrity
    image_hash = generate_sha256_hash(original_image_bytes)
    text_hash = generate_sha256_hash(original_text_content)

    # Assemble the structured manifest
    seed_manifest = {
        "version": "1.0",
        "soulzip_spec": "lossless_v1",
        "page_metadata": extract_image_metadata(image_path),
        "image_filename": os.path.basename(image_path),
        "text_filename": os.path.basename(ocr_text_path),
        "image_hash": image_hash,
        "text_hash": text_hash,
        "layout_info": page_layout_info
    }
    
    # Bundle data
    bundle_data = {
        "manifest": seed_manifest,
        "original_image_b64": base64.b64encode(original_image_bytes).decode('utf-8'),
        "original_text_b64": base64.b64encode(original_text_content.encode('utf-8')).decode('utf-8')
    }
    
    serialized_bundle = json.dumps(bundle_data, ensure_ascii=False).encode('utf-8')
    original_bundle_size = len(serialized_bundle)

    # Compress using zstd streaming
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    cctx = zstd.ZstdCompressor(level=19)
    
    compressed_size = 0
    with open(output_filepath, 'wb') as f_out:
        with cctx.stream_writer(f_out) as compressor:
            for i in range(0, len(serialized_bundle), CHUNK_SIZE):
                chunk = serialized_bundle[i:i + CHUNK_SIZE]
                compressor.write(chunk)
        compressed_size = f_out.tell()

    print(f"Original bundle size: {original_bundle_size/1024:.2f} KB")
    print(f"Compressed seed size: {compressed_size/1024:.2f} KB")
    if compressed_size > 0:
        print(f"Compression ratio: {original_bundle_size / compressed_size:.2f}:1")
    
    return output_filepath

# ============ decoder.py ============
import os
import zstandard as zstd
import json
import base64
import io
from typing import Dict, Any
from utils import safe_b64_decode

CHUNK_SIZE = 1024 * 1024 

def unpack_newspaper_page(compressed_seed_filepath: str, output_dir: str) -> Dict[str, Any]:
    """Decompresses a .soulzip file and reconstructs the original files."""
    print(f"Starting unpacking process for {compressed_seed_filepath}...")

    # Decompress the seed using zstd streaming
    dctx = zstd.ZstdDecompressor()
    decompressed_data_buffer = io.BytesIO()

    with open(compressed_seed_filepath, 'rb') as f_in:
        with dctx.stream_reader(f_in) as reader:
            while True:
                chunk = reader.read(CHUNK_SIZE)
                if not chunk:
                    break
                decompressed_data_buffer.write(chunk)
    
    decompressed_bundle_bytes = decompressed_data_buffer.getvalue()
    
    # Deserialize the bundle
    bundle_data = json.loads(decompressed_bundle_bytes.decode('utf-8'))

    # Decode base64 back to bytes
    original_image_bytes = safe_b64_decode(bundle_data["original_image_b64"])
    original_text_content = safe_b64_decode(bundle_data["original_text_b64"]).decode('utf-8')

    manifest = bundle_data["manifest"]

    # Reconstruct original files
    os.makedirs(output_dir, exist_ok=True)
    
    image_filename = os.path.basename(manifest["image_filename"])
    text_filename = os.path.basename(manifest["text_filename"])

    reconstructed_image_path = os.path.join(output_dir, image_filename)
    reconstructed_text_path = os.path.join(output_dir, text_filename)

    with open(reconstructed_image_path, 'wb') as f:
        f.write(original_image_bytes)
    print(f"Reconstructed image saved to: {reconstructed_image_path}")

    with open(reconstructed_text_path, 'w', encoding='utf-8') as f:
        f.write(original_text_content)
    print(f"Reconstructed text saved to: {reconstructed_text_path}")

    return {
        "manifest": manifest,
        "reconstructed_image_path": reconstructed_image_path,
        "reconstructed_text_path": reconstructed_text_path
    }

# ============ validator.py ============
import os
from utils import generate_sha256_hash
from typing import Dict, Any

def validate_reconstruction(manifest: Dict[str, Any], reconstructed_image_path: str, reconstructed_text_path: str) -> Dict[str, bool]:
    """Validates the reconstructed files against the hashes stored in the manifest."""
    validation_results = {}

    # Validate image hash
    with open(reconstructed_image_path, 'rb') as f:
        reconstructed_image_bytes = f.read()
    reconstructed_image_hash = generate_sha256_hash(reconstructed_image_bytes)
    validation_results["image_hash_match"] = (reconstructed_image_hash == manifest["image_hash"])

    # Validate text hash
    with open(reconstructed_text_path, 'r', encoding='utf-8') as f:
        reconstructed_text_content = f.read()
    reconstructed_text_hash = generate_sha256_hash(reconstructed_text_content)
    validation_results["text_hash_match"] = (reconstructed_text_hash == manifest["text_hash"])

    print(f"Validation results: {validation_results}")
    return validation_results