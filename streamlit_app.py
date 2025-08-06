import streamlit as st
import os
from PIL import Image
from src.encoder import pack_newspaper_page
from src.decoder import unpack_newspaper_page
from src.validator import validate_reconstruction
from src.gtd import GenerativeTokenDictionary
from src.utils import save_json, load_json
import filecmp

# Configuration
DATA_DIR = "data"
ORIGINAL_PAGES_DIR = os.path.join(DATA_DIR, "original_pages")
OCR_TEXT_DIR = os.path.join(DATA_DIR, "ocr_text")
SEEDS_DIR = os.path.join(DATA_DIR, "seeds")
REGENERATED_PAGES_DIR = os.path.join(DATA_DIR, "regenerated_pages")
GTD_FILEPATH = os.path.join(DATA_DIR, "gtd", "gtd_dictionary.json")

# Ensure directories exist
os.makedirs(ORIGINAL_PAGES_DIR, exist_ok=True)
os.makedirs(OCR_TEXT_DIR, exist_ok=True)
os.makedirs(SEEDS_DIR, exist_ok=True)
os.makedirs(REGENERATED_PAGES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(GTD_FILEPATH), exist_ok=True)

# Initialize GTD manager
gtd_manager = GenerativeTokenDictionary(gtd_filepath=GTD_FILEPATH)
gtd_manager.build_gtd_from_dataset([], [])

st.set_page_config(layout="wide", page_title="Soulzip MVR: Lossless Newspaper Compression")

st.title("Soulzip MVR: Lossless Newspaper Compression Prototype 📰")
st.markdown("---")

st.header("1. Upload Original Newspaper Page")
uploaded_image = st.file_uploader("Upload Scanned Newspaper Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
uploaded_text = st.file_uploader("Upload Corresponding OCR Text File (.txt)", type=["txt"])

if uploaded_image and uploaded_text:
    page_name = uploaded_image.name.split('.')[0]
    original_image_path = os.path.join(ORIGINAL_PAGES_DIR, uploaded_image.name)
    original_text_path = os.path.join(OCR_TEXT_DIR, uploaded_text.name)

    # Save uploaded files
    with open(original_image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    with open(original_text_path, "wb") as f:
        f.write(uploaded_text.getbuffer())

    st.success(f"Original page '{page_name}' uploaded successfully!")

    st.header("2. Pack into Soulzip Seed")
    output_seed_filepath = os.path.join(SEEDS_DIR, f"{page_name}.soulzip")
    if st.button("Pack Page into .soulzip"):
        with st.spinner("Packing... This might take a moment."):
            try:
                packed_file_path = pack_newspaper_page(original_image_path, original_text_path, gtd_manager, output_seed_filepath)
                
                with open(packed_file_path, 'rb') as f:
                    compressed_seed_bytes = f.read()

                st.success(f"Page packed into **{os.path.basename(packed_file_path)}**! (Size: {len(compressed_seed_bytes)/1024:.2f} KB)")
                st.download_button(
                    label="Download Soulzip Seed",
                    data=compressed_seed_bytes,
                    file_name=os.path.basename(packed_file_path),
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.error(f"Error during packing: {e}")
                st.exception(e)

st.markdown("---")

st.header("3. Unpack Soulzip Seed & Validate")
st.write("Select a previously packed seed from the `data/seeds/` directory.")

# List available seeds
available_seeds = [f for f in os.listdir(SEEDS_DIR) if f.endswith(".soulzip")] if os.path.exists(SEEDS_DIR) else []
if not available_seeds:
    st.info("No .soulzip seeds found yet. Upload and pack a page first!")
else:
    selected_seed_filename = st.selectbox("Select a .soulzip seed to unpack:", [""] + available_seeds)

    if selected_seed_filename:
        selected_seed_filepath = os.path.join(SEEDS_DIR, selected_seed_filename)
        
        if st.button(f"Unpack '{selected_seed_filename}'"):
            with st.spinner("Unpacking and Validating..."):
                try:
                    output_page_name = selected_seed_filename.replace(".soulzip", "")
                    reconstructed_output_dir = os.path.join(REGENERATED_PAGES_DIR, output_page_name)
                    
                    unpack_results = unpack_newspaper_page(selected_seed_filepath, reconstructed_output_dir)
                    
                    manifest = unpack_results["manifest"]
                    reconstructed_image_path = unpack_results["reconstructed_image_path"]
                    reconstructed_text_path = unpack_results["reconstructed_text_path"]

                    validation_results = validate_reconstruction(manifest, reconstructed_image_path, reconstructed_text_path)

                    st.success("Unpacking and Validation Complete!")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Original Page")
                        original_img_name = manifest.get("image_filename")
                        original_img_path_from_manifest = os.path.join(ORIGINAL_PAGES_DIR, original_img_name)
                        original_txt_name = manifest.get("text_filename")
                        original_txt_path_from_manifest = os.path.join(OCR_TEXT_DIR, original_txt_name)

                        if os.path.exists(original_img_path_from_manifest):
                            st.image(original_img_path_from_manifest, caption="Original Scanned Image", use_column_width=True)
                        else:
                            st.warning(f"Original image not found for display.")
                        
                        if os.path.exists(original_txt_path_from_manifest):
                            with open(original_txt_path_from_manifest, 'r', encoding='utf-8') as f:
                                st.text_area("Original OCR Text", f.read(), height=300)
                        else:
                            st.warning(f"Original text not found for display.")

                    with col2:
                        st.subheader("Reconstructed Page")
                        st.image(reconstructed_image_path, caption="Reconstructed Image", use_column_width=True)
                        with open(reconstructed_text_path, 'r', encoding='utf-8') as f:
                            st.text_area("Reconstructed OCR Text", f.read(), height=300)

                    st.subheader("Validation Results")
                    if validation_results.get("image_hash_match", False) and validation_results.get("text_hash_match", False):
                        st.balloons()
                        st.success("🎉 **ALL HASHES MATCH!** The reconstructed data is identical to the original. Lossless fidelity achieved!")
                    else:
                        st.error("❌ **HASH MISMATCH!** There was an error in reconstruction. Please check logs.")
                    
                    st.json(validation_results)
                    st.write("---")
                    st.subheader("Seed Manifest (for debugging)")
                    st.json(manifest)

                    st.write("---")
                    st.subheader("Direct File Comparison (for MVR testing)")
                    # Direct file comparison for images
                    image_comparison_result = "N/A (Original file not found)"
                    if os.path.exists(original_img_path_from_manifest):
                        if filecmp.cmp(original_img_path_from_manifest, reconstructed_image_path, shallow=False):
                            image_comparison_result = "✅ Images are byte-for-byte identical!"
                        else:
                            image_comparison_result = "❌ Images are NOT byte-for-byte identical!"
                    st.write(f"**Image File Comparison:** {image_comparison_result}")

                    # Direct file comparison for text
                    text_comparison_result = "N/A (Original file not found)"
                    if os.path.exists(original_txt_path_from_manifest):
                        if filecmp.cmp(original_txt_path_from_manifest, reconstructed_text_path, shallow=False):
                            text_comparison_result = "✅ Text files are byte-for-byte identical!"
                        else:
                            text_comparison_result = "❌ Text files are NOT byte-for-byte identical!"
                    st.write(f"**Text File Comparison:** {text_comparison_result}")

                except Exception as e:
                    st.error(f"Error during unpacking or validation: {e}")
                    st.exception(e)