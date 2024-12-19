import os
import json
import hashlib
import subprocess
import logging
import numpy as np
from tqdm import tqdm
from backend.models import UserSettings
from pydantic import BaseModel
import google.generativeai as genai
import torch
from torch.utils.data import DataLoader

from pdf2image import convert_from_path
from pypdf import PdfReader
from PIL import Image
import pytesseract

# ColPali model and processor
from colpali_engine.models import ColPali, ColPaliProcessor
from vidore_benchmark.utils.image_utils import scale_image, get_base64_image

logger = logging.getLogger("vespa_app")



def feed_documents_to_vespa(settings: UserSettings, user_id: str, model: ColPali, processor: ColPaliProcessor, docNames: dict[str, str]):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(base_dir))
        storage_dir = os.path.join(parent_dir, "src/storage/user_documents", user_id)
        app_dir = os.path.join(parent_dir, "application")

        VESPA_TENANT_NAME = settings.tenant_name
        VESPA_APPLICATION_NAME = settings.app_name
        VESPA_INSTANCE_NAME = settings.instance_name
        VESPA_SCHEMA_NAME = "pdf_page"
        GEMINI_API_KEY = settings.gemini_token

        # Configure Google Generative AI
        genai.configure(api_key=GEMINI_API_KEY)

        logger.info(f"Looking for documents in: {storage_dir}")

        if not os.path.exists(storage_dir):
            return {"status": "error", "message": f"Directory not found: {storage_dir}"}

        pdfPaths = [
            os.path.join(storage_dir, f)
            for f in os.listdir(storage_dir)
            if f.endswith(".pdf") and os.path.splitext(f)[0] in docNames
        ]

        imgPaths = [
            os.path.join(storage_dir, f)
            for f in os.listdir(storage_dir)
            if (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg"))
            and os.path.splitext(f)[0] in docNames
        ]

        if not pdfPaths and not imgPaths:
            return {"status": "error", "message": f"No PDF or image files found in {storage_dir}"}

        logger.info(f"Found {len(pdfPaths)} new PDF files and {len(imgPaths)} new image files to process")

        pdf_pages = []

        # Process PDFs
        for pdf_file in pdfPaths:
            try:
                logger.debug(f"Processing PDF: {os.path.basename(pdf_file)}")
                images, texts = get_pdf_images(pdf_file)
                logger.debug(f"Extracted {len(images)} pages from {os.path.basename(pdf_file)}")
                for page_no, (image, text) in enumerate(zip(images, texts)):
                    doc_id = os.path.splitext(os.path.basename(pdf_file))[0]
                    title = docNames.get(doc_id, "")
                    static_path = f"/storage/user_documents/{user_id}/{os.path.basename(pdf_file)}"
                    pdf_pages.append(
                        {
                            "title": title,
                            "path": pdf_file,
                            "url": static_path,
                            "image": image,
                            "text": text,
                            "page_no": page_no,
                        }
                    )
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_file}: {str(e)}")
                return {"status": "error", "message": f"Error processing PDF {os.path.basename(pdf_file)}: {str(e)}"}

        # Process Images
        for img_file in imgPaths:
            try:
                logger.debug(f"Processing image: {os.path.basename(img_file)}")
                images, texts = get_image_with_text(img_file)
                logger.debug(f"Extracted text from {os.path.basename(img_file)}")
                for page_no, (image, text) in enumerate(zip(images, texts)):
                    doc_id = os.path.splitext(os.path.basename(img_file))[0]
                    title = docNames.get(doc_id, "")
                    static_path = f"/storage/user_documents/{user_id}/{os.path.basename(img_file)}"
                    pdf_pages.append(
                        {
                            "title": title,
                            "path": img_file,
                            "url": static_path,
                            "image": image,
                            "text": text,
                            "page_no": page_no,
                        }
                    )
            except Exception as e:
                logger.error(f"Error processing image {img_file}: {str(e)}")
                return {"status": "error", "message": f"Error processing image {os.path.basename(img_file)}: {str(e)}"}

        logger.info(f"Total processed: {len(pdf_pages)} pages")

        prompt_text, pydantic_model = settings.prompt, GeneratedQueries

        logger.debug(f"Generating queries")

        try:
            for pdf in tqdm(pdf_pages):
                image = pdf.get("image")
                pdf["queries"] = generate_queries(image, prompt_text, pydantic_model)
        except Exception as e:
            logger.error(f"Error generating queries: {str(e)}")
            return {"status": "error", "message": f"Error generating queries: {str(e)}"}

        try:
            images = [pdf["image"] for pdf in pdf_pages]
            embeddings = generate_embeddings(images, model, processor)
            logger.info(f"Generated {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return {"status": "error", "message": f"Error generating embeddings: {str(e)}"}

        vespa_feed = []
        try:
            for pdf, embedding in zip(pdf_pages, embeddings):
                title = pdf["title"]
                image = pdf["image"]
                url = pdf["url"]
                text = pdf.get("text", "")
                page_no = pdf["page_no"]
                query_dict = pdf["queries"]
                questions = [v for k, v in query_dict.items() if "question" in k and v]
                queries = [v for k, v in query_dict.items() if "query" in k and v]
                base_64_image = get_base64_image(
                    scale_image(image, 32), add_url_prefix=False
                )
                base_64_full_image = get_base64_image(image, add_url_prefix=False)
                embedding_dict = {k: v for k, v in enumerate(embedding)}
                binary_embedding = float_to_binary_embedding(embedding_dict)
                id_hash = hashlib.md5(f"{title}_{page_no}".encode()).hexdigest()
                page = {
                    "id": id_hash,
                    "fields": {
                        "id": id_hash,
                        "title": title,
                        "url": url,
                        "page_number": page_no,
                        "blur_image": base_64_image,
                        "full_image": base_64_full_image,
                        "text": text,
                        "embedding": binary_embedding,
                        "queries": queries,
                        "questions": questions,
                    },
                }
                vespa_feed.append(page)
        except Exception as e:
            logger.error(f"Error preparing Vespa feed: {str(e)}")
            return {"status": "error", "message": f"Error preparing Vespa feed: {str(e)}"}

        try:
            with open(app_dir + "/vespa_feed.json", "w") as f:
                vespa_feed_to_save = []
                for page in vespa_feed:
                    document_id = page["id"]
                    put_id = f"id:{VESPA_APPLICATION_NAME}:{VESPA_SCHEMA_NAME}::{document_id}"
                    vespa_feed_to_save.append({"put": put_id, "fields": page["fields"]})
                json.dump(vespa_feed_to_save, f)

            logger.debug(f"Saved vespa feed to {app_dir}/vespa_feed.json")
        except Exception as e:
            logger.error(f"Error saving Vespa feed file: {str(e)}")
            return {"status": "error", "message": f"Error saving Vespa feed file: {str(e)}"}

        current_dir = os.getcwd()

        try:
            logger.debug("Feeding vespa application")
            os.chdir(app_dir)
            result = subprocess.run(
                ["vespa", "feed", "vespa_feed.json", "-a", f"{VESPA_TENANT_NAME}.{VESPA_APPLICATION_NAME}.{VESPA_INSTANCE_NAME}"],
                check=True,
                capture_output=True,
                text=True
            )

            # Check for specific error messages in the output
            if "no endpoints found" in result.stderr.lower():
                logger.error("No Vespa endpoints found. The application might not be deployed or accessible.")
                return {
                    "status": "error",
                    "message": "No Vespa endpoints found. Please make sure the application is deployed and accessible."
                }

            logger.info(f"Feeding completed successfully!")
            return {"status": "success"}
        except subprocess.CalledProcessError as e:
            error_output = e.stderr if e.stderr else e.stdout
            logger.error(f"Error feeding Vespa: {error_output}")

            if "no endpoints found" in str(error_output).lower():
                return {
                    "status": "error",
                    "message": "No Vespa endpoints found. Please make sure the application is deployed and accessible."
                }

            return {
                "status": "error",
                "message": f"Error feeding Vespa: {error_output}"
            }
        finally:
            os.chdir(current_dir)

    except Exception as e:
        logger.error(f"Unexpected error in feed_documents_to_vespa: {str(e)}")
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

def get_pdf_images(pdf_path):
    reader = PdfReader(pdf_path)
    page_texts = []
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text = page.extract_text()
        page_texts.append(text)
    images = convert_from_path(pdf_path)
    # Convert to PIL images
    assert len(images) == len(page_texts)
    return images, page_texts

def float_to_binary_embedding(float_query_embedding: dict) -> dict:
    """Utility function to convert float query embeddings to binary query embeddings."""
    binary_query_embeddings = {}
    for k, v in float_query_embedding.items():
        binary_vector = (
            np.packbits(np.where(np.array(v) > 0, 1, 0)).astype(np.int8).tolist()
        )
        binary_query_embeddings[k] = binary_vector
    return binary_query_embeddings

def get_image_with_text(image_path):
    """Process a single image file and extract its text using OCR"""
    try:
        # Open and process image
        image = Image.open(image_path)

        # Extract text using OCR
        text = pytesseract.image_to_string(image)

        # Return tuple of image and text (similar to get_pdf_images format)
        return [image], [text]
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        raise

class GeneratedQueries(BaseModel):
    broad_topical_question: str
    broad_topical_query: str
    specific_detail_question: str
    specific_detail_query: str
    visual_element_question: str
    visual_element_query: str

def generate_queries(image, prompt_text, pydantic_model):
    gemini_model = genai.GenerativeModel("gemini-1.5-flash-8b")

    try:
        response = gemini_model.generate_content(
            [image, "\n\n", prompt_text],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=pydantic_model,
            ),
        )
        queries = json.loads(response.text)
    except Exception as _e:
        queries = {
            "broad_topical_question": "",
            "broad_topical_query": "",
            "specific_detail_question": "",
            "specific_detail_query": "",
            "visual_element_question": "",
            "visual_element_query": "",
        }
    return queries

def generate_embeddings(images, model, processor, batch_size=1) -> np.ndarray:
    """
    Generate embeddings for a list of images.
    Move to CPU only once per batch.

    Args:
        images (List[PIL.Image]): List of PIL images.
        model (nn.Module): The model to generate embeddings.
        processor: The processor to preprocess images.
        batch_size (int, optional): Batch size for processing. Defaults to 64.

    Returns:
        np.ndarray: Embeddings for the images, shape
                    (len(images), processor.max_patch_length (1030 for ColPali), model.config.hidden_size (Patch embedding dimension - 128 for ColPali)).
    """

    def collate_fn(batch):
        # Batch is a list of images
        return processor.process_images(batch)  # Should return a dict of tensors

    dataloader = DataLoader(
        images,
        shuffle=False,
        collate_fn=collate_fn,
    )

    embeddings_list = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            embeddings_batch = model(**batch)
            # Convert tensor to numpy array and append to list
            embeddings_list.extend(
                [t.cpu().numpy() for t in torch.unbind(embeddings_batch)]
            )

    # Stack all embeddings into a single numpy array
    all_embeddings = np.stack(embeddings_list, axis=0)
    return all_embeddings

def remove_document_from_vespa(settings: UserSettings, document_id: str):
    """Remove a document from Vespa by ID"""
    logger = logging.getLogger("vespa_app")
    logger.info(f"Removing document {document_id} from Vespa")

    VESPA_TENANT_NAME = settings.tenant_name
    VESPA_APPLICATION_NAME = settings.app_name
    VESPA_INSTANCE_NAME = settings.instance_name
    VESPA_SCHEMA_NAME = "pdf_page"

    # Get application directory path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(base_dir))
    app_dir = os.path.join(parent_dir, "application")

    # Store current directory
    current_dir = os.getcwd()

    try:
        logger.debug("Removing document from Vespa")

        # Change to application directory
        os.chdir(app_dir)

        # Construct the full document ID
        vespa_doc_id = f"id:{VESPA_APPLICATION_NAME}:{VESPA_SCHEMA_NAME}::{document_id}"

        # Run the remove command
        subprocess.run(
            ["vespa", "document", "remove", vespa_doc_id, "-a", f"{VESPA_TENANT_NAME}.{VESPA_APPLICATION_NAME}.{VESPA_INSTANCE_NAME}"],
            check=True
        )

        logger.info(f"Successfully removed document {document_id} from Vespa")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to remove document from Vespa: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error removing document from Vespa: {str(e)}")
        raise
    finally:
        # Always restore the original working directory
        os.chdir(current_dir)
