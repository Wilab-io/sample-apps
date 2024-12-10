import os
import json
import hashlib
import torch
import numpy as np
from dotenv import load_dotenv
import logging
from pydantic import BaseModel
from vespa.package import (
    ApplicationPackage,
    AuthClient,
    Parameter,
    RankProfile,
)
from vespa.configuration.services import (
    services,
    container,
    search,
    document_api,
    document_processing,
    clients,
    client,
    config,
    content,
    redundancy,
    documents,
    node,
    certificate,
    token,
    document,
    nodes,
)
from vespa.configuration.vt import vt
from vespa.package import ServicesConfiguration
from backend.models import UserSettings

# Google Generative AI
import google.generativeai as genai

# Torch and other ML libraries
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pdf2image import convert_from_path
from pypdf import PdfReader

# ColPali model and processor
from colpali_engine.models import ColPali, ColPaliProcessor
from vidore_benchmark.utils.image_utils import scale_image, get_base64_image

# Other utilities
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from PIL import Image
import pytesseract

import pty
import subprocess
import re
import time
import select

from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("vespa_app")

async def deploy_application_step_1(settings: UserSettings):
    logger.info("Validating settings")
    if not all([
        settings.tenant_name,
        settings.app_name,
        settings.vespa_token_id,
        settings.vespa_token_value,
        settings.gemini_token
    ]):
        raise ValueError("Missing required settings")

    VESPA_TENANT_NAME = settings.tenant_name
    VESPA_APPLICATION_NAME = settings.app_name

    logger.info(f"Deploying application {VESPA_APPLICATION_NAME} to tenant {VESPA_TENANT_NAME}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(base_dir))
    app_dir = os.path.join(parent_dir, "application")

    logger.info(f"Running vespa commands on the application directory: {app_dir}")

    current_dir = os.getcwd()

    try:
        os.chdir(app_dir)

        subprocess.run(["vespa", "config", "set", "target", "cloud"], check=True)

        app_config = f"{VESPA_TENANT_NAME}.{VESPA_APPLICATION_NAME}"
        subprocess.run(["vespa", "config", "set", "application", app_config], check=True)

        subprocess.run(["vespa", "auth", "cert", "-f"], check=True)

        def run_auth_login():
            master, slave = pty.openpty()
            process = subprocess.Popen(
                ["vespa", "auth", "login"],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                text=True,
                cwd=app_dir
            )

            output = ""
            answered_prompt = False

            while True:
                r, _, _ = select.select([master], [], [], 0.1)
                if r:
                    try:
                        data = os.read(master, 1024).decode()
                        output += data

                        if not answered_prompt and "Automatically open confirmation page in your default browser? [Y/n]" in output:
                            os.write(master, "n\n".encode())
                            answered_prompt = True

                        # Once we find the URL, we can return
                        if "Please open link in your browser: " in data:
                            # Kill the process since we don't need to wait for completion
                            process.kill()
                            os.close(master)
                            os.close(slave)
                            return output

                    except OSError:
                        break

                # If process ends before we find URL, that's an error
                if process.poll() is not None:
                    break

            # Clean up
            process.kill()
            os.close(master)
            os.close(slave)
            return output

        # Create and start daemon thread for auth login
        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_auth_login)
            output = future.result()

        logger.debug(f"Full output: {output}")

        url_match = re.search(r'Please open link in your browser: (https://[^\s]+)', output)
        if not url_match:
            raise Exception("Could not find authentication URL in command output")

        auth_url = url_match.group(1)
        logger.info(f"Authentication URL found: {auth_url}")

        # Load certificate files
        cert_dir = os.path.expanduser(f"~/.vespa/{VESPA_TENANT_NAME}.{VESPA_APPLICATION_NAME}.default")
        logger.debug(f"Looking for certificates in: {cert_dir}")

        private_key_path = os.path.join(cert_dir, "data-plane-private-key.pem")
        public_cert_path = os.path.join(cert_dir, "data-plane-public-cert.pem")

        # Wait a bit for files to be created
        max_retries = 10
        retry_count = 0
        while retry_count < max_retries:
            if os.path.exists(private_key_path) and os.path.exists(public_cert_path):
                break
            time.sleep(1)
            retry_count += 1
            logger.debug(f"Waiting for certificate files... attempt {retry_count}/{max_retries}")

        if not os.path.exists(private_key_path) or not os.path.exists(public_cert_path):
            raise FileNotFoundError(f"Certificate files not found in {cert_dir}")

        # Read certificate files
        with open(private_key_path, 'r') as f:
            private_key = f.read()
        with open(public_cert_path, 'r') as f:
            public_cert = f.read()

        # Set environment variables
        os.environ["VESPA_CLOUD_MTLS_KEY"] = private_key
        os.environ["VESPA_CLOUD_MTLS_CERT"] = public_cert

        logger.info("Successfully loaded certificate files")

        return {"status": "success", "auth_url": auth_url}

    except Exception as e:
        raise
    finally:
        os.chdir(current_dir)

async def deploy_application_step_2(request, settings: UserSettings, user_id: str, model: ColPali, processor: ColPaliProcessor, docNames: dict[str, str]):
    """Deploy the Vespa application"""
    # Load environment variables
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    VESPA_APPLICATION_NAME = settings.app_name
    VESPA_SCHEMA_NAME = "pdf_page"
    VESPA_TOKEN_ID_WRITE = settings.vespa_token_id
    GEMINI_API_KEY = settings.gemini_token

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(base_dir))
    storage_dir = os.path.join(parent_dir, "src/storage/user_documents", user_id)
    app_dir = os.path.join(parent_dir, "application")

    try:
        logger.debug("Generating services.xml")

        service_config = ServicesConfiguration(
            application_name=VESPA_APPLICATION_NAME,
            services_config=services(
                container(
                    search(),
                    document_api(),
                    document_processing(),
                    clients(
                        client(
                            certificate(file="security/clients.pem"),
                            id="mtls",
                            permissions="read,write",
                        ),
                        client(
                            token(id=f"{VESPA_TOKEN_ID_WRITE}"),
                            id="token_write",
                            permissions="read,write",
                        ),
                    ),
                    config(
                        vt("tag")(
                            vt("bold")(
                                vt("open", "<strong>"),
                                vt("close", "</strong>"),
                            ),
                            vt("separator", "..."),
                        ),
                        name="container.qr-searchers",
                    ),
                    id=f"{VESPA_APPLICATION_NAME}_container",
                    version="1.0",
                ),
                content(
                    redundancy("1"),
                    documents(document(type="pdf_page", mode="index")),
                    nodes(node(distribution_key="0", hostalias="node1")),
                    config(
                        vt("max_matches", "2", replace_underscores=False),
                        vt("length", "1000"),
                        vt("surround_max", "500", replace_underscores=False),
                        vt("min_length", "300", replace_underscores=False),
                        name="vespa.config.search.summary.juniperrc",
                    ),
                    id=f"{VESPA_APPLICATION_NAME}_content",
                    version="1.0",
                ),
                version="1.0",
            ),
        )

        # Create the Vespa application package and save it to services.xml
        vespa_application_package = ApplicationPackage(
            name=VESPA_APPLICATION_NAME,
            services_config=service_config,
            auth_clients=[
                AuthClient(
                    id="mtls",
                    permissions="read,write",
                    parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
                ),
            ],
        )

        servicesXml = vespa_application_package.services_to_text
        services_xml_path = os.path.join(app_dir, "services.xml")
        with open(services_xml_path, "w") as f:
            f.write(servicesXml)

        # Write the schema to its file
        from pathlib import Path

        schema_dir = Path(parent_dir) / "application/schemas"
        schema_file = schema_dir / "pdf_page.sd"

        logger.debug(f"Writing schema to {schema_file}")

        schema_file.write_text(settings.schema)

        import time
        time.sleep(2)

        import subprocess

        # Store current directory
        current_dir = os.getcwd()

        try:
            logger.debug("Running deploy commands")

            # Change to application directory
            os.chdir(app_dir)

            process = subprocess.Popen(
                ["vespa", "deploy", "--wait", "500"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            endpoint_url = ""
            for line in iter(process.stdout.readline, ''):
                logger.info(line.strip())
                if "Found endpoints:" in line:
                    # Read the next two lines to get to the URL line
                    next(process.stdout)  # Skip "- dev.aws-us-east-*" line
                    url_line = next(process.stdout)
                    # Extract URL from line like " |-- https://d110fb1d.f78833a9.z.vespa-app.cloud/ (cluster '*_container')"
                    match = re.search(r'https://[^\s]+', url_line)
                    if match:
                        endpoint_url = match.group(0)
                        logger.info(f"Found endpoint URL: {endpoint_url}")

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, "vespa deploy")

            if not endpoint_url:
                raise Exception("Failed to find endpoint URL in deployment output")

            # Save endpoint_url to the database
            await request.app.db.update_settings(user_id, {"vespa_cloud_endpoint": endpoint_url})

            logger.info(f"Deployment completed successfully! Endpoint URL: {endpoint_url}")
        finally:
            # Always restore the original working directory
            os.chdir(current_dir)

        # Configure Google Generative AI
        genai.configure(api_key=GEMINI_API_KEY)

        logger.info(f"Looking for documents in: {storage_dir}")

        if not os.path.exists(storage_dir):
            raise FileNotFoundError(f"Directory not found: {storage_dir}")

        pdfPaths = [
            os.path.join(storage_dir, f)
            for f in os.listdir(storage_dir)
            if f.endswith(".pdf")
        ]

        imgPaths = [
            os.path.join(storage_dir, f)
            for f in os.listdir(storage_dir)
            if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")
        ]

        if not pdfPaths and not imgPaths:
            raise FileNotFoundError(f"No PDF or image files found in {storage_dir}")

        logger.info(f"Found {len(pdfPaths)} PDF files and {len(imgPaths)} image files to process")

        pdf_pages = []

        # Process PDFs
        for pdf_file in pdfPaths:
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

        # Process Images
        for img_file in imgPaths:
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

        logger.info(f"Total processed: {len(pdf_pages)} pages")

        prompt_text, pydantic_model = settings.prompt, GeneratedQueries

        logger.debug(f"Generating queries")

        for pdf in tqdm(pdf_pages):
            image = pdf.get("image")
            pdf["queries"] = generate_queries(image, prompt_text, pydantic_model)

        images = [pdf["image"] for pdf in pdf_pages]
        embeddings = generate_embeddings(images, model, processor)

        logger.info(f"Generated {len(embeddings)} embeddings")

        vespa_feed = []
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
            )  # Scaled down image to return fast on search (~1kb)
            base_64_full_image = get_base64_image(image, add_url_prefix=False)
            embedding_dict = {k: v for k, v in enumerate(embedding)}
            binary_embedding = float_to_binary_embedding(embedding_dict)
            # id_hash should be md5 hash of url and page_number
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

        with open(app_dir + "/vespa_feed.json", "w") as f:
            vespa_feed_to_save = []
            for page in vespa_feed:
                document_id = page["id"]
                put_id = f"id:{VESPA_APPLICATION_NAME}:{VESPA_SCHEMA_NAME}::{document_id}"
                vespa_feed_to_save.append({"put": put_id, "fields": page["fields"]})
            json.dump(vespa_feed_to_save, f)

        logger.debug(f"Saved vespa feed to {app_dir}/vespa_feed.json")

        current_dir = os.getcwd()

        try:
            logger.debug("Feeding vespa application")

            # Change to application directory
            os.chdir(app_dir)

            subprocess.run(["vespa", "feed", "vespa_feed.json", "--progress", "5"], check=True)

            logger.info(f"Feeding completed successfully!")
        finally:
            # Always restore the original working directory
            os.chdir(current_dir)

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

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

def float_to_binary_embedding(float_query_embedding: dict) -> dict:
    """Utility function to convert float query embeddings to binary query embeddings."""
    binary_query_embeddings = {}
    for k, v in float_query_embedding.items():
        binary_vector = (
            np.packbits(np.where(np.array(v) > 0, 1, 0)).astype(np.int8).tolist()
        )
        binary_query_embeddings[k] = binary_vector
    return binary_query_embeddings

# A function to create an inherited rank profile which also returns quantized similarity scores
def with_quantized_similarity(rank_profile: RankProfile) -> RankProfile:
    return RankProfile(
        name=f"{rank_profile.name}_sim",
        first_phase=rank_profile.first_phase,
        inherits=rank_profile.name,
        summary_features=["quantized"],
    )
