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
    Field,
    Schema,
    Document,
    HNSW,
    RankProfile,
    Function,
    FieldSet,
    SecondPhaseRanking,
    Summary,
    DocumentSummary,
)
from vespa.deployment import VespaCloud
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
from colpali_engine.utils.torch_utils import get_torch_device
from vidore_benchmark.utils.image_utils import scale_image, get_base64_image

# Other utilities
from bs4 import BeautifulSoup
import httpx
from urllib.parse import urljoin, urlparse

from PIL import Image
import pytesseract

logger = logging.getLogger("vespa_app")

async def deploy_application(request, settings: UserSettings, user_id: str, model: ColPali, processor: ColPaliProcessor, docNames: dict[str, str]):
    """Deploy the Vespa application"""
    try:
        logger.info("Starting deployment process")

        # Load environment variables
        load_dotenv()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        VESPA_SCHEMA_NAME = "pdf_page"
        VESPA_TOKEN_ID_WRITE = settings.vespa_token_id
        VESPA_TEAM_API_KEY = None
        GEMINI_API_KEY = settings.gemini_token

        # Configure Google Generative AI
        genai.configure(api_key=GEMINI_API_KEY)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        storage_dir = os.path.join(base_dir, "storage/user_documents", user_id)

        logger.info(f"Looking for PDFs in: {storage_dir}")

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
            logger.info(f"Processing PDF: {os.path.basename(pdf_file)}")
            images, texts = get_pdf_images(pdf_file)
            logger.info(f"Extracted {len(images)} pages from {os.path.basename(pdf_file)}")
            for page_no, (image, text) in enumerate(zip(images, texts)):
                doc_id = os.path.splitext(os.path.basename(pdf_file))[0]
                title = docNames.get(doc_id, "")
                pdf_pages.append(
                    {
                        "title": title,
                        "path": pdf_file,
                        "image": image,
                        "text": text,
                        "page_no": page_no,
                    }
                )

        # Process Images
        for img_file in imgPaths:
            logger.info(f"Processing image: {os.path.basename(img_file)}")
            images, texts = get_image_with_text(img_file)
            logger.info(f"Extracted text from {os.path.basename(img_file)}")
            for page_no, (image, text) in enumerate(zip(images, texts)):
                doc_id = os.path.splitext(os.path.basename(img_file))[0]
                title = docNames.get(doc_id, "")
                pdf_pages.append(
                    {
                        "title": title,
                        "path": img_file,
                        "image": image,
                        "text": text,
                        "page_no": page_no,
                    }
                )

        logger.info(f"Total processed: {len(pdf_pages)} pages")

        prompt_text, pydantic_model = settings.prompt, GeneratedQueries

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

        # Define the Vespa schema
        colpali_schema = Schema(
            name=VESPA_SCHEMA_NAME,
            document=Document(
                fields=[
                    Field(
                        name="id",
                        type="string",
                        indexing=["summary", "index"],
                        match=["word"],
                    ),
                    Field(name="url", type="string", indexing=["summary", "index"]),
                    Field(name="year", type="int", indexing=["summary", "attribute"]),
                    Field(
                        name="title",
                        type="string",
                        indexing=["summary", "index"],
                        match=["text"],
                        index="enable-bm25",
                    ),
                    Field(name="page_number", type="int", indexing=["summary", "attribute"]),
                    Field(name="blur_image", type="raw", indexing=["summary"]),
                    Field(name="full_image", type="raw", indexing=["summary"]),
                    Field(
                        name="text",
                        type="string",
                        indexing=["summary", "index"],
                        match=["text"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="embedding",
                        type="tensor<int8>(patch{}, v[16])",
                        indexing=[
                            "attribute",
                            "index",
                        ],
                        ann=HNSW(
                            distance_metric="hamming",
                            max_links_per_node=32,
                            neighbors_to_explore_at_insert=400,
                        ),
                    ),
                    Field(
                        name="questions",
                        type="array<string>",
                        indexing=["summary", "attribute"],
                        summary=Summary(fields=["matched-elements-only"]),
                    ),
                    Field(
                        name="queries",
                        type="array<string>",
                        indexing=["summary", "attribute"],
                        summary=Summary(fields=["matched-elements-only"]),
                    ),
                ]
            ),
            fieldsets=[
                FieldSet(
                    name="default",
                    fields=["title", "text"],
                ),
            ],
            document_summaries=[
                DocumentSummary(
                    name="default",
                    summary_fields=[
                        Summary(
                            name="text",
                            fields=[("bolding", "on")],
                        ),
                        Summary(
                            name="snippet",
                            fields=[("source", "text"), "dynamic"],
                        ),
                    ],
                    from_disk=True,
                ),
                DocumentSummary(
                    name="suggestions",
                    summary_fields=[
                        Summary(name="questions"),
                    ],
                    from_disk=True,
                ),
            ],
        )

        # Define similarity functions used in all rank profiles
        mapfunctions = [
            Function(
                name="similarities",  # computes similarity scores between each query token and image patch
                expression="""
                        sum(
                            query(qt) * unpack_bits(attribute(embedding)), v
                        )
                    """,
            ),
            Function(
                name="normalized",  # normalizes the similarity scores to [-1, 1]
                expression="""
                        (similarities - reduce(similarities, min)) / (reduce((similarities - reduce(similarities, min)), max)) * 2 - 1
                    """,
            ),
            Function(
                name="quantized",  # quantizes the normalized similarity scores to signed 8-bit integers [-128, 127]
                expression="""
                        cell_cast(normalized * 127.999, int8)
                    """,
            ),
        ]

        # Define the 'bm25' rank profile
        bm25 = RankProfile(
            name="bm25",
            inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
            first_phase="bm25(title) + bm25(text)",
            functions=mapfunctions,
        )

        colpali_schema.add_rank_profile(bm25)
        colpali_schema.add_rank_profile(with_quantized_similarity(bm25))


        # Update the 'colpali' rank profile
        input_query_tensors = []
        MAX_QUERY_TERMS = 64
        for i in range(MAX_QUERY_TERMS):
            input_query_tensors.append((f"query(rq{i})", "tensor<int8>(v[16])"))

        input_query_tensors.extend(
            [
                ("query(qt)", "tensor<float>(querytoken{}, v[128])"),
                ("query(qtb)", "tensor<int8>(querytoken{}, v[16])"),
            ]
        )

        colpali = RankProfile(
            name="colpali",
            inputs=input_query_tensors,
            first_phase="max_sim_binary",
            second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10),
            functions=mapfunctions
            + [
                Function(
                    name="max_sim",
                    expression="""
                        sum(
                            reduce(
                                sum(
                                    query(qt) * unpack_bits(attribute(embedding)), v
                                ),
                                max, patch
                            ),
                            querytoken
                        )
                    """,
                ),
                Function(
                    name="max_sim_binary",
                    expression="""
                        sum(
                            reduce(
                                1 / (1 + sum(
                                    hamming(query(qtb), attribute(embedding)), v)
                                ),
                                max, patch
                            ),
                            querytoken
                        )
                    """,
                ),
            ],
        )
        colpali_schema.add_rank_profile(colpali)
        colpali_schema.add_rank_profile(with_quantized_similarity(colpali))

        # Update the 'hybrid' rank profile
        hybrid = RankProfile(
            name="hybrid",
            inputs=input_query_tensors,
            first_phase="max_sim_binary",
            second_phase=SecondPhaseRanking(
                expression="max_sim + 2 * (bm25(text) + bm25(title))", rerank_count=10
            ),
            functions=mapfunctions
            + [
                Function(
                    name="max_sim",
                    expression="""
                        sum(
                            reduce(
                                sum(
                                    query(qt) * unpack_bits(attribute(embedding)), v
                                ),
                                max, patch
                            ),
                            querytoken
                        )
                    """,
                ),
                Function(
                    name="max_sim_binary",
                    expression="""
                        sum(
                            reduce(
                                1 / (1 + sum(
                                    hamming(query(qtb), attribute(embedding)), v)
                                ),
                                max, patch
                            ),
                            querytoken
                        )
                    """,
                ),
            ],
        )
        colpali_schema.add_rank_profile(hybrid)
        colpali_schema.add_rank_profile(with_quantized_similarity(hybrid))

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

        # Create the Vespa application package
        vespa_application_package = ApplicationPackage(
            name=VESPA_APPLICATION_NAME,
            schema=[colpali_schema],
            services_config=service_config,
        )

        vespa_cloud = VespaCloud(
            tenant=VESPA_TENANT_NAME,
            application=VESPA_APPLICATION_NAME,
            key_content=VESPA_TEAM_API_KEY,
            application_package=vespa_application_package,
            auth_client_token_id=f"{VESPA_TOKEN_ID_WRITE}",
        )

        logger.info(f"Deploying application {VESPA_APPLICATION_NAME} to tenant {VESPA_TENANT_NAME}")

        # Deploy the application
        vespa_cloud.deploy()

        endpoint_url = vespa_cloud.get_token_endpoint()
        logger.info(f"Application deployed. Token endpoint URL: {endpoint_url}")

        # Save endpoint_url to the database
        await request.app.db.update_user_settings(user_id, {"vespa_app_url": endpoint_url})

        logger.info("Deployment completed successfully!")

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
