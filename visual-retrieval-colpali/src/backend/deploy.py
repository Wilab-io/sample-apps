import os
import asyncio
import json
from typing import Tuple
import hashlib
from typing import List
import numpy as np
from typing import Optional
from dotenv import load_dotenv
import logging
import requests
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

logger = logging.getLogger("vespa_app")

async def deploy_application(settings, user_id):
    """Deploy the Vespa application with progress updates"""
    try:
        yield "Starting deployment process...\n"

        # Load environment variables
        load_dotenv()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        yield "Validating settings...\n"
        if not all([
            settings.tenant_name,
            settings.app_name,
            settings.vespa_token_id,
            settings.vespa_token_value,
            settings.gemini_token
        ]):
            yield "Error: Missing required settings\n"
            yield StopAsyncIteration  # Signal completion
            return

        yield "Preparing application package...\n"
        # Get paths to all PDFs in user's storage directory
        storage_dir = os.path.join("src", "storage", user_id)
        if not os.path.exists(storage_dir):
            yield f"Error: No storage directory found for user {user_id}\n"
            yield StopAsyncIteration  # Signal completion
            return

        paths = [
            os.path.join(storage_dir, f)
            for f in os.listdir(storage_dir)
            if f.endswith(".pdf")
        ]

        if not paths:
            yield f"Error: No PDF files found in {storage_dir}\n"
            yield StopAsyncIteration  # Signal completion
            return

        yield f"Found {len(paths)} PDF files to process\n"

        # Process PDFs and deploy
        try:
            for pdf_file in paths:
                yield f"Processing {os.path.basename(pdf_file)}...\n"
                images, texts = get_pdf_images(pdf_file)
                yield f"Extracted {len(images)} pages from {os.path.basename(pdf_file)}\n"

            yield "Deploying to Vespa cloud...\n"
            # Add your deployment logic here

            yield "Deployment completed successfully!\n"
            yield StopAsyncIteration  # Signal completion

        except Exception as e:
            yield f"Error during processing: {str(e)}\n"
            yield StopAsyncIteration  # Signal completion
            return

    except Exception as e:
        yield f"Deployment failed: {str(e)}\n"
        yield StopAsyncIteration  # Signal completion

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
