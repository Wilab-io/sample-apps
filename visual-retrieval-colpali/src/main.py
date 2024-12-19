import asyncio
import base64
import os
import time
import logging
import sys
import torch
import pytesseract
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from fasthtml.common import StaticFiles
from pathlib import Path

import google.generativeai as genai
from fastcore.parallel import threaded
from fasthtml.common import (
    Aside,
    Div,
    FileResponse,
    HighlightJS,
    Img,
    JSONResponse,
    Link,
    Main,
    P,
    Redirect,
    Script,
    StreamingResponse,
    fast_app,
    serve,
)
from PIL import Image
from shad4fast import ShadHead
from sqlalchemy import select
from backend.auth import verify_password
from backend.database import Database
from backend.models import User

from backend.colpali import SimMapGenerator
from backend.vespa_app import VespaQueryClient
from backend.models import UserSettings
from frontend.app import (
    AboutThisDemo,
    Home,
    Search,
    SearchBox,
    SearchResult,
    SimMapButtonPoll,
    SimMapButtonReady,
)
from frontend.layout import Layout
from frontend.components.login import Login
from backend.middleware import login_required
from backend.init_db import init_default_users, clear_image_queries
from frontend.components.my_documents import MyDocuments, DocumentProcessingModal, DocumentProcessingErrorModal
from frontend.components.settings import Settings, TabContent
from backend.deploy import deploy_application_step_1, deploy_application_step_2
from backend.feed import feed_documents_to_vespa, remove_document_from_vespa
from frontend.components.deployment import DeploymentModal, DeploymentLoginModal,DeploymentSuccessModal, DeploymentErrorModal
from frontend.components.image_search import ImageSearchModal

highlight_js_theme_link = Link(id="highlight-theme", rel="stylesheet", href="")
highlight_js_theme = Script(src="/static/js/highlightjs-theme.js")
highlight_js = HighlightJS(
    langs=["python", "javascript", "java", "json", "xml"],
    dark="github-dark",
    light="github",
)

overlayscrollbars_link = Link(
    rel="stylesheet",
    href="https://cdnjs.cloudflare.com/ajax/libs/overlayscrollbars/2.10.0/styles/overlayscrollbars.min.css",
    type="text/css",
)
overlayscrollbars_js = Script(
    src="https://cdnjs.cloudflare.com/ajax/libs/overlayscrollbars/2.10.0/browser/overlayscrollbars.browser.es5.min.js"
)
awesomplete_link = Link(
    rel="stylesheet",
    href="https://cdnjs.cloudflare.com/ajax/libs/awesomplete/1.1.7/awesomplete.min.css",
    type="text/css",
)
awesomplete_js = Script(
    src="https://cdnjs.cloudflare.com/ajax/libs/awesomplete/1.1.7/awesomplete.min.js"
)
sselink = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")
deployment_js = Script(src="/static/js/deployment.js")
upload_documents_js = Script(src="/static/js/upload-documents.js")
image_search_js = Script(src="/static/js/image-search.js")

# Get log level from environment variable, default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Configure logger
logger = logging.getLogger("vespa_app")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter(
        "%(levelname)s: \t %(asctime)s \t %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logger.addHandler(handler)
logger.setLevel(getattr(logging, LOG_LEVEL))

# Add the settings.js script to the headers
settings_js = Script(src="/static/js/settings.js")

app, rt = fast_app(
    htmlkw={"cls": "grid h-full"},
    pico=False,
    hdrs=(
        highlight_js,
        highlight_js_theme_link,
        highlight_js_theme,
        overlayscrollbars_link,
        overlayscrollbars_js,
        awesomplete_link,
        awesomplete_js,
        sselink,
        ShadHead(tw_cdn=False, theme_handle=True),
        settings_js,
        deployment_js,
        upload_documents_js,
        image_search_js,
    ),
)
thread_pool = ThreadPoolExecutor()
app.deployed = False
app.results_cache = {}  # Initialize the results cache

def configure_static_routes(app):
    os.makedirs("storage", exist_ok=True)
    app.mount("/storage", StaticFiles(directory="storage"), name="storage")

configure_static_routes(app)

# Gemini config
def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)
    GEMINI_SYSTEM_PROMPT = """If the user query is a question, try your best to answer it based on the provided images.
    If the user query can not be interpreted as a question, or if the answer to the query can not be inferred from the images,
    answer with the exact phrase "I am sorry, I can't find enough relevant information on these pages to answer your question.".
    Your response should be HTML formatted, but only simple tags, such as <b>. <p>, <i>, <br> <ul> and <li> are allowed. No HTML tables.
    This means that newlines will be replaced with <br> tags, bold text will be enclosed in <b> tags, and so on.
    Do NOT include backticks (`) in your response. Only simple HTML tags and text.
    """
    app.gemini_model = genai.GenerativeModel(
        "gemini-1.5-flash-8b", system_instruction=GEMINI_SYSTEM_PROMPT
    )


STATIC_DIR = Path("static")
IMG_DIR = STATIC_DIR / "full_images"
SIM_MAP_DIR = STATIC_DIR / "sim_maps"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(SIM_MAP_DIR, exist_ok=True)

app.db = Database()

@app.on_event("shutdown")
def shutdown_db():
    app.db.close()

@app.on_event("startup")
def load_model_on_startup():
    app.sim_map_generator = SimMapGenerator(logger=logger)
    return

@app.on_event("startup")
async def keepalive():
    asyncio.create_task(poll_vespa_keepalive())
    return


@app.on_event("startup")
async def startup_event():
    try:
        os.environ["USE_MTLS"] = "true"
        await clear_image_queries(logger)
        await init_default_users(logger)
    except SystemExit:
        logger.error("Application Startup Failed")
        raise RuntimeError("Failed to initialize application")


def generate_query_id(query, ranking_value):
    hash_input = (query + ranking_value).encode("utf-8")
    return str(abs(hash(hash_input)))


@rt("/static/{filepath:path}")
def serve_static(filepath: str):
    return FileResponse(STATIC_DIR / filepath)


@rt("/")
@login_required
async def get(request):
    if "user_id" not in request.session:
        return Redirect("/login")
    user_id = request.session["user_id"]
    settings = await request.app.db.get_user_settings(user_id)
    return await Layout(Main(await Home(request, settings.ranker.value, app.deployed)), is_home=True, request=request)


@rt("/about-this-demo")
@login_required
async def get(request):
    return await Layout(Main(AboutThisDemo()), request=request)


@rt("/search")
@login_required
async def get(request, query: str = "", ranking: str = "colpali", image_query: str = None, query_id: str = None):
    logger.info(f"/search: Fetching results for query: {query}, ranking: {ranking}, image_query: {image_query}, query_id: {query_id}")

    # Generate a unique query_id if not provided
    if not query_id:
        query_id = generate_query_id(image_query or query, ranking)

    # Check if we have results in cache
    if query_id in request.app.results_cache:
        logger.info(f"Found cached results for query_id: {query_id}")
        cached_results = request.app.results_cache[query_id]
        return await Layout(
            Main(Search(request, cached_results, query=query, image_query=image_query, query_id=query_id)),
            request=request
        )

    # Always render the SearchBox first if no query parameters
    if not any([query, image_query]):
        return await Layout(
            Main(
                Div(
                    SearchBox(query_value=query, ranking_value=ranking),
                    Div(
                        P(
                            "No query provided. Please enter a query.",
                            cls="text-center text-muted-foreground",
                        ),
                        cls="p-10",
                    ),
                    cls="grid",
                )
            ),
            request=request
        )

    if image_query:
        logger.info(f"Processing image query with ID: {image_query}")
        try:
            # Get image query data from database
            image_query_data = await request.app.db.get_image_query(image_query)
            if not image_query_data:
                logger.error(f"Image query not found: {image_query}")
                return await Layout(
                    Main(
                        Div(
                            SearchBox(query_value=query, ranking_value=ranking),
                            Div(
                                P(
                                    "Image query not found.",
                                    cls="text-center text-muted-foreground",
                                ),
                                cls="p-10",
                            ),
                            cls="grid",
                        )
                    ),
                    request=request
                )

            logger.info(f"Found image query data: visual_only={image_query_data.is_visual_only}")
            embeddings = torch.tensor(image_query_data.embeddings)

            # Use the appropriate query method based on visual_only flag
            app = request.app.vespa_app
            if image_query_data.is_visual_only:
                logger.info("Using visual-only search")
                response = await app.query_vespa_colpali(
                    query="",  # Empty query for visual-only search
                    q_emb=embeddings,
                    ranking="colpali",
                    sim_map=True,
                    visual_only=True
                )
            else:
                logger.info("Using hybrid search with text and visual features")
                response = await app.query_vespa_colpali(
                    query=image_query_data.text,
                    q_emb=embeddings,
                    ranking=ranking,
                    sim_map=True
                )

            # Process the Vespa response
            if not response or 'root' not in response:
                raise ValueError("Invalid response from Vespa")

            root = response['root']
            if 'children' not in root:
                raise ValueError("No results found in Vespa response")

            search_results = []
            for child in root['children']:
                if 'fields' not in child:
                    continue
                fields = child['fields']
                if 'id' not in fields:
                    continue
                search_results.append({
                    'fields': fields,
                    'relevance': child.get('relevance', 0.0)
                })

            # Cache results using only query_id
            request.app.results_cache[query_id] = search_results
            logger.info(f"Cached {len(search_results)} results for query_id: {query_id}")

            return await Layout(
                Main(Search(request, search_results, query=query, image_query=image_query, query_id=query_id)),
                request=request
            )
        except Exception as e:
            logger.error(f"Error processing image query: {str(e)}")
            return await Layout(
                Main(
                    Div(
                        SearchBox(query_value=query, ranking_value=ranking),
                        Div(
                            P(
                                f"Error processing image query: {str(e)}",
                                cls="text-center text-muted-foreground",
                            ),
                            cls="p-10",
                        ),
                        cls="grid",
                    )
                ),
                request=request
            )

    # Show the loading message if a query is provided
    return await Layout(
        Main(
            Search(request, query_id=query_id, query=query),
            data_overlayscrollbars_initialize=True,
            cls="border-t"
        ),
        request=request
    )


@rt("/fetch_results")
@login_required
async def get(session, request, query: str, ranking: str):
    if "hx-request" not in request.headers:
        return Redirect("/search")

    # Get the hash of the query and ranking value
    query_id = generate_query_id(query, ranking)
    logger.info(f"Query id in /fetch_results: {query_id}")

    # Run the embedding and query against Vespa app
    start_inference = time.perf_counter()
    q_embs, idx_to_token = app.sim_map_generator.get_query_embeddings_and_token_map(query)
    end_inference = time.perf_counter()
    logger.info(f"Inference time for query_id: {query_id} \t {end_inference - start_inference:.2f} seconds")

    start = time.perf_counter()
    # Fetch real search results from Vespa
    result = await app.vespa_app.get_result_from_query(
        query=query,
        q_embs=q_embs,
        ranking=ranking,
        idx_to_token=idx_to_token,
    )
    end = time.perf_counter()
    logger.info(f"Search results fetched in {end - start:.2f} seconds. Vespa search time: {result['timing']['searchtime']}")
    search_time = result["timing"]["searchtime"]
    total_count = result.get("root", {}).get("fields", {}).get("totalCount", 0)

    search_results = app.vespa_app.results_to_search_results(result, idx_to_token)

    # Store the results in the cache using string query_id
    request.app.results_cache[query_id] = search_results
    logger.info(f"Stored {len(search_results)} results in cache with query_id: {query_id}")

    get_and_store_sim_maps(
        query_id=query_id,
        query=query,
        q_embs=q_embs,
        ranking=ranking,
        idx_to_token=idx_to_token,
        doc_ids=[result["fields"]["id"] for result in search_results],
    )
    return SearchResult(search_results, query, query_id, search_time, total_count)


def get_results_children(result):
    search_results = (
        result["root"]["children"]
        if "root" in result and "children" in result["root"]
        else []
    )
    return search_results


async def poll_vespa_keepalive():
    while True:
        await asyncio.sleep(5)
        if hasattr(app, "vespa_app") and app.vespa_app:
            await app.vespa_app.keepalive()
            logger.debug(f"Vespa keepalive: {time.time()}")


@threaded
def get_and_store_sim_maps(
    query_id, query: str, q_embs, ranking, idx_to_token, doc_ids
):
    try:
        logger.info(f"Starting sim map generation for query_id: {query_id}")
        ranking_sim = ranking + "_sim"
        vespa_sim_maps = app.vespa_app.get_sim_maps_from_query(
            query=query,
            q_embs=q_embs,
            ranking=ranking_sim,
            idx_to_token=idx_to_token,
        )
        logger.info(f"Retrieved {len(vespa_sim_maps)} sim maps from Vespa")

        img_paths = [IMG_DIR / f"{doc_id}.jpg" for doc_id in doc_ids]
        logger.info(f"Checking for images at paths: {img_paths}")

        # Download any missing images first
        missing_images = [(doc_id, path) for doc_id, path in zip(doc_ids, img_paths) if not os.path.exists(path)]
        if missing_images:
            logger.info(f"Downloading {len(missing_images)} missing images...")
            for doc_id, path in missing_images:
                try:
                    image_data = asyncio.run(app.vespa_app.get_full_image_from_vespa(doc_id))
                    with open(path, "wb") as f:
                        f.write(base64.b64decode(image_data))
                    logger.debug(f"Downloaded image for doc_id: {doc_id}")
                except Exception as e:
                    logger.error(f"Failed to download image for doc_id {doc_id}: {str(e)}")
                    return False

        # Verify all images are now available
        if not all([os.path.exists(img_path) for img_path in img_paths]):
            logger.error("Some images still missing after download attempt")
            return False

        logger.debug("All images found, generating similarity maps")
        sim_map_generator = app.sim_map_generator.gen_similarity_maps(
            query=query,
            query_embs=q_embs,
            token_idx_map=idx_to_token,
            images=img_paths,
            vespa_sim_maps=vespa_sim_maps,
        )

        for idx, token, token_idx, blended_img_base64 in sim_map_generator:
            sim_map_path = SIM_MAP_DIR / f"{query_id}_{idx}_{token_idx}.png"
            try:
                with open(sim_map_path, "wb") as f:
                    f.write(base64.b64decode(blended_img_base64))
                logger.info(
                    f"Sim map saved to disk for query_id: {query_id}, idx: {idx}, token: {token}"
                )
            except Exception as e:
                logger.error(f"Error saving sim map to {sim_map_path}: {str(e)}")

        logger.info(f"Completed sim map generation for query_id: {query_id}")
        return True
    except Exception as e:
        logger.error(f"Error in get_and_store_sim_maps: {str(e)}")
        logger.error("Error traceback:", exc_info=True)
        return False


@rt("/get_sim_map")
@login_required
async def get_sim_map(request, query_id: str, idx: int, token: str, token_idx: int):
    """
    Endpoint that each of the sim map button polls to get the sim map image
    when it is ready. If it is not ready, returns a SimMapButtonPoll, that
    continues to poll every 1 second.
    """
    sim_map_path = SIM_MAP_DIR / f"{query_id}_{idx}_{token_idx}.png"
    if not os.path.exists(sim_map_path):
        logger.debug(
            f"Sim map not ready for query_id: {query_id}, idx: {idx}, token: {token}"
        )
        return SimMapButtonPoll(
            query_id=query_id, idx=idx, token=token, token_idx=token_idx
        )
    else:
        return SimMapButtonReady(
            query_id=query_id,
            idx=idx,
            token=token,
            token_idx=token_idx,
            img_src=sim_map_path,
        )


@rt("/full_image")
async def full_image(doc_id: str):
    """
    Endpoint to get the full quality image for a given result id.
    """
    img_path = IMG_DIR / f"{doc_id}.jpg"
    if not os.path.exists(img_path):
        image_data = await app.vespa_app.get_full_image_from_vespa(doc_id)
        # image data is base 64 encoded string. Save it to disk as jpg.
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(image_data))
        logger.debug(f"Full image saved to disk for doc_id: {doc_id}")
    else:
        with open(img_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    return Img(
        src=f"data:image/jpeg;base64,{image_data}",
        alt="something",
        cls="result-image w-full h-full object-contain",
    )


@rt("/suggestions")
async def get_suggestions(query: str = ""):
    """Endpoint to get suggestions as user types in the search box"""
    query = query.lower().strip()

    if query:
        suggestions = await app.vespa_app.get_suggestions(query)
        if len(suggestions) > 0:
            return JSONResponse({"suggestions": suggestions})

    return JSONResponse({"suggestions": []})


async def message_generator(query_id: str, query: str, doc_ids: list):
    """Generator function to yield SSE messages for chat response"""
    images = []
    num_images = min(3, len(doc_ids))  # Use min to avoid index out of range
    max_wait = 10  # seconds
    start_time = time.time()

    # Check if full images are ready on disk
    while (
        len(images) < num_images
        and time.time() - start_time < max_wait
    ):
        images = []
        for idx in range(num_images):
            image_filename = IMG_DIR / f"{doc_ids[idx]}.jpg"
            if not os.path.exists(image_filename):
                logger.debug(
                    f"Message generator: Full image not ready for query_id: {query_id}, idx: {idx}"
                )
                continue
            else:
                logger.debug(
                    f"Message generator: image ready for query_id: {query_id}, idx: {idx}"
                )
                images.append(Image.open(image_filename))
        if len(images) < num_images:
            await asyncio.sleep(0.2)

    # yield message with number of images ready
    yield f"event: message\ndata: Generating response based on {len(images)} images...\n\n"
    if not images:
        yield "event: message\ndata: Failed to send images to Gemini-8B!\n\n"
        yield "event: close\ndata: \n\n"
        return

    # If newlines are present in the response, the connection will be closed.
    def replace_newline_with_br(text):
        return text.replace("\n", "<br>")

    # If query is empty (visual search), use a default prompt
    if not query.strip():
        query = "Describe what you see in these images and their key visual elements."

    response_text = ""
    async for chunk in await app.gemini_model.generate_content_async(
        images + ["\n\n Query: ", query], stream=True
    ):
        if chunk.text:
            response_text += chunk.text
            response_text = replace_newline_with_br(response_text)
            yield f"event: message\ndata: {response_text}\n\n"
            await asyncio.sleep(0.1)
    yield "event: close\ndata: \n\n"


@rt("/get-message")
@login_required
async def get_message(request, query_id: str, query: str, doc_ids: str):
    return StreamingResponse(
        message_generator(query_id=query_id, query=query, doc_ids=doc_ids.split(",")),
        media_type="text/event-stream",
    )


@rt("/app")
@login_required
async def get(request):
    return await Layout(
        Main(Div(P(f"Connected to Vespa at {app.vespa_app.url}"), cls="p-4")),
        request=request
    )


@rt("/login")
async def get(request):
    if "user_id" in request.session:
        return Redirect("/")
    return await Layout(Main(Login()))


@rt("/api/login", methods=["POST"])
async def login(request, username: str, password: str):
    async with app.db.get_session() as session:
        try:
            result = await session.execute(
                select(User).where(User.username == username)
            )
            user = result.scalar_one_or_none()

            if not user:
                logger.debug("User not found: %s", username)
                return Login(error_message="The user does not exist")
            if not verify_password(password, user.password_hash, logger):
                logger.debug("Invalid credentials for user: %s", username)
                return Login(error_message="Invalid password")

            request.session["user_id"] = str(user.user_id)
            request.session["username"] = user.username
            logger.debug("Successful login for user: %s", username)

            return Redirect("/")

        except Exception as e:
            logger.error("Login error: %s", str(e))
            return Login(error_message="An error occurred during login. Please try again.")


@rt("/my-documents")
@login_required
async def get_my_documents(request):
    user_id = request.session["user_id"]
    logger.debug(f"Fetching documents for user_id: {user_id}")
    documents = await app.db.get_user_documents(user_id)
    logger.debug(f"Found {len(documents) if documents else 0} documents")
    return await Layout(
        Main(await MyDocuments(documents=documents, app_deployed=app.deployed)()),
        request=request
    )


@rt("/logout")
async def logout(request):
    if "user_id" in request.session:
        del request.session["user_id"]
        del request.session["username"]
    return Redirect("/login")

@rt("/upload-files", methods=["POST"])
@login_required
async def upload_files(request):
    user_id = request.session["user_id"]
    settings: UserSettings = await request.app.db.get_user_settings(user_id)
    if not settings:
        logger.error("Settings not found")
        return {"status": "error", "message": "Settings not found"}

    try:
        form = await request.form()
        files = form.getlist("files")
        logger.info(f"Received {len(files)} files")

        doc_names = dict()

        for file in files:
            if file.filename:
                content = await file.read()
                document_id = await app.db.add_user_document(
                    user_id=user_id,
                    document_name=file.filename,
                    file_content=content
                )
                doc_names[document_id] = file.filename

        model = app.sim_map_generator.model
        processor = app.sim_map_generator.processor

        try:
            result = feed_documents_to_vespa(settings, user_id, model, processor, doc_names)
            if result["status"] == "error":
                logger.error(f"Error during vespa feed: {result['message']}")
                # Clean up documents on error
                logger.info(f"Deleting {len(doc_names)} documents from database")
                for doc_id in doc_names.keys():
                    await app.db.delete_document(doc_id)
                return result
            return {"status": "success"}

        except Exception as e:
            logger.error(f"Error during vespa feed: {str(e)}")
            # Clean up documents on error
            logger.info(f"Deleting {len(doc_names)} documents from database")
            for doc_id in doc_names.keys():
                await app.db.delete_document(doc_id)
            return {"status": "error", "message": str(e)}

    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        return {"status": "error", "message": str(e)}

@rt("/delete-document/{document_id}", methods=["DELETE"])
@login_required
async def delete_document(request, document_id: str):
    logger.info(f"Delete document request for document_id: {document_id}")
    user_id = request.session["user_id"]

    try:
        # Get settings for Vespa removal
        settings = await request.app.db.get_user_settings(user_id)
        if not settings:
            logger.error("Settings not found")
            return {"status": "error", "message": "Settings not found"}

        # First remove from Vespa
        remove_document_from_vespa(settings, document_id)

        # Then delete from database and filesystem
        await app.db.delete_document(document_id)

        # Return updated table
        documents = await app.db.get_user_documents(user_id)
        return MyDocuments(documents=documents).documents_table()

    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise

@rt("/settings")
@login_required
async def get(request):
    user_id = request.session["user_id"]
    tab = request.query_params.get("tab", "demo-questions")

    if "username" not in request.session:
        user = await request.app.db.get_user_by_id(user_id)
        request.session["username"] = user.username if user else None

    if request.session["username"] != "admin" and tab == "prompt":
        tab = "demo-questions"

    settings = await request.app.db.get_user_settings(user_id)
    app_configured = await request.app.db.is_application_configured(user_id)

    logger.debug(f"Application configuration check: {app_configured}")

    return await Layout(
        Settings(
            active_tab=tab,
            settings=settings,
            username=request.session["username"],
            appConfigured=app_configured,
        ),
        request=request
    )

@rt("/settings/content")
@login_required
async def get_settings_content(request):
    user_id = request.session["user_id"]
    tab = request.query_params.get("tab", "demo-questions")

    if "username" not in request.session:
        user = await request.app.db.get_user_by_id(user_id)
        request.session["username"] = user.username if user else None

    if request.session["username"] != "admin" and tab == "prompt":
        tab = "demo-questions"

    settings = await request.app.db.get_user_settings(user_id)
    app_configured = await request.app.db.is_application_configured(user_id)

    logger.debug(f"Application configuration check: {app_configured}")
    return TabContent(
        tab,
        settings,
        username=request.session["username"],
        appConfigured=app_configured
    )

@rt("/api/settings/demo-questions", methods=["POST"])
@login_required
async def update_demo_questions(request):
    form_data = await request.form()
    questions = []
    i = 0

    while f"question_{i}" in form_data:
        question = form_data[f"question_{i}"].strip()
        if question:
            questions.append(question)
        i += 1

    if questions:
        user_id = request.session["user_id"]
        await request.app.db.update_settings(user_id, {'demo_questions': questions})

    return Redirect("/settings?tab=demo-questions")

@rt("/api/settings/ranker", methods=["POST"])
@login_required
async def update_ranker(request):
    user_id = request.session["user_id"]
    form = await request.form()
    ranker = form.get("ranker")

    await request.app.db.update_settings(user_id, {'ranker': ranker})

    return Redirect("/settings?tab=ranker")

@rt("/api/settings/connection", methods=["POST"])
@login_required
async def update_connection_settings(request):
    user_id = request.session["user_id"]
    form = await request.form()

    settings = {
        'gemini_token': form.get('gemini_token'),
    }

    # Handle API key file upload
    api_key_file = form.get("api_key_file")
    if api_key_file and hasattr(api_key_file, "filename"):
        from pathlib import Path
        import shutil

        # Create user's key directory if it doesn't exist
        user_key_dir = Path("storage/user_keys") / str(user_id)
        user_key_dir.mkdir(parents=True, exist_ok=True)

        # Remove any existing .pem files
        for existing_file in user_key_dir.glob("*.pem"):
            existing_file.unlink()

        # Save the new file
        file_content = await api_key_file.read()
        new_file_path = user_key_dir / f"{api_key_file.filename}"
        with open(new_file_path, "wb") as f:
            f.write(file_content)

    await request.app.db.update_settings(user_id, settings)

    return Redirect("/settings?tab=connection")

@rt("/api/settings/application-package", methods=["POST"])
@login_required
async def update_application_package_settings(request):
    user_id = request.session["user_id"]
    form = await request.form()

    settings = {
        'tenant_name': form.get('tenant_name'),
        'app_name': form.get('app_name'),
        'instance_name': form.get('instance_name'),
        'schema': form.get('schema')
    }

    await request.app.db.update_settings(user_id, settings)

    return Redirect("/settings?tab=application-package")


@rt("/api/settings/prompt", methods=["POST"])
@login_required
async def update_prompt_settings(request):
    if request.session["username"] != "admin":
        return Redirect("/settings?tab=demo-questions")

    form = await request.form()
    prompt = form.get('prompt')
    await request.app.db.update_settings(request.session["user_id"], {'prompt': prompt})

    return Redirect("/settings?tab=prompt")

@rt("/login", methods=["POST"])
async def login(request):
    form = await request.form()
    username = form.get("username")
    password = form.get("password")

    user = await app.db.fetch_one(
        select(User).where(User.username == username)
    )

    if user and verify_password(password, user["password_hash"]):
        request.session["user_id"] = str(user["user_id"])
        request.session["username"] = username
        return Redirect("/")

    return Redirect("/login?error=invalid")

@app.post("/api/deploy-part-1")
@login_required
async def deploy_part_1(request):
    try:
        user_id = request.session["user_id"]
        settings: UserSettings = await request.app.db.get_user_settings(user_id)
        if not settings:
            logger.error("Settings not found")
            return {"status": "error", "message": "Settings not found"}

        result = await deploy_application_step_1(settings)
        return result

    except Exception as e:
        logger.error(f"Deployment error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/api/deploy-part-2")
@login_required
async def deploy_part_2(request):
    try:
        user_id = request.session["user_id"]
        settings: UserSettings = await request.app.db.get_user_settings(user_id)
        if not settings:
            logger.error("Settings not found")
            return {"status": "error", "message": "Settings not found"}

        result = await deploy_application_step_2(request, settings, user_id)

        # Read the settings again to get the new URL
        settings: UserSettings = await request.app.db.get_user_settings(user_id)
        if not settings:
            logger.error("Settings not found")
            return {"status": "error", "message": "Settings not found"}

        app.vespa_app = VespaQueryClient(logger=logger, settings=settings)

        # Configure Gemini with the API key
        configure_gemini(settings.gemini_token)

        app.deployed = True

        return result

    except Exception as e:
        logger.error(f"Deployment error: {str(e)}")
        return {"status": "error", "message": str(e)}

@rt("/deployment-modal")
@login_required
async def get_deployment_modal(request):
    return DeploymentModal()

@rt("/deployment-modal/login")
@login_required
async def get_deployment_login_modal(request):
    auth_url = request.query_params.get("auth_url", "")
    return DeploymentLoginModal(url=auth_url)

@rt("/deployment-modal/success")
@login_required
async def get_deployment_success_modal(request):
    return DeploymentSuccessModal()

@rt("/deployment-modal/error")
@login_required
async def get_deployment_error_modal(request):
    return DeploymentErrorModal()

@rt("/document-processing-modal")
@login_required
async def get_document_processing_modal(request):
    return DocumentProcessingModal()

@rt("/document-processing-modal/error")
@login_required
async def get_document_processing_error_modal(request):
    message = request.query_params.get("message", None)
    return DocumentProcessingErrorModal(message=message)

@rt("/detail")
@login_required
async def get(request, doc_id: str, query_id: str, query: str):
    logger.info(f"/detail: Showing details for query_id: {query_id}, query: {query}")

    try:
        results = request.app.results_cache.get(query_id, [])
        logger.info(f"Found {len(results)} results in cache for query_id: {query_id}")
    except (ValueError, TypeError) as e:
        logger.error(f"Error getting result from cache: {str(e)}")
        results = []

    return await Layout(
        Main(
            SearchResult(
                results=results,
                query=query,
                query_id=query_id,
                doc_id=doc_id,
            ),
            data_overlayscrollbars_initialize=True,
            cls="border-t",
        ),
        request=request
    )

@rt("/api/image-search", methods=["POST"])
@login_required
async def image_search(request):
    try:
        logger.info("Image search endpoint called")
        form = await request.form()
        image_file = form["image"]
        logger.info(f"Received image file: {image_file.filename}")

        image_content = await image_file.read()
        logger.info(f"Read image content, size: {len(image_content)} bytes")

        image = Image.open(BytesIO(image_content))
        logger.info(f"Opened image: {image.size}, mode: {image.mode}")

        # Process the image using the processor
        logger.info("Processing image with ColPali processor")
        processed_image = app.sim_map_generator.processor.process_images([image])
        logger.info(f"Processed image keys: {processed_image.keys()}")

        processed_image = {k: v.to(app.sim_map_generator.model.device) for k, v in processed_image.items()}
        logger.info(f"Moved tensors to device: {app.sim_map_generator.model.device}")

        # Generate embeddings using the model
        logger.info("Generating embeddings")
        with torch.no_grad():
            embeddings = app.sim_map_generator.model(**processed_image)
        logger.info(f"Generated embeddings shape: {embeddings.shape}")

        try:
            logger.info("Extracting text with OCR")
            text = pytesseract.image_to_string(image)
            logger.info(f"Extracted text length: {len(text)}")
        except Exception as ocr_error:
            logger.error(f"OCR failed: {str(ocr_error)}")
            text = ""

        # Store the query information
        query_id = generate_query_id(text or "image_query", "image_search")
        logger.info(f"Generated query ID: {query_id}")

        # If no text was found, we'll rely purely on visual similarity
        is_visual_only = not bool(text.strip())
        logger.info(f"Is visual only: {is_visual_only}")

        logger.info("Storing image query in database")
        await app.db.store_image_query(query_id, embeddings[0], text, is_visual_only)
        logger.info("Successfully stored image query")

        response_data = {
            "query_id": query_id,
            "is_visual_only": is_visual_only
        }
        logger.info(f"Returning response: {response_data}")
        return JSONResponse(response_data)

    except Exception as e:
        logger.error(f"Error processing image search: {str(e)}")
        logger.error(f"Error traceback:", exc_info=True)  # This will log the full traceback
        return JSONResponse({"error": str(e)}, status_code=400)

@rt("/image-search-modal")
@login_required
async def get_image_search_modal(request):
    return ImageSearchModal()

if __name__ == "__main__":
    HOT_RELOAD = os.getenv("HOT_RELOAD", "False").lower() == "true"
    logger.info(f"Starting app with hot reload: {HOT_RELOAD}")
    serve(port=7860, reload=HOT_RELOAD)
