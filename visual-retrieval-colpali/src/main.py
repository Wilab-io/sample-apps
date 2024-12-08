import asyncio
import base64
import os
import time
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
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
from vespa.application import Vespa
from sqlalchemy import select
from backend.auth import verify_password
from backend.database import Database
from backend.models import User

from backend.colpali import SimMapGenerator
from backend.vespa_app import VespaQueryClient
from frontend.app import (
    AboutThisDemo,
    ChatResult,
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
from backend.init_db import init_default_users
from frontend.components.my_documents import MyDocuments
from frontend.components.settings import Settings, TabContent

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
    ),
)
vespa_app: Vespa = VespaQueryClient(logger=logger)
thread_pool = ThreadPoolExecutor()
# Gemini config

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_SYSTEM_PROMPT = """If the user query is a question, try your best to answer it based on the provided images.
If the user query can not be interpreted as a question, or if the answer to the query can not be inferred from the images,
answer with the exact phrase "I am sorry, I can't find enough relevant information on these pages to answer your question.".
Your response should be HTML formatted, but only simple tags, such as <b>. <p>, <i>, <br> <ul> and <li> are allowed. No HTML tables.
This means that newlines will be replaced with <br> tags, bold text will be enclosed in <b> tags, and so on.
Do NOT include backticks (`) in your response. Only simple HTML tags and text.
"""
gemini_model = genai.GenerativeModel(
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
        await init_default_users(logger)
    except SystemExit:
        logger.error("Application Startup Failed")
        raise RuntimeError("Failed to initialize application")


def generate_query_id(query, ranking_value):
    hash_input = (query + ranking_value).encode("utf-8")
    return hash(hash_input)


@rt("/static/{filepath:path}")
def serve_static(filepath: str):
    return FileResponse(STATIC_DIR / filepath)


@rt("/")
@login_required
async def get(request):
    return await Layout(Main(await Home(request)), is_home=True, request=request)


@rt("/about-this-demo")
@login_required
async def get(request):
    return await Layout(Main(AboutThisDemo()), request=request)


@rt("/search")
@login_required
async def get(request, query: str = "", ranking: str = "hybrid"):
    logger.info(f"/search: Fetching results for query: {query}, ranking: {ranking}")

    # Always render the SearchBox first
    if not query:
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
    # Generate a unique query_id based on the query and ranking value
    query_id = generate_query_id(query, ranking)
    # Show the loading message if a query is provided
    return await Layout(
        Main(Search(request), data_overlayscrollbars_initialize=True, cls="border-t"),
        Aside(
            ChatResult(query_id=query_id, query=query),
            cls="border-t border-l hidden md:block",
        ),
        request=request
    )  # Show SearchBox and Loading message initially


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
    q_embs, idx_to_token = app.sim_map_generator.get_query_embeddings_and_token_map(
        query
    )
    end_inference = time.perf_counter()
    logger.info(
        f"Inference time for query_id: {query_id} \t {end_inference - start_inference:.2f} seconds"
    )

    start = time.perf_counter()
    # Fetch real search results from Vespa
    result = await vespa_app.get_result_from_query(
        query=query,
        q_embs=q_embs,
        ranking=ranking,
        idx_to_token=idx_to_token,
    )
    end = time.perf_counter()
    logger.info(
        f"Search results fetched in {end - start:.2f} seconds. Vespa search time: {result['timing']['searchtime']}"
    )
    search_time = result["timing"]["searchtime"]
    # Safely get total_count with a default of 0
    total_count = result.get("root", {}).get("fields", {}).get("totalCount", 0)

    search_results = vespa_app.results_to_search_results(result, idx_to_token)

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
        await vespa_app.keepalive()
        logger.debug(f"Vespa keepalive: {time.time()}")


@threaded
def get_and_store_sim_maps(
    query_id, query: str, q_embs, ranking, idx_to_token, doc_ids
):
    ranking_sim = ranking + "_sim"
    vespa_sim_maps = vespa_app.get_sim_maps_from_query(
        query=query,
        q_embs=q_embs,
        ranking=ranking_sim,
        idx_to_token=idx_to_token,
    )
    img_paths = [IMG_DIR / f"{doc_id}.jpg" for doc_id in doc_ids]
    # All images should be downloaded, but best to wait 5 secs
    max_wait = 5
    start_time = time.time()
    while (
        not all([os.path.exists(img_path) for img_path in img_paths])
        and time.time() - start_time < max_wait
    ):
        time.sleep(0.2)
    if not all([os.path.exists(img_path) for img_path in img_paths]):
        logger.warning(f"Images not ready in 5 seconds for query_id: {query_id}")
        return False
    sim_map_generator = app.sim_map_generator.gen_similarity_maps(
        query=query,
        query_embs=q_embs,
        token_idx_map=idx_to_token,
        images=img_paths,
        vespa_sim_maps=vespa_sim_maps,
    )
    for idx, token, token_idx, blended_img_base64 in sim_map_generator:
        with open(SIM_MAP_DIR / f"{query_id}_{idx}_{token_idx}.png", "wb") as f:
            f.write(base64.b64decode(blended_img_base64))
        logger.debug(
            f"Sim map saved to disk for query_id: {query_id}, idx: {idx}, token: {token}"
        )
    return True


@app.get("/get_sim_map")
@login_required
async def get_sim_map(query_id: str, idx: int, token: str, token_idx: int):
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


@app.get("/full_image")
@login_required
async def full_image(doc_id: str):
    """
    Endpoint to get the full quality image for a given result id.
    """
    img_path = IMG_DIR / f"{doc_id}.jpg"
    if not os.path.exists(img_path):
        image_data = await vespa_app.get_full_image_from_vespa(doc_id)
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
@login_required
async def get_suggestions(query: str = ""):
    """Endpoint to get suggestions as user types in the search box"""
    query = query.lower().strip()

    if query:
        suggestions = await vespa_app.get_suggestions(query)
        if len(suggestions) > 0:
            return JSONResponse({"suggestions": suggestions})

    return JSONResponse({"suggestions": []})


async def message_generator(query_id: str, query: str, doc_ids: list):
    """Generator function to yield SSE messages for chat response"""
    images = []
    num_images = 3  # Number of images before firing chat request
    max_wait = 10  # seconds
    start_time = time.time()
    # Check if full images are ready on disk
    while (
        len(images) < min(num_images, len(doc_ids))
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

    response_text = ""
    async for chunk in await gemini_model.generate_content_async(
        images + ["\n\n Query: ", query], stream=True
    ):
        if chunk.text:
            response_text += chunk.text
            response_text = replace_newline_with_br(response_text)
            yield f"event: message\ndata: {response_text}\n\n"
            await asyncio.sleep(0.1)
    yield "event: close\ndata: \n\n"


@app.get("/get-message")
@login_required
async def get_message(query_id: str, query: str, doc_ids: str):
    return StreamingResponse(
        message_generator(query_id=query_id, query=query, doc_ids=doc_ids.split(",")),
        media_type="text/event-stream",
    )


@rt("/app")
@login_required
async def get(request):
    return await Layout(
        Main(Div(P(f"Connected to Vespa at {vespa_app.url}"), cls="p-4")),
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
        Main(await MyDocuments(documents=documents)()),
        request=request
    )


@rt("/logout")
async def logout(request):
    if "user_id" in request.session:
        del request.session["user_id"]
        del request.session["username"]
    return Redirect("/login")

STORAGE_DIR = Path("storage/user_documents")

@rt("/upload-files", methods=["POST"])
@login_required
async def upload_files(request):
    logger.info("Upload files endpoint called")
    user_id = request.session["user_id"]

    try:
        form = await request.form()
        files = form.getlist("files")
        logger.info(f"Received {len(files)} files")

        for file in files:
            if file.filename:
                content = await file.read()
                await app.db.add_user_document(
                    user_id=user_id,
                    document_name=file.filename,
                    file_content=content
                )

        # Update frontend table
        documents = await app.db.get_user_documents(user_id)
        return MyDocuments(documents=documents).documents_table()

    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        raise

@rt("/delete-document/{document_id}", methods=["DELETE"])
@login_required
async def delete_document(request, document_id: str):
    logger.info(f"Delete document request for document_id: {document_id}")
    user_id = request.session["user_id"]

    try:
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

    return await Layout(
        Settings(
            active_tab=tab,
            settings=settings,
            username=request.session["username"]
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

    return TabContent(
        tab,
        settings,
        username=request.session["username"]
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
        await request.app.db.update_demo_questions(user_id, questions)

    return Redirect("/settings?tab=ranker")

@rt("/api/settings/ranker", methods=["POST"])
@login_required
async def update_ranker(request):
    user_id = request.session["user_id"]
    form = await request.form()
    ranker = form.get("ranker", "colpali")
    await request.app.db.update_user_ranker(user_id, ranker)

    return Redirect("/settings?tab=connection")

@rt("/api/settings/connection", methods=["POST"])
@login_required
async def update_connection_settings(request):
    user_id = request.session["user_id"]
    form = await request.form()

    settings = {
        'vespa_host': form.get('vespa_host'),
        'vespa_port': int(form.get('vespa_port')) if form.get('vespa_port') else None,
        'vespa_token': form.get('vespa_token'),
        'gemini_token': form.get('gemini_token'),
        'vespa_cloud_endpoint': form.get('vespa_cloud_endpoint')
    }

    await request.app.db.update_connection_settings(user_id, settings)

    return Redirect("/settings?tab=prompt")

@rt("/api/settings/prompt", methods=["POST"])
@login_required
async def update_prompt_settings(request):
    if request.session["username"] != "admin":
        return Redirect("/settings?tab=demo-questions")

    form = await request.form()
    prompt = form.get('prompt')
    await request.app.db.update_prompt_settings(request.session["user_id"], prompt)

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

if __name__ == "__main__":
    HOT_RELOAD = os.getenv("HOT_RELOAD", "False").lower() == "true"
    logger.info(f"Starting app with hot reload: {HOT_RELOAD}")
    serve(port=7860, reload=HOT_RELOAD)
