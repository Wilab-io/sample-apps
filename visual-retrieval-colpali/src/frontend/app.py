from typing import Optional
from urllib.parse import quote_plus

from fasthtml.components import (
    H1,
    H2,
    H3,
    Br,
    Div,
    Form,
    Img,
    NotStr,
    P,
    Hr,
    Span,
    A,
    Script,
    Button,
    Ul,
    Li,
    Strong,
    Iframe,
)
from fasthtml.xtend import A, Script
from lucide_fasthtml import Lucide
from shad4fast import Badge, Button, Input, Separator

# JavaScript to check the input value and enable/disable the search button and radio buttons
check_input_script = Script(
    """
        window.onload = function() {
            const input = document.getElementById('search-input');
            const button = document.querySelector('[data-button="search-button"]');
            const radioGroupItems = document.querySelectorAll('button[data-ref="radio-item"]');  // Get all radio buttons

            function checkInputValue() {
                const isInputEmpty = input.value.trim() === "";
                button.disabled = isInputEmpty;  // Disable the submit button
                radioGroupItems.forEach(item => {
                    item.disabled = isInputEmpty;  // Disable/enable the radio buttons
                });
            }

            input.addEventListener('input', checkInputValue);  // Listen for input changes
            checkInputValue();  // Initial check when the page loads
        };
    """
)

# JavaScript to handle the image swapping, reset button, and active class toggling
image_swapping = Script(
    """
    document.addEventListener('click', function (e) {
        if (e.target.classList.contains('sim-map-button') || e.target.classList.contains('reset-button')) {
            const imgContainer = e.target.closest('.relative');
            const overlayContainer = imgContainer.querySelector('.overlay-container');
            const newSrc = e.target.getAttribute('data-image-src');

            // If it's a reset button, remove the overlay image
            if (e.target.classList.contains('reset-button')) {
                overlayContainer.innerHTML = '';  // Clear the overlay container, showing only the full image
            } else {
                // Create a new overlay image
                const img = document.createElement('img');
                img.src = newSrc;
                img.classList.add('overlay-image', 'absolute', 'top-0', 'left-0', 'w-full', 'h-full');
                overlayContainer.innerHTML = '';  // Clear any previous overlay
                overlayContainer.appendChild(img);  // Add the new overlay image
            }

            // Toggle active class on buttons
            const activeButton = document.querySelector('.sim-map-button.active');
            if (activeButton) {
                activeButton.classList.remove('active');
            }
            if (e.target.classList.contains('sim-map-button')) {
                e.target.classList.add('active');
            }
        }
    });
    """
)

toggle_text_content = Script(
    """
    function toggleTextContent(idx) {
        const textColumn = document.getElementById(`text-column-${idx}`);
        const imageTextColumns = document.getElementById(`image-text-columns-${idx}`);
        const toggleButton = document.getElementById(`toggle-button-${idx}`);

        if (textColumn.classList.contains('md-grid-text-column')) {
          // Hide the text column
          textColumn.classList.remove('md-grid-text-column');
          imageTextColumns.classList.remove('grid-image-text-columns');
          toggleButton.innerText = `Show Text`;
        } else {
          // Show the text column
          textColumn.classList.add('md-grid-text-column');
          imageTextColumns.classList.add('grid-image-text-columns');
          toggleButton.innerText = `Hide Text`;
        }
    }
    """
)

autocomplete_script = Script(
    """
    document.addEventListener('DOMContentLoaded', function() {
        const input = document.querySelector('#search-input');
        const awesomplete = new Awesomplete(input, { minChars: 1, maxItems: 5 });

        input.addEventListener('input', function() {
            if (this.value.length >= 1) {
                // Use template literals to insert the input value dynamically in the query parameter
                fetch(`/suggestions?query=${encodeURIComponent(this.value)}`)
                    .then(response => response.json())
                    .then(data => {
                        // Update the Awesomplete list dynamically with fetched suggestions
                        awesomplete.list = data.suggestions;
                    })
                    .catch(err => console.error('Error fetching suggestions:', err));
            }
        });
    });
    """
)

dynamic_elements_scrollbars = Script(
    """
    (function () {
        const { applyOverlayScrollbars, getScrollbarTheme } = OverlayScrollbarsManager;

        function applyScrollbarsToDynamicElements() {
            const scrollbarTheme = getScrollbarTheme();

            // Apply scrollbars to dynamically loaded result-text-full and result-text-snippet elements
            const resultTextFullElements = document.querySelectorAll('[id^="result-text-full"]');
            const resultTextSnippetElements = document.querySelectorAll('[id^="result-text-snippet"]');

            resultTextFullElements.forEach(element => {
                applyOverlayScrollbars(element, scrollbarTheme);
            });

            resultTextSnippetElements.forEach(element => {
                applyOverlayScrollbars(element, scrollbarTheme);
            });
        }

        // Apply scrollbars after dynamic content is loaded (e.g., after search results)
        applyScrollbarsToDynamicElements();

        // Observe changes in the 'dark' class to adjust the theme dynamically if needed
        const observer = new MutationObserver(applyScrollbarsToDynamicElements);
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    })();
    """
)

submit_form_on_radio_change = Script(
    """
    document.addEventListener('click', function (e) {
        // if target has data-ref="radio-item" and type is button
        if (e.target.getAttribute('data-ref') === 'radio-item' && e.target.type === 'button') {
            console.log('Radio button clicked');
            const form = e.target.closest('form');
            form.submit();
        }
    });
    """
)


def ShareButtons():
    title = "Visual RAG over PDFs with Vespa and ColPali"
    url = "https://huggingface.co/spaces/vespa-engine/colpali-vespa-visual-retrieval"
    return Div(
        A(
            Img(src="/static/img/linkedin.svg", aria_hidden="true", cls="h-[21px]"),
            "Share on LinkedIn",
            href=f"https://www.linkedin.com/sharing/share-offsite/?url={quote_plus(url)}",
            rel="noopener noreferrer",
            target="_blank",
            cls="bg-[#0A66C2] text-white inline-flex items-center gap-x-1.5 px-2.5 py-1.5 border rounded-md text-sm font-semibold",
        ),
        A(
            Img(src="/static/img/x.svg", aria_hidden="true", cls="h-[21px]"),
            "Share on X",
            href=f"https://twitter.com/intent/tweet?text={quote_plus(title)}&url={quote_plus(url)}",
            rel="noopener noreferrer",
            target="_blank",
            cls="bg-black text-white inline-flex items-center gap-x-1.5 px-2.5 py-1.5 border rounded-md text-sm font-semibold",
        ),
        cls="flex items-center justify-center space-x-8 mt-5",
    )


class SearchBox:
    grid_cls = "grid gap-2 p-3 rounded-md border border-input bg-muted w-full ring-offset-background focus-within:outline-none focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 focus-within:border-input rounded-[10px]"

    def __init__(self, query_value: str = "", ranking_value: str = "colpali", is_deployed: bool = False):
        self.query_value = query_value
        self.ranking_value = ranking_value
        self.is_deployed = is_deployed

    def __ft__(self):
        return Div(
            Form(
                Div(
                    Input(
                        type="search",
                        name="query",
                        placeholder="Setup and deploy the application to use the search feature" if not self.is_deployed else "Search...",
                        value=self.query_value,
                        cls="w-full px-4 py-2 text-lg border rounded-[10px] bg-background disabled:opacity-50 disabled:cursor-not-allowed",
                        autofocus=True,
                        disabled=not self.is_deployed
                    ),
                    Button(
                        Lucide(icon="camera", size="20"),
                        type="button",
                        cls="absolute right-2 top-1/2 -translate-y-1/2 p-2 hover:bg-muted rounded-full",
                        disabled=not self.is_deployed,
                        onclick="document.getElementById('image-upload').click()"
                    ),
                    Input(
                        type="file",
                        id="image-upload",
                        name="image",
                        accept="image/*",
                        cls="hidden",
                        onchange="handleImageUpload(this)"
                    ),
                    cls="relative"
                ),
                P(
                    f"Ranking by: {self.ranking_value.title()}" if self.is_deployed else "",
                    cls="text-sm text-muted-foreground mt-2"
                ),
                check_input_script,
                autocomplete_script,
                submit_form_on_radio_change,
                action=f"/search?query={quote_plus(self.query_value)}&ranking={quote_plus(self.ranking_value)}",
                method="GET",
                hx_get=f"/fetch_results?query={quote_plus(self.query_value)}&ranking={quote_plus(self.ranking_value)}",
                hx_trigger="load",
                hx_target="#search-results",
                hx_swap="outerHTML",
                hx_indicator="#loading-indicator",
                cls=self.grid_cls,
            ),
            cls="py-8 w-full"
        )


async def SampleQueries(request=None):
    sample_queries = []

    if request and "user_id" in request.session:
        user_id = request.session["user_id"]
        sample_queries = await request.app.db.get_demo_questions(user_id)

    # If no questions in DB, don't show anything
    if not sample_queries:
        return Div()

    query_badges = []
    for query in sample_queries:
        query_badges.append(
            A(
                Badge(
                    Div(
                        Lucide(
                            icon="text-search", size="18", cls="text-muted-foreground"
                        ),
                        Span(query, cls="text-base font-normal"),
                        cls="flex gap-2 items-center",
                    ),
                    variant="outline",
                    cls="text-base font-normal text-muted-foreground hover:border-black dark:hover:border-white",
                ),
                href=f"/search?query={quote_plus(query)}",
                cls="no-underline",
            )
        )

    return Div(*query_badges, cls="grid gap-2 justify-items-center")


def Hero():
    return Div(
        H1(
            "Visual RAG over PDFs",
            cls="text-5xl md:text-6xl font-bold tracking-wide md:tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-black to-slate-700 dark:from-white dark:to-slate-300 animate-fade-in",
        ),
        P(
            "See how Vespa and ColPali can be used for Visual RAG in this demo",
            cls="text-base md:text-2xl text-muted-foreground md:tracking-wide",
        ),
        cls="grid gap-5 text-center",
    )


async def Home(request, ranker: str = "colpali", app_deployed: bool = False):
    return Div(
        Div(
            Hero(),
            SearchBox(ranking_value=ranker, is_deployed=app_deployed),
            await SampleQueries(request),
            ShareButtons(),
            cls="grid gap-8 content-start mt-[13vh]",
        ),
        cls="grid w-full h-full max-w-screen-md gap-4 mx-auto",
    )


def LinkResource(text, href):
    return Li(
        A(
            Lucide(icon="external-link", size="18"),
            text,
            href=href,
            target="_blank",
            cls="flex items-center gap-1.5 hover:underline bold text-md",
        ),
    )


def AboutThisDemo():
    resources = [
        {
            "text": "Vespa Blog: How we built this demo",
            "href": "https://blog.vespa.ai/visual-rag-in-practice",
        },
        {
            "text": "Notebook to set up Vespa application and feed dataset",
            "href": "https://pyvespa.readthedocs.io/en/latest/examples/visual_pdf_rag_with_vespa_colpali_cloud.html",
        },
        {
            "text": "Web App (FastHTML) Code",
            "href": "https://github.com/vespa-engine/sample-apps/tree/master/visual-retrieval-colpali",
        },
        {
            "text": "Vespa Blog: Scaling ColPali to Billions",
            "href": "https://blog.vespa.ai/scaling-colpali-to-billions/",
        },
        {
            "text": "Vespa Blog: Retrieval with Vision Language Models",
            "href": "https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/",
        },
    ]
    return Div(
        H1(
            "About This Demo",
            cls="text-3xl md:text-5xl font-bold tracking-wide md:tracking-wider",
        ),
        P(
            "This demo showcases a Visual Retrieval-Augmented Generation (RAG) application over PDFs using ColPali embeddings in Vespa, built entirely in Python, using FastHTML. The code is fully open source.",
            cls="text-base",
        ),
        Img(
            src="/static/img/colpali_child.png",
            alt="Example of token level similarity map",
            cls="w-full",
        ),
        H2("Resources", cls="text-2xl font-semibold"),
        Ul(
            *[
                LinkResource(resource["text"], resource["href"])
                for resource in resources
            ],
            cls="space-y-2 list-disc pl-5",
        ),
        H2("Architecture Overview", cls="text-2xl font-semibold"),
        Img(
            src="/static/img/visual-retrieval-demoapp-arch.png",
            alt="Architecture Overview",
            cls="w-full",
        ),
        Ul(
            Li(
                Strong("Vespa Application: "),
                "Vespa Application that handles indexing, search, ranking and queries, leveraging features like phased ranking and multivector MaxSim calculations.",
            ),
            Li(
                Strong("Frontend: "),
                "Built with FastHTML, offering a professional and responsive user interface without the complexity of separate frontend frameworks.",
            ),
            Li(
                Strong("Backend: "),
                "Also built with FastHTML. Handles query embedding inference using ColPali, serves static files, and is responsible for orchestrating interactions between Vespa and the frontend.",
            ),
            Li(
                Strong("Gemini API: "),
                "VLM for the AI response, providing responses based on the top results from Vespa.",
                cls="list-disc list-inside",
            ),
            H2("User Experience Highlights", cls="text-2xl font-semibold"),
            Ul(
                Li(
                    Strong("Fast and Responsive: "),
                    "Optimized for quick loading times, with phased content delivery to display essential information immediately while loading detailed data in the background.",
                ),
                Li(
                    Strong("Similarity Maps: "),
                    "Provides visual highlights of the most relevant parts of a page in response to a query, enhancing interpretability.",
                ),
                Li(
                    Strong("Type-Ahead Suggestions: "),
                    "Offers query suggestions to assist users in formulating effective searches.",
                ),
                cls="list-disc list-inside",
            ),
            cls="grid gap-5",
        ),
        H2("Dataset", cls="text-2xl font-semibold"),
        P(
            "The dataset used in this demo is retrieved from reports published by the Norwegian Government Pension Fund Global. It contains 6,992 pages from 116 PDF reports (2000–2024). The information is often presented in visual formats, making it an ideal dataset for visual retrieval applications.",
            cls="text-base",
        ),
        Iframe(
            src="https://huggingface.co/datasets/vespa-engine/gpfg-QA/embed/viewer",
            frameborder="0",
            width="100%",
            height="500",
        ),
        Hr(),  # To add some margin to bottom. Probably a much better way to do this, but the mb-[16vh] class doesn't seem to be applied
        cls="w-full h-full max-w-screen-md gap-4 mx-auto mt-[8vh] mb-[16vh] grid gap-8 content-start",
    )


def Search(request, search_results=[]):
    query_value = request.query_params.get("query", "").strip()
    ranking_value = request.query_params.get("ranking", "colpali")
    query_id = request.query_params.get("query_id", "")

    return Div(
        Div(
            SearchBox(query_value=query_value, ranking_value=ranking_value, is_deployed=True),
            Div(
                Div(
                    LoadingMessage(),
                    id="search-results",
                ),
                cls="grid gap-4 w-full max-w-screen-xl mx-auto px-4",
            ),
        ),
    )


def LoadingMessage(display_text="Retrieving search results"):
    return Div(
        Lucide(icon="loader-circle", cls="size-5 mr-1.5 animate-spin"),
        Span(display_text, cls="text-base text-center"),
        cls="p-10 text-muted-foreground flex items-center justify-center",
        id="loading-indicator",
    )


def LoadingSkeleton():
    return Div(
        Div(cls="h-5 bg-muted"),
        Div(cls="h-5 bg-muted"),
        Div(cls="h-5 bg-muted"),
        cls="grid gap-2 animate-pulse",
    )


def SimMapButtonReady(query_id, idx, token, token_idx, img_src):
    return Button(
        token.replace("\u2581", ""),
        size="sm",
        data_image_src=img_src,
        id=f"sim-map-button-{query_id}-{idx}-{token_idx}-{token}",
        cls="sim-map-button pointer-events-auto font-mono text-xs h-5 rounded-none px-2",
    )


def SimMapButtonPoll(query_id, idx, token, token_idx):
    return Button(
        Lucide(icon="loader-circle", size="15", cls="animate-spin"),
        size="sm",
        disabled=True,
        hx_get=f"/get_sim_map?query_id={query_id}&idx={idx}&token={token}&token_idx={token_idx}",
        hx_trigger="every 0.5s",
        hx_swap="outerHTML",
        cls="pointer-events-auto text-xs h-5 rounded-none px-2",
    )


def SearchInfo(search_time, total_count):
    return Div(
        Span(
            "Retrieved ",
            Strong(total_count),
            Span(" results"),
            Span(" in "),
            Strong(f"{search_time:.3f}"),  # 3 significant digits
            Span(" seconds."),
        ),
        cls="text-sm text-center p-3",
    )


def ResultsList(results: list, query: str, query_id: Optional[str] = None, search_time: float = 0, total_count: int = 0):
    if not results:
        return Div(
            P(
                "No results found for your query.",
                cls="text-muted-foreground text-base text-center",
            ),
            cls="grid p-10",
        )

    result_items = []
    doc_ids = []
    for idx, result in enumerate(results):
        fields = result["fields"]
        doc_id = fields["id"]
        doc_ids.append(doc_id)

        result_items.append(
            A(
                Div(
                    Div(
                        H2(fields["title"], cls="text-lg font-semibold text-blue-500"),
                        Div(
                            Badge(
                                f"Relevance score: {result['relevance']:.4f}",
                                cls="text-sm rounded-full text-white bg-black border-none p-2",
                            ),
                            Badge(
                                "Best match",
                                cls="text-sm rounded-full text-white bg-green-500 border-none p-2",
                            ) if idx == 0 else None,
                            cls="flex gap-2 items-center",
                        ),
                        cls="flex justify-between items-center",
                    ),
                    cls="p-4 hover:bg-muted transition-colors rounded-[10px]",
                ),
                href=f"/detail?doc_id={doc_id}&query_id={query_id}&query={quote_plus(query)}",
                cls="bg-white dark:bg-gray-900 rounded-[10px] shadow-md border border-gray-200 dark:border-gray-700 no-underline block",
            )
        )

    return Div(
        SearchInfo(search_time, total_count),
        Div(
            Div(
                ChatResult(query_id=query_id, query=query, doc_ids=doc_ids),
                cls="bg-white dark:bg-gray-900 rounded-[10px] shadow-md shadow-green-500 border-4 border-green-500 dark:border-green-700 p-4",
            ),
            cls="mb-8",
        ),
        Div(
            *result_items,
            cls="grid gap-4 mt-4",
        ),
        id="search-results",
        cls="w-full max-w-screen-xl mx-auto px-4",
    )


def ChatResult(query_id: str, query: str, doc_ids: Optional[list] = None):
    messages = Div(LoadingSkeleton())

    if doc_ids:
        messages = Div(
            LoadingSkeleton(),
            hx_ext="sse",
            sse_connect=f"/get-message?query_id={query_id}&doc_ids={','.join(doc_ids)}&query={quote_plus(query)}",
            sse_swap="message",
            sse_close="close",
            hx_swap="innerHTML",
        )

    return Div(
        Div("AI-response (Gemini-8B)", cls="text-xl font-semibold p-5"),
        Div(
            Div(
                messages,
            ),
            id="chat-messages",
            cls="overflow-auto min-h-0 grid items-end px-5",
        ),
        id="chat_messages",
        cls="h-full grid grid-rows-[auto_1fr_auto] min-h-0 gap-3",
    )


def SearchResult(
    results: list,
    query: str,
    query_id: Optional[str] = None,
    search_time: float = 0,
    total_count: int = 0,
    doc_id: Optional[str] = None,
):
    if not results:
        return Div(
            P(
                "No results found for your query.",
                cls="text-muted-foreground text-base text-center",
            ),
            cls="grid p-10",
        )
    # If no doc_id is provided, show the results list view
    if doc_id is None:
        return ResultsList(results, query, query_id, search_time, total_count)

    # Otherwise, find the specific result and show the detail view
    result = next((r for r in results if r["fields"]["id"] == doc_id), None)
    if not result:
        return Div(
            P(
                "Document not found.",
                cls="text-muted-foreground text-base text-center",
            ),
            cls="grid p-10",
        )

    fields = result["fields"]
    blur_image_base64 = f"data:image/jpeg;base64,{fields['blur_image']}"

    sim_map_fields = {
        key: value
        for key, value in fields.items()
        if key.startswith("sim_map_")
    }

    # Generate buttons for the sim_map fields
    sim_map_buttons = []
    for key, value in sim_map_fields.items():
        token = key.split("_")[-2]
        token_idx = int(key.split("_")[-1])
        if value is not None:
            sim_map_base64 = f"data:image/jpeg;base64,{value}"
            sim_map_buttons.append(
                SimMapButtonReady(
                    query_id=query_id,
                    idx=0,  # Since we're showing a single result
                    token=token,
                    token_idx=token_idx,
                    img_src=sim_map_base64,
                )
            )
        else:
            sim_map_buttons.append(
                SimMapButtonPoll(
                    query_id=query_id,
                    idx=0,  # Since we're showing a single result
                    token=token,
                    token_idx=token_idx,
                )
            )

    # Add "Reset Image" button
    reset_button = Button(
        "Reset",
        variant="outline",
        size="sm",
        data_image_src=blur_image_base64,
        cls="reset-button pointer-events-auto font-mono text-xs h-5 rounded-none px-2",
    )

    tokens_icon = Lucide(icon="images", size="15")
    tokens_button = Button(
        tokens_icon,
        "Tokens",
        size="sm",
        cls="tokens-button flex gap-[3px] font-bold pointer-events-none font-mono text-xs h-5 rounded-none px-2",
    )

    return Div(
        Div(
            SearchBox(query_value=query, ranking_value="colpali", is_deployed=True),
            Div(
                Div(
                    Div(
                        A(
                            Lucide(icon="arrow-left"),
                            href=f"/search?query={quote_plus(query)}",
                            cls="text-sm hover:underline rounded-full text-white bg-black border-none p-1",
                        ),
                        Lucide(icon="file-text"),
                        H2(fields["title"], cls="text-xl md:text-2xl font-semibold"),
                        Separator(orientation="vertical"),
                        Badge(
                            f"Relevance score: {result['relevance']:.4f}",
                            cls="flex gap-1.5 items-center justify-center",
                        ),
                        cls="flex items-center gap-2",
                    ),
                    Div(
                        Button(
                            "Hide Text",
                            size="sm",
                            id=f"toggle-button-0",
                            onclick=f"toggleTextContent(0)",
                            cls="hidden md:block",
                        ),
                    ),
                    cls="flex flex-wrap items-center justify-between bg-background px-3 py-4",
                ),
                Div(
                    Div(
                        Div(
                            tokens_button,
                            *sim_map_buttons,
                            reset_button,
                            cls="flex flex-wrap gap-px w-full pointer-events-none",
                        ),
                        Div(
                            Div(
                                Div(
                                    Img(
                                        src=blur_image_base64,
                                        hx_get=f"/full_image?doc_id={doc_id}",
                                        style="backdrop-filter: blur(5px);",
                                        hx_trigger="load",
                                        hx_swap="outerHTML",
                                        alt=fields["title"],
                                        cls="result-image w-full h-full object-contain",
                                    ),
                                    Div(
                                        cls="overlay-container absolute top-0 left-0 w-full h-full pointer-events-none"
                                    ),
                                    cls="relative w-full h-full",
                                ),
                                cls="grid bg-muted p-2",
                            ),
                            cls="block",
                        ),
                        id=f"image-column-0",
                        cls="image-column relative bg-background px-3 py-5 grid-image-column",
                    ),
                    Div(
                        Div(
                            A(
                                Lucide(icon="external-link", size="18"),
                                f"PDF Source (Page {fields['page_number'] + 1})",
                                href=f"{fields['url']}#page={fields['page_number'] + 1}",
                                target="_blank",
                                cls="flex items-center gap-1.5 font-mono bold text-sm",
                            ),
                            cls="flex items-center justify-end",
                        ),
                        Div(
                            Div(
                                Div(
                                    Div(
                                        Div(
                                            H3(
                                                "Dynamic summary",
                                                cls="text-base font-semibold",
                                            ),
                                            P(
                                                NotStr(fields.get("snippet", "")),
                                                cls="text-highlight text-muted-foreground",
                                            ),
                                            cls="grid grid-rows-[auto_0px] content-start gap-y-3",
                                        ),
                                        id=f"result-text-snippet-0",
                                        cls="grid gap-y-3 p-8 border border-dashed",
                                    ),
                                    Div(
                                        Div(
                                            Div(
                                                H3(
                                                    "Full text",
                                                    cls="text-base font-semibold",
                                                ),
                                                Div(
                                                    P(
                                                        NotStr(fields.get("text", "")),
                                                        cls="text-highlight text-muted-foreground",
                                                    ),
                                                    Br(),
                                                ),
                                                cls="grid grid-rows-[auto_0px] content-start gap-y-3",
                                            ),
                                            id=f"result-text-full-0",
                                            cls="grid gap-y-3 p-8 border border-dashed",
                                        ),
                                        Div(
                                            cls="absolute inset-x-0 bottom-0 bg-gradient-to-t from-[#fcfcfd] dark:from-[#1c2024] pt-[7%]"
                                        ),
                                        cls="relative grid",
                                    ),
                                    cls="grid grid-rows-[1fr_1fr] xl:grid-rows-[1fr_2fr] gap-y-8 p-8 text-sm",
                                ),
                                cls="grid bg-background",
                            ),
                            cls="grid bg-muted p-2",
                        ),
                        id=f"text-column-0",
                        cls="text-column relative bg-background px-3 py-5 hidden md-grid-text-column",
                    ),
                    id=f"image-text-columns-0",
                    cls="relative grid grid-cols-1 border-t grid-image-text-columns",
                ),
                cls="grid grid-cols-1 grid-rows-[auto_auto_1fr]",
            ),
            cls="grid gap-4 w-full max-w-screen-xl mx-auto px-4",
        ),
    )
