import json

from fasthtml.common import *
from shad4fast import *
from vespa.application import Vespa

from backend.colpali import load_model, get_result_dummy
from backend.vespa_app import get_vespa_app
from frontend.app import Home, Search
from frontend.layout import Layout

highlight_js_theme_link = Link(id='highlight-theme', rel="stylesheet", href="")
highlight_js_theme = Script(src="/static/js/highlightjs-theme.js")
highlight_js = HighlightJS(langs=['python', 'javascript', 'java', 'json', 'xml'], dark="github-dark", light="github")

app, rt = fast_app(
    htmlkw={'cls': "h-full"},
    pico=False,
    hdrs=(
        ShadHead(tw_cdn=False, theme_handle=True),
        highlight_js,
        highlight_js_theme_link,
        highlight_js_theme
    ),
)
vespa_app: Vespa = get_vespa_app()

# Initialize variables to None
model = None
processor = None


# Define a function to get the model and processor
def get_model_and_processor():
    global model, processor
    if model is None or processor is None:
        model, processor = load_model()
    return model, processor


@rt("/static/{filepath:path}")
def serve_static(filepath: str):
    return FileResponse(f'./static/{filepath}')


@rt("/")
def get():
    return Layout(Home())


@rt("/search")
def get():
    return Layout(Search())


@rt("/app")
def get():
    return Layout(
        Div(
            P(f"Connected to Vespa at {vespa_app.url}"),
            cls="p-4"
        )
    )


@rt("/run_query")
def get(query: str, nn: bool = False):
    # dummy-function to avoid running the query every time
    result = get_result_dummy(query, nn)
    # If we want to run real, uncomment the following lines
    # model, processor = get_model_and_processor()
    # result = asyncio.run(
    #     get_result_from_query(vespa_app, processor=processor, model=model, query=query, nn=nn)
    # )
    return Layout(
        Div(
            H1("Result"),
            Pre(Code(json.dumps(result, indent=2))),
            cls="p-4"
        )
    )


serve()