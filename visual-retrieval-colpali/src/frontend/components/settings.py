from fasthtml.common import Div, H1, H2, Input, Main, Button, P, Form, Label, Span
from lucide_fasthtml import Lucide
from backend.models import UserSettings

def TabButton(text: str, value: str, active_tab: str):
    is_active = value == active_tab
    return Div(
        text,
        cls=f"""
            px-4 py-2 rounded-[10px]
            {
                'bg-white dark:bg-gray-900 text-black dark:text-white' if is_active
                else 'text-gray-500 dark:text-gray-400 opacity-50'
            }
            {'cursor-default'}  # Never show pointer cursor since tabs aren't clickable
        """
    )

def TabButtons(active_tab: str):
    return Div(
        Div(
            TabButton("Demo questions", "demo-questions", active_tab),
            TabButton("Ranker", "ranker", active_tab),
            TabButton("Connection", "connection", active_tab),
            TabButton("Application package", "application-package", active_tab),
            cls="flex gap-2 p-1 bg-gray-100 dark:bg-gray-800 rounded-[10px]",
        ),
        Button(
            "Deploy",
            disabled=True,
            cls="bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 px-6 py-2 rounded-[10px]"
        ),
        cls="flex justify-between items-center mb-8 gap-4",
        id="tab-buttons"
    )

def TabContent(active_tab: str, settings: UserSettings = None):
    return Div(
        TabButtons(active_tab),
        Div(
            _get_tab_content(active_tab, settings),
            cls="bg-white dark:bg-gray-900 p-4 rounded-[10px] shadow-md w-full border border-gray-200 dark:border-gray-700",
        ),
        id="settings-content"
    )

def _get_tab_content(active_tab: str, settings: UserSettings = None):
    if active_tab == "demo-questions":
        return DemoQuestions(questions=settings.demo_questions if settings else [])
    elif active_tab == "ranker":
        return RankerSettings(ranker=settings.ranker if settings else None)
    elif active_tab == "connection":
        return ConnectionSettings(settings=settings)
    elif active_tab == "application-package":
        return "Application package settings coming soon..."
    return ""

def DemoQuestions(questions: list[str]):
    if not questions:
        questions = [""]

    return Div(
        Div(
            H2("Homepage demo questions", cls="text-xl font-semibold px-4 mb-4"),
            cls="border-b border-gray-200 dark:border-gray-700 -mx-4 mb-6"
        ),
        Div(
            *[
                Div(
                    Input(
                        value=q,
                        cls="flex-1 w-full rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background",
                        name=f"question_{i}",
                        **{"data-original": q}
                    ),
                    Button(
                        Lucide("trash-2", size=20),
                        variant="ghost",
                        size="icon",
                        cls="delete-question ml-2",
                    ) if i > 0 else None,
                    cls="flex items-center mb-2",
                )
                for i, q in enumerate(questions)
            ],
            id="questions-container",
            cls="space-y-2"
        ),
        Button(
            "Add question",
            id="add-question",
            variant="default",
            cls="mt-1 ml-auto rounded-[10px] border border-gray-200 dark:border-gray-700 px-3 py-2"
        ),
        Div(
            Div(
                P(
                    "Unsaved changes",
                    cls="text-red-500 text-sm hidden text-right mt-6",
                    id="unsaved-changes"
                ),
                cls="flex-grow self-center"
            ),
            Button(
                "Next",
                cls="mt-6 bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 px-6 py-2 rounded-[10px] disabled-next",
                id="save-questions-disabled",
                disabled=True,
            ),
            Button(
                "Next",
                cls="mt-6 bg-black dark:bg-black text-white px-6 py-2 rounded-[10px] hover:opacity-80 enabled-next hidden",
                id="save-questions",
                **{
                    "hx-post": "/api/settings/demo-questions",
                    "hx-trigger": "click"
                }
            ),
            cls="flex items-center w-full gap-4"
        ),
        cls="space-y-4"
    )

def RankerSettings(ranker: str = "colpali"):
    if hasattr(ranker, 'value'):
        ranker = ranker.value

    return Div(
        Div(
            H2("Results ranker selection", cls="text-xl font-semibold px-4 mb-4"),
            cls="border-b border-gray-200 dark:border-gray-700 -mx-4 mb-6"
        ),
        Form(
            Div(
                Div(
                    Input(
                        type="radio",
                        id="colpali",
                        name="ranker",
                        value="colpali",
                        checked=ranker == "colpali",
                        cls="mr-2"
                    ),
                    "ColPali",
                    cls="flex items-center space-x-2"
                ),
                Div(
                    Input(
                        type="radio",
                        id="bm25",
                        name="ranker",
                        value="bm25",
                        checked=ranker == "bm25",
                        cls="mr-2"
                    ),
                    "BM25",
                    cls="flex items-center space-x-2 mb-4"
                ),
                Div(
                    Input(
                        type="radio",
                        id="hybrid",
                        name="ranker",
                        value="hybrid",
                        checked=ranker == "hybrid",
                        cls="mr-2"
                    ),
                    "Hybrid ColPali + BM25",
                    cls="flex items-center space-x-2 mb-4"
                ),
                cls="space-y-2 mb-8"
            ),
            Div(
                Button(
                    "Next",
                    cls="mt-6 bg-black dark:bg-black text-white px-6 py-2 rounded-[10px] hover:opacity-80",
                    id="save-ranker",
                    type="submit"
                ),
                cls="flex justify-end w-full"
            ),
            **{
                "hx-post": "/api/settings/ranker",
                "hx-trigger": "submit"
            }
        ),
        cls="space-y-4"
    )

def ConnectionSettings(settings: UserSettings = None):
    return Div(
        Div(
            H2("Connection settings", cls="text-xl font-semibold px-4 mb-4"),
            cls="border-b border-gray-200 dark:border-gray-700 -mx-4 mb-6"
        ),
        Form(
            Div(
                # Content wrapper with max-width to limit input width
                Div(
                    Div(
                        H2("Vespa.ai endpoint connection", cls="text-lg font-semibold mb-4"),
                        Div(
                            Label(
                                "Vespa.ai host ",
                                Span("*", cls="text-red-500"),
                                htmlFor="vespa-host",
                                cls="text-sm font-medium"
                            ),
                            Input(
                                value=settings.vespa_host if settings else '',
                                cls="flex-1 w-full rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background",
                                name="vespa_host",
                                required=True
                            ),
                            cls="space-y-2 mb-4"
                        ),
                        Div(
                            Label(
                                "Vespa.ai port ",
                                Span("*", cls="text-red-500"),
                                htmlFor="vespa-port",
                                cls="text-sm font-medium"
                            ),
                            Input(
                                value=settings.vespa_port if settings else '',
                                cls="flex-1 w-full rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background",
                                name="vespa_port",
                                type="number",
                                required=True
                            ),
                            cls="space-y-2 mb-4"
                        ),
                        cls="mb-8"
                    ),
                    Div(
                        H2("Tokens", cls="text-lg font-semibold mb-4"),
                        Div(
                            Label(
                                "Vespa.ai token ",
                                Span("*", cls="text-red-500"),
                                htmlFor="vespa-token",
                                cls="text-sm font-medium"
                            ),
                            Input(
                                value=settings.vespa_token if settings else '',
                                cls="flex-1 w-full rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background",
                                name="vespa_token",
                                required=True
                            ),
                            cls="space-y-2 mb-4"
                        ),
                        Div(
                            Label(
                                "Gemini token ",
                                Span("*", cls="text-red-500"),
                                htmlFor="gemini-token",
                                cls="text-sm font-medium"
                            ),
                            Input(
                                value=settings.gemini_token if settings else '',
                                cls="flex-1 w-full rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background",
                                name="gemini_token",
                                required=True
                            ),
                            cls="space-y-2 mb-4"
                        ),
                        cls="mb-8"
                    ),
                    Div(
                        H2("Vespa Cloud endpoint", cls="text-lg font-semibold mb-4"),
                        Div(
                            Label("Endpoint URL", htmlFor="vespa-cloud-endpoint", cls="text-sm font-medium"),
                            Input(
                                value=settings.vespa_cloud_endpoint if settings else '',
                                cls="flex-1 w-full rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background",
                                name="vespa_cloud_endpoint"
                            ),
                            cls="space-y-2"
                        ),
                    ),
                    cls="max-w-[50%]"
                ),
                cls="w-full"
            ),
            Div(
                Div(
                    P(
                        "Unsaved changes",
                        cls="text-red-500 text-sm hidden text-right mt-6",
                        id="connection-unsaved-changes"
                    ),
                    cls="flex-grow self-center"
                ),
                Button(
                    "Next",
                    cls="mt-6 bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 px-6 py-2 rounded-[10px] disabled-next",
                    id="save-connection-disabled",
                    disabled=True,
                ),
                Button(
                    "Next",
                    cls="mt-6 bg-black dark:bg-black text-white px-6 py-2 rounded-[10px] hover:opacity-80 enabled-next hidden",
                    id="save-connection",
                    type="submit"
                ),
                cls="flex items-center w-full gap-4"
            ),
            cls="space-y-4",
            **{
                "hx-post": "/api/settings/connection",
                "hx-trigger": "submit"
            }
        ),
        cls="space-y-4"
    )

def Settings(active_tab: str = "demo-questions", settings: UserSettings = None):
    return Main(
        H1("Settings", cls="text-4xl font-bold mb-8 text-center"),
        Div(
            TabContent(active_tab, settings),
            cls="w-full max-w-screen-xl mx-auto"
        ),
        cls="container mx-auto px-4 py-8 w-full min-h-0"
    )
