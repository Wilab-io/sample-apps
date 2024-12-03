from fasthtml.common import Div, H1, H2, Input, Main, Button, P
from lucide_fasthtml import Lucide

def TabButton(text: str, value: str, active_tab: str):
    is_active = value == active_tab
    return Div(
        text,
        cls=f"""
            px-4 py-2 rounded-[10px] cursor-pointer
            {
                'bg-white dark:bg-gray-900 text-black dark:text-white' if is_active
                else 'text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
            }
        """,
        **{
            "hx-get": f"/settings/content?tab={value}",
            "hx-push-url": f"/settings?tab={value}",
            "hx-target": "#settings-content",
            "hx-swap": "outerHTML"
        }
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

def TabContent(active_tab: str, questions: list[str] = None):
    return Div(
        TabButtons(active_tab),
        Div(
            _get_tab_content(active_tab, questions),
            cls="bg-white dark:bg-gray-900 p-4 rounded-[10px] shadow-md w-full border border-gray-200 dark:border-gray-700",
        ),
        id="settings-content"
    )

def _get_tab_content(active_tab: str, questions: list[str] = None):
    if active_tab == "demo-questions":
        return DemoQuestions(questions or [])
    elif active_tab == "ranker":
        return "Ranker settings coming soon..."
    elif active_tab == "connection":
        return "Connection settings coming soon..."
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

def Settings(active_tab: str = "demo-questions", questions: list[str] = None):
    return Main(
        H1("Settings", cls="text-4xl font-bold mb-8 text-center"),
        Div(
            TabContent(active_tab, questions),
            cls="w-full max-w-screen-xl mx-auto"
        ),
        cls="container mx-auto px-4 py-8 w-full min-h-0"
    )
