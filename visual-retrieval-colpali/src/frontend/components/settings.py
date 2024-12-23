from fasthtml.common import Div, H1, H2, Input, Main, Button, P, Form, Label, Span, Textarea
from lucide_fasthtml import Lucide
from pathlib import Path
from backend.models import UserSettings, User
from shad4fast import (
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
)

def TabButton(text: str, value: str, active_tab: str):
    is_active = value == active_tab
    return Div(
        text,
        cls=f"""
            px-4 py-2 rounded-[10px] cursor-pointer
            {
                'bg-white dark:bg-gray-900 text-black dark:text-white' if is_active
                else 'text-gray-500 dark:text-gray-400 opacity-50'
            }
        """,
        **{
            "hx-get": f"/settings/content?tab={value}",
            "hx-push-url": f"/settings?tab={value}",
            "hx-target": "#settings-content",
            "hx-swap": "outerHTML"
        }
    )

def TabButtons(active_tab: str, username: str = None, appConfigured: bool = False):
    return Div(
        Div(
            TabButton("Demo questions", "demo-questions", active_tab),
            TabButton("Ranker", "ranker", active_tab),
            TabButton("Connection", "connection", active_tab),
            TabButton("Application package", "application-package", active_tab),
            TabButton("Prompt", "prompt", active_tab) if username == "admin" else None,
            TabButton("Users", "users", active_tab) if username == "admin" else None,
            cls="flex gap-2 p-1 bg-gray-100 dark:bg-gray-800 rounded-[10px]",
        ),
        Div(
            Button(
                "Deploy",
                cls="bg-black dark:bg-black text-white px-6 py-2 rounded-[10px] hover:opacity-80",
                id="deploy-button",
                **{
                    "hx-post": "/api/deploy-part-1",
                    "hx-swap": "none"
                }
            ) if appConfigured else Button(
                "Deploy",
                cls="bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 px-6 py-2 rounded-[10px]",
                id="deploy-button",
                disabled=not appConfigured,
            ),
            Div(
                Lucide(
                    "info",
                    cls="size-5 cursor-pointer ml-6 dark:brightness-0 dark:invert"
                ),
                P(
                    "All required settings must be set\nbefore deploying the application.",
                    cls="absolute invisible group-hover:visible bg-white dark:bg-gray-900 text-black dark:text-white p-3 rounded-[10px] text-sm -mt-12 ml-2 shadow-sm min-w-[300px] max-w-[300px]"
                ),
                cls="relative inline-block group"
            ),
            cls="flex items-center"
        ),
        cls="flex justify-between items-center mb-8 gap-4",
        id="tab-buttons"
    )

def TabContent(active_tab: str, settings: UserSettings = None, users: list[User] = None, username: str = None, appConfigured: bool = False):
    return Div(
        TabButtons(active_tab, username, appConfigured),
        Div(
            _get_tab_content(active_tab, settings, users),
            cls="bg-white dark:bg-gray-900 p-4 rounded-[10px] shadow-md w-full border border-gray-200 dark:border-gray-700",
        ),
        id="settings-content"
    )

def _get_tab_content(active_tab: str, settings: UserSettings = None, users: list[User] = None):
    if active_tab == "demo-questions":
        return DemoQuestions(questions=settings.demo_questions if settings else [])
    elif active_tab == "ranker":
        return RankerSettings(ranker=settings.ranker if settings else None)
    elif active_tab == "connection":
        return ConnectionSettings(settings=settings)
    elif active_tab == "application-package":
        return ApplicationPackageSettings(settings=settings)
    elif active_tab == "prompt":
        return PromptSettings(settings=settings)
    elif active_tab == "users":
        return UsersSettings(users=users)
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
                "Save",
                cls="mt-6 bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 px-6 py-2 rounded-[10px] disabled-next",
                id="save-questions-disabled",
                disabled=True,
            ),
            Button(
                "Save",
                cls="mt-6 bg-black dark:bg-black text-white px-6 py-2 rounded-[10px] hover:opacity-80 enabled-next hidden",
                id="save-questions",
                hx_post="/api/settings/demo-questions",
                hx_trigger="click",
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
                        cls="mr-2",
                        **{"data-original": ranker}
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
                Div(
                    P(
                        "Unsaved changes",
                        cls="text-red-500 text-sm hidden text-right mt-6",
                        id="ranker-unsaved-changes"
                    ),
                    cls="flex-grow self-center"
                ),
                Button(
                    "Save",
                    cls="mt-6 bg-black dark:bg-black text-white px-6 py-2 rounded-[10px] hover:opacity-80",
                    id="save-ranker",
                    type="submit",
                    hx_post="/api/settings/ranker",
                    hx_trigger="click",
                ),
                cls="flex items-center w-full gap-4"
            ),
        ),
        cls="space-y-4"
    )

def ConnectionSettings(settings: UserSettings = None):
    has_api_key = False
    current_filename = None
    if settings:
        user_key_dir = Path("storage/user_keys") / str(settings.user_id)
        if user_key_dir.exists():
            pem_files = list(user_key_dir.glob("*.pem"))
            has_api_key = len(pem_files) > 0
            if has_api_key:
                current_filename = pem_files[0].name

    return Div(
        Div(
            H2("Connection settings", cls="text-xl font-semibold px-4 mb-4"),
            cls="border-b border-gray-200 dark:border-gray-700 -mx-4 mb-6"
        ),
        Form(
            Div(
                Div(
                    Div(
                        H2("Tokens", cls="text-lg font-semibold mb-4"),
                        Div(
                            Label(
                                "Gemini API token ",
                                Span("*", cls="text-red-500"),
                                htmlFor="gemini-token",
                                cls="text-sm font-medium"
                            ),
                            Input(
                                value=settings.gemini_token if settings else '',
                                cls="flex-1 w-full rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background",
                                name="gemini_token",
                                required=True,
                                **{"data-original": settings.gemini_token if settings else ''}
                            ),
                            cls="space-y-2 mb-4"
                        ),
                        Div(
                            Label(
                                "API key file ",
                                Span("*", cls="text-red-500"),
                                htmlFor="api-key-file",
                                cls="text-sm font-medium"
                            ),
                            P(
                                "Generate a random key on Vespa Console and upload it here",
                                cls="text-sm text-gray-500 mb-2"
                            ),
                            Div(
                                Input(
                                    type="file",
                                    accept=".pem",
                                    cls="flex-1 w-full rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background",
                                    name="api_key_file",
                                    required=not has_api_key,
                                    hx_post="/api/settings/connection",
                                    hx_encoding="multipart/form-data",
                                    hx_trigger="change",
                                    **{"data-has-file": "true"} if has_api_key else {}
                                ),
                                P(
                                    f"Current file: {current_filename}" if current_filename else None,
                                    cls="text-sm text-gray-500 mt-2"
                                ) if current_filename else None,
                                cls="space-y-2"
                            ),
                            cls="space-y-2 mb-4"
                        ),
                        cls="mb-8"
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
                    "Save",
                    cls="mt-6 bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 px-6 py-2 rounded-[10px] disabled-next",
                    id="save-connection-disabled",
                    disabled=True,
                ),
                Button(
                    "Save",
                    cls="mt-6 bg-black dark:bg-black text-white px-6 py-2 rounded-[10px] hover:opacity-80 enabled-next hidden",
                    id="save-connection",
                    type="submit",
                    hx_post="/api/settings/connection",
                    hx_trigger="click",
                ),
                cls="flex items-center w-full gap-4"
            ),
            cls="space-y-4",
        ),
        cls="space-y-4"
    )

def ApplicationPackageSettings(settings: UserSettings = None):
    return Div(
        Div(
            H2("Application package settings", cls="text-xl font-semibold px-4 mb-4"),
            cls="border-b border-gray-200 dark:border-gray-700 -mx-4 mb-6"
        ),
        Form(
            Div(
                Div(
                    Div(
                        Div(
                            Label(
                                "Tenant name ",
                                Span("*", cls="text-red-500"),
                                htmlFor="tenant-name",
                                cls="text-sm font-medium"
                            ),
                            P(
                                "Create a tenant from vespa.ai/free-trial the trial includes $300 credit",
                                cls="text-sm text-gray-500 mb-2"
                            ),
                            Input(
                                value=settings.tenant_name if settings else '',
                                cls="flex-1 w-full rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background",
                                name="tenant_name",
                                required=True,
                                **{"data-original": settings.tenant_name if settings else ''}
                            ),
                            cls="space-y-2 mb-4"
                        ),
                        Div(
                            Label(
                                "Application name ",
                                Span("*", cls="text-red-500"),
                                htmlFor="app-name",
                                cls="text-sm font-medium"
                            ),
                            P(
                                "Only lowercase letters and numbers allowed, no spaces or special characters, start with a letter",
                                cls="text-sm text-gray-500 mb-2"
                            ),
                            Input(
                                value=settings.app_name if settings else '',
                                cls="flex-1 w-full rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background",
                                name="app_name",
                                required=True,
                                pattern="[a-z0-9]+",
                                title="Only lowercase letters and numbers allowed, no spaces or special characters, start with a letter",
                                **{"data-original": settings.app_name if settings else ''}
                            ),
                            cls="space-y-2 mb-4"
                        ),
                        Div(
                            Label(
                                "Instance name ",
                                Span("*", cls="text-red-500"),
                                htmlFor="instance-name",
                                cls="text-sm font-medium"
                            ),
                            Input(
                                value=settings.instance_name if settings else '',
                                cls="flex-1 w-full rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background",
                                name="instance_name",
                                required=True,
                                **{"data-original": settings.instance_name if settings else ''}
                            ),
                            cls="space-y-2 mb-4"
                        ),
                        Div(
                            Label(
                                "Schema ",
                                Span("*", cls="text-red-500"),
                                htmlFor="schema",
                                cls="text-sm font-medium"
                            ),
                            Textarea(
                                settings.schema if settings else '',
                                cls="flex-1 w-full h-[200px] rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background",
                                name="schema",
                                required=True,
                                **{"data-original": settings.schema if settings else ''}
                            ),
                            cls="space-y-2 mb-4"
                        ),
                        cls="mb-8"
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
                        id="application-package-unsaved-changes"
                    ),
                    cls="flex-grow self-center"
                ),
                Button(
                    "Save",
                    cls="mt-6 bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 px-6 py-2 rounded-[10px] disabled-next",
                    id="save-application-package-disabled",
                    disabled=True,
                ),
                Button(
                    "Save",
                    cls="mt-6 bg-black dark:bg-black text-white px-6 py-2 rounded-[10px] hover:opacity-80 enabled-next hidden",
                    id="save-application-package",
                    type="submit",
                    hx_post="/api/settings/application-package",
                    hx_trigger="click",
                ),
                cls="flex items-center w-full gap-4"
            ),
            cls="space-y-4",
        ),
        cls="space-y-4"
    )

def PromptSettings(settings: UserSettings = None):
    return Div(
        Div(
            H2("Custom prompt setting", cls="text-xl font-semibold px-4 mb-4"),
            cls="border-b border-gray-200 dark:border-gray-700 -mx-4 mb-6"
        ),
        Form(
            Div(
                Label("Custom prompt", htmlFor="prompt", cls="text-lg font-semibold"),
                Textarea(
                    settings.prompt if settings else '',
                    cls="flex-1 w-full h-[400px] rounded-[10px] border border-input bg-background px-3 py-2 text-sm ring-offset-background mt-4",
                    name="prompt",
                    id="prompt",
                    **{"data-original": settings.prompt if settings else ''}
                ),
                cls="space-y-2"
            ),
            Div(
                Div(
                    P(
                        "Unsaved changes",
                        cls="text-red-500 text-sm hidden text-right mt-6",
                        id="prompt-unsaved-changes"
                    ),
                    cls="flex-grow self-center"
                ),
                Button(
                    "Save",
                    cls="mt-6 bg-black dark:bg-black text-white px-6 py-2 rounded-[10px] enabled:hover:opacity-80",
                    type="submit",
                    hx_post="/api/settings/prompt",
                    hx_trigger="click",
                ),
                cls="flex items-center w-full gap-4"
            ),
            cls="space-y-4",
        ),
        cls="space-y-4"
    )


def UsersSettings(users: list = None):
    users = users if users else []

    return Div(
        Div(
            H2("Users settings", cls="text-xl font-semibold px-4 mb-4"),
            cls="border-b border-gray-200 dark:border-gray-700 -mx-4 mb-6"
        ),
        Table(
            TableHeader(
                TableRow(
                    TableHead("Username", cls="text-left p-4"),
                    TableHead("Password", cls="text-left p-4"),
                    TableHead("Actions", cls="text-left p-4"),
                )
            ),
            TableBody(
                *[
                    TableRow(
                        TableCell(
                            user.get("username", ""),
                            cls="p-4",
                            name=f"username_{i}",
                            **{"data-original": user.get("username", "")}
                        ),
                        TableCell(
                            Input(
                                type="hidden",
                                placeholder=user.get("password", ""),
                                value=user.get("password", ""),
                                name=f"password_{i}",
                                **{"data-original": user.get("password", "")}
                            ),
                            "********",
                            name=f"password_{i}",
                            cls="p-4"
                        ),
                        TableCell(
                            user.get("user_id", None),
                            cls="p-4",
                            name=f"user_id_{i}",
                            hidden=True
                        ),
                        TableCell(
                            Button(
                                Lucide("trash-2", size=20),
                                variant="ghost",
                                size="icon",
                                cls="delete-user ml-2" if not user.get("new") else "hidden",
                                **{"data-user-id": user.get("user_id", None)}
                            ) if (i > 0 and not user.get("username") == "admin") else None,
                            cls="flex items-center mb-2"
                        ),
                    **{"data-user-id": user.get("user_id")}
                    )
                    for i, user in enumerate(users)
                ]
            ),
            id="users-container",
            cls="w-full"
        ),
        Button(
            "Add User",
            id="add-user",
            variant="default",
            cls="mt-1 ml-auto rounded-[10px] border border-gray-200 dark:border-gray-700 px-3 py-2"
        ),
        Div(
            Div(
                P(
                    "Unsaved changes",
                    cls="text-red-500 text-sm hidden text-right mt-6",
                    id="unsaved-changes-users"
                ),
                cls="flex-grow self-center"
            ),
            Button(
                "Save",
                cls="mt-6 bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 px-6 py-2 rounded-[10px] disabled-next-users",
                id="save-users-disabled",
                disabled=True,
            ),
            Button(
                "Save",
                cls="mt-6 bg-black dark:bg-black text-white px-6 py-2 rounded-[10px] hover:opacity-80 enabled-next-users hidden",
                id="save-users",
                hx_post="/api/settings/users",
                hx_trigger="click",
            ),
            cls="flex items-center w-full gap-4"
        ),
        cls="space-y-4"
    )



def Settings(active_tab: str = "demo-questions", settings: UserSettings = None, users: list[User] = None, username: str = None, appConfigured: bool = False):
    return Main(
        H1("Settings", cls="text-4xl font-bold mb-8 text-center"),
        Div(
            TabContent(active_tab, settings, users, username, appConfigured),
            cls="w-full max-w-4xl mx-auto"
        ),
        cls="container mx-auto px-4 py-8 w-full min-h-0"
    )
