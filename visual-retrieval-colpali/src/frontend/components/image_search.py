from fasthtml.components import Div, P
from lucide_fasthtml import Lucide

def ImageSearchModal():
    return Div(
        Div(
            Div(
                Div(
                    Lucide(icon="image-plus", cls="size-12 text-blue-500"),
                    P("Processing image", cls="text-xl font-semibold"),
                    P(
                        "Generating metadata to perform the search",
                        cls="text-sm text-muted-foreground"
                    ),
                    Div(
                        Lucide(
                            icon="loader-circle",
                            size=24,
                            cls="animate-spin text-primary"
                        ),
                        cls="mt-4"
                    ),
                    cls="grid place-items-center gap-4 text-center p-6"
                ),
                cls="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white dark:bg-gray-900 p-8 rounded-[10px] shadow-md max-w-md w-full text-center",
            ),
            cls="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-40 flex items-center justify-center",
        ),
        id="image-search-modal",
        cls="relative z-50",
    )
