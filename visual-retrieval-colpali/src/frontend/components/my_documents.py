from fasthtml.common import Button, Div, H1, Form, Input, P, H2
from shad4fast import (
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
)
from lucide_fasthtml import Lucide

class MyDocuments:
    def __init__(self, documents=None, app_deployed=False):
        self.documents = documents
        self.app_deployed = app_deployed

    def documents_table(self):
        def get_file_icon(file_extension: str) -> str:
            if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
                return "🏞️"
            return "📄"

        return Table(
            TableHeader(
                TableRow(
                    TableHead(
                        "Document name",
                        cls="text-left p-4"
                    ),
                    TableHead(
                        "Upload time",
                        cls="text-left p-4"
                    ),
                    TableHead(
                        "Actions",
                        cls="text-left p-4"
                    ),
                ),
                cls="border-b border-gray-200 dark:border-gray-700"
            ),
            TableBody(
                *([
                    TableRow(
                        TableCell(
                            "No documents uploaded",
                            colSpan="3",
                            cls="text-center p-4 text-muted-foreground"
                        )
                    )
                ] if not self.documents else [
                    TableRow(
                        TableCell(
                            get_file_icon(doc.file_extension) + " " + doc.document_name,
                            cls="p-4"
                        ),
                        TableCell(
                            doc.upload_ts.strftime("%Y-%m-%d %H:%M"),
                            cls="p-4 text-muted-foreground"
                        ),
                        TableCell(
                            Button(
                                Lucide("trash-2", cls="dark:brightness-0 dark:invert", size='20'),
                                type="button",
                                cls="hover:opacity-80",
                                hx_delete=f"/delete-document/{doc.document_id}",
                                hx_target="#documents-list",
                                hx_confirm=f"Are you sure you want to delete {doc.document_name}?"
                            ),
                            cls="p-4"
                        ),
                        cls="border-b border-gray-200 dark:border-gray-700"
                    )
                    for doc in self.documents
                ])
            ),
        )

    async def __call__(self):
        return Div(
            H1("Uploaded documents", cls="text-4xl font-bold mb-8 text-center"),
            Div(
                Form(
                    Input(
                        type="file",
                        name="files",
                        multiple=True,
                        accept=".pdf,.png,.jpg,.jpeg",
                        cls="hidden",
                        id="file-input",
                        hx_trigger="change",
                        hx_post="/upload-files",
                        hx_encoding="multipart/form-data"
                    ),
                    Div(
                        Button(
                            "Upload new",
                            type="button",
                            cls="bg-black dark:bg-gray-900 text-white px-6 py-2 rounded-[10px] hover:opacity-80",
                            onclick="document.getElementById('file-input').click()"
                        ) if self.app_deployed else Button(
                            "Upload new",
                            cls="bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 px-6 py-2 rounded-[10px]",
                            id="deploy-button",
                            disabled=True,
                        ),
                        Div(
                            Lucide(
                                "info",
                                cls="size-5 cursor-pointer ml-6 dark:brightness-0 dark:invert"
                            ),
                            P(
                                "The app must be deployed to upload documents. Supported formats: PDF, PNG, JPG, JPEG. Other formats will be ignored.",
                                cls="absolute invisible group-hover:visible bg-white dark:bg-gray-900 text-black dark:text-white p-3 rounded-[10px] text-sm -mt-12 ml-2 shadow-sm min-w-[300px] max-w-[300px]"
                            ),
                            cls="relative inline-block group"
                        ),
                        cls="flex items-center"
                    ),
                    cls="flex justify-end mb-4"
                ),
                Div(
                    self.documents_table(),
                    cls="bg-white dark:bg-gray-900 rounded-[10px] shadow-md overflow-hidden border border-gray-200 dark:border-gray-700",
                    id="documents-list"
                ),
                cls="container mx-auto max-w-4xl p-8"
            )
        )

def DocumentProcessingModal():
    return Div(
        Div(
            Div(
                H2(
                    "Processing the uploaded documents",
                    cls="text-xl font-semibold mb-2 text-gray-900 dark:text-white"
                ),
                P(
                    "Processing documents' information with AI to feed Vespa",
                    cls="text-gray-500 dark:text-gray-400"
                ),
                Div(
                    Lucide(icon="loader-circle", cls="size-10 animate-spin"),
                    cls="mt-6 flex justify-center"
                ),
                cls="bg-white dark:bg-gray-900 p-8 rounded-[10px] shadow-md max-w-md w-full text-center"
            ),
            cls="fixed inset-0 flex items-center justify-center z-50 p-4"
        ),
        cls="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-40"
    )

def DocumentProcessingErrorModal(message: str):
    return Div(
        Div(
            Div(
                Div(
                    Lucide(icon="circle-x", cls="size-12 text-red-500"),
                    cls="flex justify-center mb-4"
                ),
                H2(
                    "Document processing failed",
                    cls="text-xl font-semibold mb-2 text-gray-900 dark:text-white"
                ),
                P(
                    "The document processing failed, check the console logs for more details",
                    cls="text-gray-500 dark:text-gray-400 mb-6"
                ) if message is None else P(
                    message,
                    cls="text-gray-500 dark:text-gray-400 mb-6"
                ),
                Button(
                    "OK",
                    cls="w-full p-4 bg-black text-white rounded-[10px] hover:bg-gray-800 transition-colors",
                    onclick="document.getElementById('document-processing-modal').remove(); window.location.href = '/my-documents';"
                ),
                cls="bg-white dark:bg-gray-900 p-8 rounded-[10px] shadow-md max-w-md w-full text-center"
            ),
            cls="fixed inset-0 flex items-center justify-center z-50 p-4"
        ),
        cls="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-40"
    )
