from fasthtml.common import Button, Div, H1, Form, Input, Img, P
from shad4fast import (
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
)
from lucide_fasthtml import Lucide
from datetime import datetime

class MyDocuments:
    def __init__(self, documents=None):
        self.documents = documents

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
                        hx_encoding="multipart/form-data",
                        hx_target="#documents-list",
                    ),
                    Div(
                        Button(
                            "Upload new",
                            type="button",
                            cls="bg-black dark:bg-gray-900 text-white px-6 py-2 rounded-[10px] hover:opacity-80",
                            onclick="document.getElementById('file-input').click()"
                        ),
                        Div(
                            Lucide(
                                "info",
                                cls="size-5 cursor-pointer ml-6 dark:brightness-0 dark:invert"
                            ),
                            P(
                                "Supported formats: PDF, PNG, JPG, JPEG. Other formats will be ignored.",
                                cls="absolute invisible group-hover:visible bg-white dark:bg-gray-900 text-black dark:text-white p-3 rounded-[10px] text-sm -mt-12 ml-2 shadow-sm min-w-[400px]"
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
