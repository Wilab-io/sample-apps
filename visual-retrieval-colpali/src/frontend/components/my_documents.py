from fasthtml.common import Button, Div, H1, Form, Input, Img
from shad4fast import (
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
)
from datetime import datetime

class MyDocuments:
    def __init__(self, documents=None):
        self.documents = documents

    def documents_table(self):
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
                )
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
                            "📄 " + doc.document_name,
                            cls="p-4"
                        ),
                        TableCell(
                            doc.upload_ts.strftime("%Y-%m-%d %H:%M"),
                            cls="p-4 text-muted-foreground"
                        ),
                        TableCell(
                            Button(
                                Img(
                                    src="/static/img/trash.svg",
                                    alt="Delete",
                                    cls="h-4 w-4 dark:brightness-0 dark:invert"
                                ),
                                type="button",
                                cls="hover:opacity-80",
                                hx_delete=f"/delete-document/{doc.document_id}",
                                hx_target="#documents-list",
                                hx_confirm=f"Are you sure you want to delete {doc.document_name}?"
                            ),
                            cls="p-4"
                        ),
                    )
                    for doc in self.documents
                ])
            ),
        )

    async def __call__(self):
        return Div(
            H1("Uploaded documents", cls="text-4xl font-bold mb-8 text-center"),
            Form(
                Input(
                    type="file",
                    name="files",
                    multiple=True,
                    accept=".pdf",
                    cls="hidden",
                    id="file-input",
                    hx_trigger="change",
                    hx_post="/upload-files",
                    hx_encoding="multipart/form-data",
                    hx_target="#documents-list",
                ),
                Button(
                    "Upload new",
                    type="button",
                    cls="bg-black dark:bg-gray-900 text-white px-6 py-2 rounded-[10px] hover:opacity-80",
                    onclick="document.getElementById('file-input').click()"
                ),
                cls="flex justify-end mb-4"
            ),
            Div(
                self.documents_table(),
                cls="bg-white dark:bg-gray-900 rounded-[10px] shadow-lg overflow-hidden",
                id="documents-list"
            ),
            cls="container mx-auto max-w-4xl p-8"
        )
