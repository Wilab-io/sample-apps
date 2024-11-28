from fasthtml.common import Button, Div, H1, P
from shad4fast import (
    Table,
    TableBody,
    TableCaption,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
)
from datetime import datetime

class MyDocuments:
    def __init__(self, documents=None):
        self.documents = documents

    async def __call__(self):
        return Div(
            H1("Uploaded documents", cls="text-4xl font-bold mb-8 text-center"),
            # Div(
            #     Button(
            #         "Upload new",
            #         cls="bg-black text-white px-6 py-2 rounded-[10px] hover:bg-gray-800",
            #     ),
            #     cls="flex justify-end mb-4"
            # ),
            Div(
                Table(
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
                        )
                    ),
                    TableBody(
                        *([
                            TableRow(
                                TableCell(
                                    "No documents uploaded",
                                    colSpan="2",
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
                            )
                            for doc in self.documents
                        ])
                    ),
                ),
                cls="bg-white dark:bg-gray-900 rounded-[10px] shadow-lg overflow-hidden"
            ),
            cls="container mx-auto max-w-4xl p-8"
        )
