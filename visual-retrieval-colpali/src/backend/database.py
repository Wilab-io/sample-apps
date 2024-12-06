from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.schema import CreateTable
from sqlalchemy import select, delete
from uuid import UUID
import os
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from .models import User, UserDocument, UserSettings, RankerType
from .base import Base
import logging
from pathlib import Path

DATABASE_URL = (
    f"postgresql+asyncpg://"
    f"{os.getenv('POSTGRES_USER', 'postgres')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'postgres')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'postgres')}"
)

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

STORAGE_DIR = Path("storage/user_documents")

class Database:
    def __init__(self):
        self.session_maker = async_session

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        async with self.get_session() as session:
            result = await session.execute(
                select(User).where(User.user_id == user_id)
            )
            return result.scalar_one_or_none()

    async def fetch_one(self, query, *args):
        async with self.get_session() as session:
            result = await session.execute(query, args)
            return result.mappings().first()

    async def close(self):
        await engine.dispose()

    async def init_tables(self):
        """Create tables if they don't exist"""
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_create_table_sql(self):
        """Get SQL to create tables (for debugging)"""
        return "\n".join(
            str(CreateTable(table).compile(engine))
            for table in Base.metadata.tables.values()
        )

    async def get_user_documents(self, user_id: UUID):
        """Get all documents for a user"""
        logger = logging.getLogger("vespa_app")
        logger.debug(f"Database: Fetching documents for user_id: {user_id}")
        async with self.get_session() as session:
            result = await session.execute(
                select(UserDocument).where(UserDocument.user_id == user_id)
            )
            documents = result.scalars().all()
            logger.debug(f"Database: Found {len(documents)} documents")
            return documents

    async def add_user_document(self, user_id: str, document_name: str, file_content: bytes):
        """Add a new document to both filesystem and database"""
        logger = logging.getLogger("vespa_app")
        logger.debug(f"Adding document {document_name} for user {user_id}")

        try:
            file_ext = Path(document_name).suffix.lower()
            if file_ext not in ['.pdf', '.png', '.jpg', '.jpeg']:
                raise ValueError(f"Unsupported file type: {file_ext}")

            async with self.get_session() as session:
                new_document = UserDocument(
                    user_id=UUID(user_id),
                    document_name=document_name,
                    file_extension=file_ext,
                )
                session.add(new_document)
                await session.commit()
                await session.refresh(new_document)

                user_dir = STORAGE_DIR / str(user_id)
                user_dir.mkdir(parents=True, exist_ok=True)

                save_path = user_dir / f"{new_document.document_id}{file_ext}"
                save_path.write_bytes(file_content)

                logger.debug(f"Successfully added document {document_name} with ID {new_document.document_id}")
                return new_document

        except Exception as e:
            logger.error(f"Error adding document {document_name}: {str(e)}")
            if 'new_document' in locals():
                await self.delete_document(new_document.document_id)
            raise

    async def delete_document(self, document_id: str):
        """Delete a document from both database and filesystem"""
        logger = logging.getLogger("vespa_app")
        logger.debug(f"Deleting document {document_id}")

        try:
            async with self.get_session() as session:
                result = await session.execute(
                    select(UserDocument).where(UserDocument.document_id == document_id)
                )
                document = result.scalar_one_or_none()

                if not document:
                    logger.warning(f"Document {document_id} not found in database")
                    return

                storage_dir = Path("storage/user_documents")
                file_path = storage_dir / str(document.user_id) / f"{document_id}{document.file_extension}"
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted file {file_path}")

                await session.execute(
                    delete(UserDocument).where(UserDocument.document_id == document_id)
                )
                await session.commit()
                logger.info(f"Deleted database entry for document {document_id}")

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise

    async def get_demo_questions(self, user_id: str) -> list[str]:
        """Get demo questions for a user"""
        async with self.get_session() as session:
            result = await session.execute(
                select(UserSettings).where(UserSettings.user_id == UUID(user_id))
            )
            settings = result.scalar_one_or_none()
            return settings.demo_questions if settings else []

    async def get_user_settings(self, user_id: str) -> UserSettings:
        """Get user settings, creating default settings if they don't exist"""
        async with self.get_session() as session:
            user_id_uuid = UUID(user_id)

            result = await session.execute(
                select(UserSettings).where(UserSettings.user_id == user_id_uuid)
            )
            settings = result.scalar_one_or_none()

            if not settings:
                settings = UserSettings(
                    user_id=user_id_uuid,
                    ranker=RankerType.colpali,
                    prompt=self.get_default_prompt()
                )
                session.add(settings)
                await session.commit()

            return settings

    async def update_settings(self, user_id: str, settings: dict) -> None:
        """Update settings for a user"""
        async with self.get_session() as session:
            user_id_uuid = UUID(user_id)

            result = await session.execute(
                select(UserSettings).where(UserSettings.user_id == user_id_uuid)
            )
            user_settings = result.scalar_one_or_none()

            if user_settings:
                for key, value in settings.items():
                    setattr(user_settings, key, value)
            else:
                user_settings = UserSettings(
                    user_id=user_id_uuid,
                    **settings
                )
                session.add(user_settings)

            await session.commit()

    @staticmethod
    def get_default_prompt() -> str:
        return """You are an investor, stock analyst and financial expert. You will be presented an image of a document page from a report published by the Norwegian Government Pension Fund Global (GPFG). The report may be annual or quarterly reports, or policy reports, on topics such as responsible investment, risk etc.
Your task is to generate retrieval queries and questions that you would use to retrieve this document (or ask based on this document) in a large corpus.
Please generate 3 different types of retrieval queries and questions.
A retrieval query is a keyword based query, made up of 2-5 words, that you would type into a search engine to find this document.
A question is a natural language question that you would ask, for which the document contains the answer.
The queries should be of the following types:
1. A broad topical query: This should cover the main subject of the document.
2. A specific detail query: This should cover a specific detail or aspect of the document.
3. A visual element query: This should cover a visual element of the document, such as a chart, graph, or image.

Important guidelines:
- Ensure the queries are relevant for retrieval tasks, not just describing the page content.
- Use a fact-based natural language style for the questions.
- Frame the queries as if someone is searching for this document in a large corpus.
- Make the queries diverse and representative of different search strategies.

Format your response as a JSON object with the structure of the following example:
{
    "broad_topical_question": "What was the Responsible Investment Policy in 2019?",
    "broad_topical_query": "responsible investment policy 2019",
    "specific_detail_question": "What is the percentage of investments in renewable energy?",
    "specific_detail_query": "renewable energy investments percentage",
    "visual_element_question": "What is the trend of total holding value over time?",
    "visual_element_query": "total holding value trend"
}

If there are no relevant visual elements, provide an empty string for the visual element question and query.
Here is the document image to analyze:
Generate the queries based on this image and provide the response in the specified JSON format.
Only return JSON. Don't return any extra explanation text."""
