from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.schema import CreateTable
from sqlalchemy import select, delete
from uuid import UUID
import os
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from .models import User, UserDocument
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
            async with self.get_session() as session:
                new_document = UserDocument(
                    user_id=UUID(user_id),
                    document_name=document_name,
                )
                session.add(new_document)
                await session.commit()
                await session.refresh(new_document)

                user_dir = STORAGE_DIR / str(user_id)
                user_dir.mkdir(parents=True, exist_ok=True)

                file_ext = Path(document_name).suffix
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
            # First get document info to know the user_id for file path
            async with self.get_session() as session:
                result = await session.execute(
                    select(UserDocument).where(UserDocument.document_id == document_id)
                )
                document = result.scalar_one_or_none()

                if not document:
                    logger.warning(f"Document {document_id} not found in database")
                    return

                storage_dir = Path("storage/user_documents")
                file_path = storage_dir / str(document.user_id) / f"{document_id}.pdf"
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
