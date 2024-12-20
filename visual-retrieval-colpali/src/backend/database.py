from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker as sessionMaker
from sqlalchemy.schema import CreateTable
from sqlalchemy import select, delete, and_, true
from uuid import UUID
import os
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from .models import User, UserDocument, UserSettings, RankerType, ImageQuery
from .base import Base
import logging
from pathlib import Path
import torch
from .auth import hash_password
import bcrypt

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
async_session = sessionMaker(engine, class_=AsyncSession, expire_on_commit=False)

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

    async def get_user_document_by_id(self, document_id: str):
        """Get a document by ID"""
        async with self.get_session() as session:
            result = await session.execute(
                select(UserDocument).where(UserDocument.document_id == document_id)
            )
            return result.scalar_one_or_none()

    async def add_user_document(self, user_id: str, document_name: str, file_content: bytes):
        """Add a new document to both filesystem and database"""
        logger = logging.getLogger("vespa_app")
        logger.debug(f"Adding document {document_name} for user {user_id}")

        try:
            file_ext = Path(document_name).suffix.lower()
            if file_ext not in ['.pdf', '.png', '.jpg', '.jpeg']:
                raise ValueError(f"Unsupported file type: {file_ext}")

            # Convert PNG to JPG before database operations
            if file_ext != ".jpg" and file_ext != ".pdf":
                from PIL import Image
                from io import BytesIO
                with BytesIO(file_content) as bio:
                    im = Image.open(bio)
                    rgb_im = im.convert('RGB')
                    output_bio = BytesIO()
                    rgb_im.save(output_bio, format='JPEG')
                    file_content = output_bio.getvalue()
                file_ext = '.jpg'

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
                async with self.get_session() as session:
                    result = await session.execute(
                        select(UserDocument).where(UserDocument.document_id == new_document.document_id)
                    )
                    document = result.scalar_one_or_none()
                    if document:
                        await self.delete_document(document.document_id)
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

                file_path = STORAGE_DIR / str(document.user_id) / f"{document_id}{document.file_extension}"
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
        settings = await self.get_user_settings(user_id)
        return settings.demo_questions if settings else []

    async def get_user_settings(self, user_id: str) -> UserSettings:
        """Get user settings, creating default settings if they don't exist"""
        async with self.get_session() as session:
            user_id_uuid = UUID(user_id)

            # First check if user exists
            user_result = await session.execute(
                select(User).where(User.user_id == user_id_uuid)
            )
            user = user_result.scalar_one_or_none()

            if not user:
                return None

            result = await session.execute(
                select(UserSettings).where(UserSettings.user_id == user_id_uuid)
            )
            settings = result.scalar_one_or_none()

            if not settings:
                settings = UserSettings(
                    user_id=user_id_uuid,
                    ranker=RankerType.colpali,
                    prompt=self.get_default_prompt(),
                    schema=self.get_default_schema()
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

    async def get_users_list(self) -> list[User]:
        """Get all users from the app_user table"""
        async with self.get_session() as session:
            result = await session.execute(
                select(User) 
            )
            users = result.scalars().all()
            return [
                {"user_id": str(user.user_id), "username": user.username, "password": user.password_hash}
                for user in users
            ]
       
    async def delete_users(self, user_ids: set) -> None:
        """Delete users and their associated data."""
        logger = logging.getLogger("vespa_app")
        async with self.get_session() as session:
            for user_id in user_ids:
                user_id_uuid = UUID(user_id)
                try:
                    await session.execute(delete(UserSettings).where(UserSettings.user_id == user_id_uuid))
                    logger.info(f"Deleted settings for user ID: {user_id}")
                    await session.execute(delete(UserDocument).where(UserDocument.user_id == user_id_uuid))
                    logger.info(f"Deleted documents for user ID: {user_id}")

                    await session.execute(delete(User).where(User.user_id == user_id_uuid))
                    logger.info(f"Deleted user with ID: {user_id}")
                except Exception as e:
                    logger.error(f"Error deleting user {user_id}: {e}")


    async def create_users(self, users: dict, existing_usernames: set) -> None:
        """Create new users based on input data."""
        logger = logging.getLogger("vespa_app")

        async with self.get_session() as session:
            for user_data in users.values():
                username = user_data.get("username")
                password = user_data.get("password")

                if not username or not password:
                    logger.warning(f"Invalid data for user {username}, skipping.")
                    continue

                if username in existing_usernames:
                    logger.info(f"Username {username} already exists, skipping.")
                    continue

                try:
                    new_user = User(username=username, password_hash=hash_password(password))
                    session.add(new_user)
                    logger.info(f"Created user: {username}")
                except Exception as e:
                    logger.error(f"Error creating user {username}: {e}")

    async def update_users(self, users_data: dict) -> None:
        """Update users by deleting and creating as necessary."""
        logger = logging.getLogger("vespa_app")

        users = {
            key.split("_")[1]: {
                "username": users_data.get(f"username_{key.split('_')[1]}"),
                "password": users_data.get(f"password_{key.split('_')[1]}"),
                "user_id": users_data.get(f"user_id_{key.split('_')[1]}"),
            }
            for key in users_data.keys()
            if key.startswith("username_")
        }
        input_user_ids = {user["user_id"] for user in users.values() if user["user_id"]}

        async with self.get_session() as session:
            db_users = await session.execute(select(User))
            db_users = db_users.scalars().all()
            db_user_ids = {str(user.user_id) for user in db_users}
            existing_usernames = {user.username for user in db_users}

            users_to_delete = db_user_ids - input_user_ids

            await self.delete_users(users_to_delete)

            await self.create_users(users, existing_usernames)

            try:
                await session.commit()
                logger.info("Users updated successfully")
            except Exception as e:
                logger.error(f"Error committing user updates: {e}")
                raise

    async def is_application_configured(self, user_id: str) -> bool:
        documents = await self.get_user_documents(UUID(user_id))
        has_documents = len(documents) > 0

        settings = await self.get_user_settings(user_id)
        if not settings:
            return False

        required_settings = [
            settings.vespa_token_id,
            settings.vespa_token_value,
            settings.gemini_token,
            settings.tenant_name,
            settings.app_name,
            settings.instance_name,
            settings.prompt
        ]

        has_required_settings = all(setting is not None for setting in required_settings)

        return has_documents and has_required_settings

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

    @staticmethod
    def get_default_schema() -> str:
        return """schema pdf_page {
    document pdf_page {
        field id type string {
            indexing: summary | index
            match {
                word
            }
        }
        field url type string {
            indexing: summary | index
        }
        field year type int {
            indexing: summary | attribute
        }
        field title type string {
            indexing: summary | index
            index: enable-bm25
            match {
                text
            }
        }
        field page_number type int {
            indexing: summary | attribute
        }
        field blur_image type raw {
            indexing: summary
        }
        field full_image type raw {
            indexing: summary
        }
        field text type string {
            indexing: summary | index
            index: enable-bm25
            match {
                text
            }
        }
        field embedding type tensor<int8>(patch{}, v[16]) {
            indexing: attribute | index
            attribute {
                distance-metric: hamming
            }
            index {
                hnsw {
                    max-links-per-node: 32
                    neighbors-to-explore-at-insert: 400
                }
            }
        }
        field questions type array<string> {
            indexing: summary | attribute
            summary: matched-elements-only
        }
        field queries type array<string> {
            indexing: summary | attribute
            summary: matched-elements-only
        }
    }
    fieldset default {
        fields: title, text
    }
    rank-profile bm25 {
        inputs {
            query(qt) tensor<float>(querytoken{}, v[128])

        }
        function similarities() {
            expression {

                                sum(
                                    query(qt) * unpack_bits(attribute(embedding)), v
                                )

            }
        }
        function normalized() {
            expression {

                                (similarities - reduce(similarities, min)) / (reduce((similarities - reduce(similarities, min)), max)) * 2 - 1

            }
        }
        function quantized() {
            expression {

                                cell_cast(normalized * 127.999, int8)

            }
        }
        first-phase {
            expression {
                bm25(title) + bm25(text)
            }
        }
    }
    rank-profile bm25_sim inherits bm25 {
        first-phase {
            expression {
                bm25(title) + bm25(text)
            }
        }
        summary-features {
            quantized
        }
    }
    rank-profile colpali {
        inputs {
            query(rq0) tensor<int8>(v[16])
            query(rq1) tensor<int8>(v[16])
            query(rq2) tensor<int8>(v[16])
            query(rq3) tensor<int8>(v[16])
            query(rq4) tensor<int8>(v[16])
            query(rq5) tensor<int8>(v[16])
            query(rq6) tensor<int8>(v[16])
            query(rq7) tensor<int8>(v[16])
            query(rq8) tensor<int8>(v[16])
            query(rq9) tensor<int8>(v[16])
            query(rq10) tensor<int8>(v[16])
            query(rq11) tensor<int8>(v[16])
            query(rq12) tensor<int8>(v[16])
            query(rq13) tensor<int8>(v[16])
            query(rq14) tensor<int8>(v[16])
            query(rq15) tensor<int8>(v[16])
            query(rq16) tensor<int8>(v[16])
            query(rq17) tensor<int8>(v[16])
            query(rq18) tensor<int8>(v[16])
            query(rq19) tensor<int8>(v[16])
            query(rq20) tensor<int8>(v[16])
            query(rq21) tensor<int8>(v[16])
            query(rq22) tensor<int8>(v[16])
            query(rq23) tensor<int8>(v[16])
            query(rq24) tensor<int8>(v[16])
            query(rq25) tensor<int8>(v[16])
            query(rq26) tensor<int8>(v[16])
            query(rq27) tensor<int8>(v[16])
            query(rq28) tensor<int8>(v[16])
            query(rq29) tensor<int8>(v[16])
            query(rq30) tensor<int8>(v[16])
            query(rq31) tensor<int8>(v[16])
            query(rq32) tensor<int8>(v[16])
            query(rq33) tensor<int8>(v[16])
            query(rq34) tensor<int8>(v[16])
            query(rq35) tensor<int8>(v[16])
            query(rq36) tensor<int8>(v[16])
            query(rq37) tensor<int8>(v[16])
            query(rq38) tensor<int8>(v[16])
            query(rq39) tensor<int8>(v[16])
            query(rq40) tensor<int8>(v[16])
            query(rq41) tensor<int8>(v[16])
            query(rq42) tensor<int8>(v[16])
            query(rq43) tensor<int8>(v[16])
            query(rq44) tensor<int8>(v[16])
            query(rq45) tensor<int8>(v[16])
            query(rq46) tensor<int8>(v[16])
            query(rq47) tensor<int8>(v[16])
            query(rq48) tensor<int8>(v[16])
            query(rq49) tensor<int8>(v[16])
            query(rq50) tensor<int8>(v[16])
            query(rq51) tensor<int8>(v[16])
            query(rq52) tensor<int8>(v[16])
            query(rq53) tensor<int8>(v[16])
            query(rq54) tensor<int8>(v[16])
            query(rq55) tensor<int8>(v[16])
            query(rq56) tensor<int8>(v[16])
            query(rq57) tensor<int8>(v[16])
            query(rq58) tensor<int8>(v[16])
            query(rq59) tensor<int8>(v[16])
            query(rq60) tensor<int8>(v[16])
            query(rq61) tensor<int8>(v[16])
            query(rq62) tensor<int8>(v[16])
            query(rq63) tensor<int8>(v[16])
            query(qt) tensor<float>(querytoken{}, v[128])
            query(qtb) tensor<int8>(querytoken{}, v[16])

        }
        function similarities() {
            expression {

                                sum(
                                    query(qt) * unpack_bits(attribute(embedding)), v
                                )

            }
        }
        function normalized() {
            expression {

                                (similarities - reduce(similarities, min)) / (reduce((similarities - reduce(similarities, min)), max)) * 2 - 1

            }
        }
        function quantized() {
            expression {

                                cell_cast(normalized * 127.999, int8)

            }
        }
        function max_sim() {
            expression {

                                sum(
                                    reduce(
                                        sum(
                                            query(qt) * unpack_bits(attribute(embedding)), v
                                        ),
                                        max, patch
                                    ),
                                    querytoken
                                )

            }
        }
        function max_sim_binary() {
            expression {

                                sum(
                                    reduce(
                                        1 / (1 + sum(
                                            hamming(query(qtb), attribute(embedding)), v)
                                        ),
                                        max, patch
                                    ),
                                    querytoken
                                )

            }
        }
        first-phase {
            expression {
                max_sim_binary
            }
        }
        second-phase {
            rerank-count: 10
            expression {
                max_sim
            }
        }
    }
    rank-profile colpali_sim inherits colpali {
        first-phase {
            expression {
                max_sim_binary
            }
        }
        summary-features {
            quantized
        }
    }
    rank-profile hybrid {
        inputs {
            query(rq0) tensor<int8>(v[16])
            query(rq1) tensor<int8>(v[16])
            query(rq2) tensor<int8>(v[16])
            query(rq3) tensor<int8>(v[16])
            query(rq4) tensor<int8>(v[16])
            query(rq5) tensor<int8>(v[16])
            query(rq6) tensor<int8>(v[16])
            query(rq7) tensor<int8>(v[16])
            query(rq8) tensor<int8>(v[16])
            query(rq9) tensor<int8>(v[16])
            query(rq10) tensor<int8>(v[16])
            query(rq11) tensor<int8>(v[16])
            query(rq12) tensor<int8>(v[16])
            query(rq13) tensor<int8>(v[16])
            query(rq14) tensor<int8>(v[16])
            query(rq15) tensor<int8>(v[16])
            query(rq16) tensor<int8>(v[16])
            query(rq17) tensor<int8>(v[16])
            query(rq18) tensor<int8>(v[16])
            query(rq19) tensor<int8>(v[16])
            query(rq20) tensor<int8>(v[16])
            query(rq21) tensor<int8>(v[16])
            query(rq22) tensor<int8>(v[16])
            query(rq23) tensor<int8>(v[16])
            query(rq24) tensor<int8>(v[16])
            query(rq25) tensor<int8>(v[16])
            query(rq26) tensor<int8>(v[16])
            query(rq27) tensor<int8>(v[16])
            query(rq28) tensor<int8>(v[16])
            query(rq29) tensor<int8>(v[16])
            query(rq30) tensor<int8>(v[16])
            query(rq31) tensor<int8>(v[16])
            query(rq32) tensor<int8>(v[16])
            query(rq33) tensor<int8>(v[16])
            query(rq34) tensor<int8>(v[16])
            query(rq35) tensor<int8>(v[16])
            query(rq36) tensor<int8>(v[16])
            query(rq37) tensor<int8>(v[16])
            query(rq38) tensor<int8>(v[16])
            query(rq39) tensor<int8>(v[16])
            query(rq40) tensor<int8>(v[16])
            query(rq41) tensor<int8>(v[16])
            query(rq42) tensor<int8>(v[16])
            query(rq43) tensor<int8>(v[16])
            query(rq44) tensor<int8>(v[16])
            query(rq45) tensor<int8>(v[16])
            query(rq46) tensor<int8>(v[16])
            query(rq47) tensor<int8>(v[16])
            query(rq48) tensor<int8>(v[16])
            query(rq49) tensor<int8>(v[16])
            query(rq50) tensor<int8>(v[16])
            query(rq51) tensor<int8>(v[16])
            query(rq52) tensor<int8>(v[16])
            query(rq53) tensor<int8>(v[16])
            query(rq54) tensor<int8>(v[16])
            query(rq55) tensor<int8>(v[16])
            query(rq56) tensor<int8>(v[16])
            query(rq57) tensor<int8>(v[16])
            query(rq58) tensor<int8>(v[16])
            query(rq59) tensor<int8>(v[16])
            query(rq60) tensor<int8>(v[16])
            query(rq61) tensor<int8>(v[16])
            query(rq62) tensor<int8>(v[16])
            query(rq63) tensor<int8>(v[16])
            query(qt) tensor<float>(querytoken{}, v[128])
            query(qtb) tensor<int8>(querytoken{}, v[16])

        }
        function similarities() {
            expression {

                                sum(
                                    query(qt) * unpack_bits(attribute(embedding)), v
                                )

            }
        }
        function normalized() {
            expression {

                                (similarities - reduce(similarities, min)) / (reduce((similarities - reduce(similarities, min)), max)) * 2 - 1

            }
        }
        function quantized() {
            expression {

                                cell_cast(normalized * 127.999, int8)

            }
        }
        function max_sim() {
            expression {

                                sum(
                                    reduce(
                                        sum(
                                            query(qt) * unpack_bits(attribute(embedding)), v
                                        ),
                                        max, patch
                                    ),
                                    querytoken
                                )

            }
        }
        function max_sim_binary() {
            expression {

                                sum(
                                    reduce(
                                        1 / (1 + sum(
                                            hamming(query(qtb), attribute(embedding)), v)
                                        ),
                                        max, patch
                                    ),
                                    querytoken
                                )

            }
        }
        first-phase {
            expression {
                max_sim_binary
            }
        }
        second-phase {
            rerank-count: 10
            expression {
                max_sim + 2 * (bm25(text) + bm25(title))
            }
        }
    }
    rank-profile hybrid_sim inherits hybrid {
        first-phase {
            expression {
                max_sim_binary
            }
        }
        summary-features {
            quantized
        }
    }
    rank-profile colpali_visual inherits colpali {
        first-phase {
            expression {
                max_sim_binary
            }
            keep-rank-count: 100
        }
        second-phase {
            rerank-count: 25
            expression {
                max_sim
            }
        }
        match-features {
            distance(embedding)
        }
        rank-properties {
            rankOnlyMatchedTerms: true
            termwiseLimit: true
            secondPhaseRerank: true
            matchPhaseLimit: 100
            maxHits: 100
        }
        constants {
            maxTermsPerQuery: 64
        }
    }
    rank-profile colpali_visual_sim inherits colpali_visual {
        summary-features {
            quantized
        }
    }
    document-summary default {
        summary text {
            bolding: on
        }
        summary snippet {
            source: text
            dynamic
        }
        from-disk
    }
    document-summary suggestions {
        summary questions {}
        from-disk
    }
}"""

    async def store_image_query(self, query_id: str, embeddings: torch.Tensor, text: str, is_visual_only: bool) -> str:
        logger = logging.getLogger("vespa_app")
        logger.debug(f"Storing image query with ID {query_id}")

        try:
            # Convert embeddings tensor to numpy array and then to list
            embeddings_list = embeddings.detach().cpu().numpy().tolist()

            async with self.get_session() as session:
                # Create new ImageQuery record
                image_query = ImageQuery(
                    query_id=query_id,
                    embeddings=embeddings_list,
                    text=text,
                    is_visual_only=is_visual_only
                )
                session.add(image_query)
                await session.commit()
                logger.debug(f"Successfully stored image query with ID {query_id}")
                return query_id

        except Exception as e:
            logger.error(f"Error storing image query: {str(e)}")
            raise

    async def get_image_query(self, query_id: str) -> Optional[ImageQuery]:
        async with self.get_session() as session:
            result = await session.execute(
                select(ImageQuery).where(ImageQuery.query_id == query_id)
            )
            return result.scalar_one_or_none()
