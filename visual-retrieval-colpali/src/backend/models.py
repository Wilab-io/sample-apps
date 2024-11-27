from sqlalchemy import String, DateTime, ARRAY, Enum, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from .database import Base
import uuid
from typing import Optional

class User(Base):
    __tablename__ = "app_user"

    user_id: Mapped[UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)

class UserDocument(Base):
    __tablename__ = "user_document"

    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[UUID] = mapped_column(UUID, nullable=False)
    document_name: Mapped[str] = mapped_column(String, nullable=False)
    upload_ts: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class UserSettings(Base):
    __tablename__ = "user_settings"

    user_id: Mapped[UUID] = mapped_column(UUID, primary_key=True)
    demo_questions: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    ranker: Mapped[str] = mapped_column(Enum('colpali', 'bm25', 'hybrid-colpali-bm25', name='ranker_type'), nullable=False, default='colpali')
    vespa_host: Mapped[Optional[str]] = mapped_column(String)
    vespa_port: Mapped[Optional[int]] = mapped_column(String)
    vespa_token: Mapped[Optional[str]] = mapped_column(String)
    gemini_token: Mapped[Optional[str]] = mapped_column(String)
    vespa_cloud_endpoint: Mapped[Optional[str]] = mapped_column(String)
    schema: Mapped[Optional[str]] = mapped_column(String)
    prompt: Mapped[Optional[str]] = mapped_column(String)
