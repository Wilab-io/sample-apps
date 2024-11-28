from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.schema import CreateTable
from sqlalchemy import select
from uuid import UUID
import os
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from .models import User
from .base import Base

# Connection URL
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
