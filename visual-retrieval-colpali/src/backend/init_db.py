from sqlalchemy import select
from .database import engine, async_session
from .auth import hash_password
from .models import User, Base
import logging

async def init_admin_user(logger: logging.Logger):
    # First create tables if they don't exist
    async with engine.begin() as conn:
        try:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise

    async with async_session() as session:
        try:
            result = await session.execute(
                select(User).where(User.username == "admin")
            )
            user = result.scalar_one_or_none()

            if user is None:
                logger.info("Creating admin user...")
                admin_user = User(
                    username="admin",
                    password_hash=hash_password("admin")  # Generate hash at runtime
                )
                session.add(admin_user)
                await session.commit()
                logger.info("Admin user created successfully")
            else:
                logger.info("Admin user already exists")
                logger.debug(f"Existing user hash: {user.password_hash}")

        except Exception as e:
            logger.error(f"Error in init_admin_user: {e}")
            raise
