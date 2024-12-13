from sqlalchemy import select, text
from .database import engine, async_session
from .auth import hash_password
from .models import User, Base
import logging
from sqlalchemy.exc import SQLAlchemyError
from asyncpg.exceptions import ConnectionDoesNotExistError, CannotConnectNowError, PostgresConnectionError
import sys

async def clear_image_queries(logger: logging.Logger):
    try:
        async with async_session() as session:
            try:
                # Use raw SQL for a simple truncate operation
                await session.execute(text("TRUNCATE TABLE image_queries"))
                await session.commit()
                logger.info("Successfully cleared image_queries table")
            except Exception as e:
                logger.error(f"Error clearing image_queries table: {e}")
                raise
    except Exception as e:
        logger.error(f"Failed to clear image_queries table: {e}")
        raise

async def init_default_users(logger: logging.Logger):
    try:
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
                        password_hash=hash_password("1")
                    )
                    session.add(admin_user)
                    await session.commit()
                    logger.info("Admin user created successfully")
                else:
                    logger.info("Admin user already exists")

            except Exception as e:
                logger.error(f"Error in init_default_users: {e}")
                raise

        async with async_session() as session:
            try:
                result = await session.execute(
                    select(User).where(User.username == "demo")
                )
                user = result.scalar_one_or_none()

                if user is None:
                    logger.info("Creating demo user...")
                    demo_user = User(
                        username="demo",
                        password_hash=hash_password("1")
                    )
                    session.add(demo_user)
                    await session.commit()
                    logger.info("Demo user created successfully")
                else:
                    logger.info("Demo user already exists")

            except Exception as e:
                logger.error(f"Error in init_default_users: {e}")
                raise

    except (ConnectionRefusedError, ConnectionDoesNotExistError,
            CannotConnectNowError, PostgresConnectionError) as e:
        logger.error("Failed to connect to the database. Please check your database configuration.")
        logger.error(f"Connection error details: {str(e)}")
        sys.exit(1)

    except SQLAlchemyError as e:
        logger.error("An error occurred while interacting with the database.")
        logger.error(f"Database error details: {str(e)}")
        sys.exit(1)

    except Exception as e:
        logger.error("An unexpected error occurred during database initialization.")
        logger.error(f"Error details: {str(e)}")
        sys.exit(1)

