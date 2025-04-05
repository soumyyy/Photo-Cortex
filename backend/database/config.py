from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Convert the URL to async format if needed
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    SYNC_DATABASE_URL = DATABASE_URL
else:
    raise ValueError("DATABASE_URL must be set and start with postgresql://")

# Async engine for FastAPI
async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=True)
async_session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for Alembic
sync_engine = create_engine(SYNC_DATABASE_URL)

async def get_async_session() -> AsyncSession:
    async with async_session() as session:
        yield session