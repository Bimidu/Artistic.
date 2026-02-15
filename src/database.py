"""
MongoDB Database Configuration

This module handles the MongoDB connection and provides
a database session for FastAPI dependency injection.
"""

from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "asd_detection")

# Global MongoDB client
_mongodb_client: Optional[AsyncIOMotorClient] = None


async def connect_to_mongo():
    """Connect to MongoDB on startup"""
    global _mongodb_client
    try:
        _mongodb_client = AsyncIOMotorClient(MONGODB_URL)
        # Test connection
        await _mongodb_client.admin.command('ping')
        logger.info(f"✓ Connected to MongoDB at {MONGODB_URL}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongo_connection():
    """Close MongoDB connection on shutdown"""
    global _mongodb_client
    if _mongodb_client:
        _mongodb_client.close()
        logger.info("MongoDB connection closed")


def get_database():
    """Get database instance"""
    if _mongodb_client is None:
        raise RuntimeError("MongoDB client not initialized. Call connect_to_mongo() first.")
    return _mongodb_client[DATABASE_NAME]
