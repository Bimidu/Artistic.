"""
CHAT File Parsers and Data Extraction

This package contains modules for parsing CHAT-formatted transcript files
and extracting structured data for analysis.
"""

from .chat_parser import CHATParser, TranscriptData
from .dataset_inventory import DatasetInventory, ParticipantInfo

__all__ = ["CHATParser", "TranscriptData", "DatasetInventory", "ParticipantInfo"]

