import re
import hashlib


# Regex pattern for text normalization (remove special characters, keep alphanumeric, Chinese characters, and spaces)
# \u4e00-\u9fff: Chinese characters (CJK Unified Ideographs)
TEXT_NORMALIZATION_PATTERN = re.compile('[^A-Za-z0-9\u4e00-\u9fff ]')


def normalize_entity_text(text: str) -> str:
    """
    Normalize entity text for consistent matching.
    
    Converts to lowercase and removes special characters, keeping only alphanumeric and spaces.
    This ensures "Apple Inc.", "apple inc", and "APPLE INC" are treated as the same entity.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text (lowercase, alphanumeric + spaces only)
    
    Examples:
        >>> normalize_entity_text("Apple Inc.")
        'apple inc'
        >>> normalize_entity_text("APPLE INC")
        'apple inc'
        >>> normalize_entity_text("New York City!")
        'new york city'
    """
    return TEXT_NORMALIZATION_PATTERN.sub(' ', text.lower()).strip()


def text_processing(text: str) -> str:
    """
    Normalize text by removing special characters and converting to lowercase.
    
    This is an alias for normalize_entity_text() for backward compatibility.
    This ensures consistent entity matching across the system.
    For example: "Apple Inc.", "apple inc", "APPLE INC" → "apple inc"
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text (lowercase, alphanumeric + spaces only)
    
    Examples:
        >>> text_processing("Apple Inc.")
        'apple inc'
        >>> text_processing("APPLE INC")
        'apple inc'
    """
    if not isinstance(text, str):
        text = str(text)
    return TEXT_NORMALIZATION_PATTERN.sub(' ', text.lower()).strip()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute MD5 hash ID for content with optional prefix.
    
    Args:
        content: String content to hash
        prefix: Optional prefix for the hash ID
        
    Returns:
        Hash ID string (prefix + MD5 hex digest)
    
    Examples:
        >>> compute_mdhash_id("apple inc", prefix="entity-")
        'entity-...'  # MD5 hash
        >>> compute_mdhash_id("some fact text", prefix="fact-")
        'fact-...'  # MD5 hash
    """
    return prefix + hashlib.md5(content.encode()).hexdigest()


def compute_entity_id(entity_name: str) -> str:
    """
    Compute MD5 hash ID for an entity name.
    
    This is a convenience function that automatically adds the 'entity-' prefix.
    The entity_name should already be normalized using normalize_entity_text().
    
    Args:
        entity_name: Normalized entity name
        
    Returns:
        Entity ID with 'entity-' prefix
    
    Examples:
        >>> compute_entity_id("apple inc")
        'entity-...'  # MD5 hash with entity- prefix
    """
    return "entity-" + hashlib.md5(entity_name.encode()).hexdigest()

