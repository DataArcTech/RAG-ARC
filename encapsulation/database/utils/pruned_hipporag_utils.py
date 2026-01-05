import hashlib
from typing import Optional, Union

from encapsulation.utils.text_processing import normalize_entity_text, text_processing

OwnerIdType = Optional[Union[str, "uuid.UUID"]]

# NOTE: `text_processing` and `normalize_entity_text` are imported from
# `encapsulation.utils.text_processing` to keep a single source of truth.


def _owner_scoped_value(value: str, owner_id: OwnerIdType = None) -> str:
    """Prefix value with owner scope when provided to keep IDs tenant-aware."""
    if owner_id is None:
        return value
    return f"{owner_id}:{value}"


def compute_mdhash_id(content: str, prefix: str = "", owner_id: OwnerIdType = None) -> str:
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
    scoped_value = _owner_scoped_value(content, owner_id)
    return prefix + hashlib.md5(scoped_value.encode()).hexdigest()


def compute_entity_id(entity_name: str, owner_id: OwnerIdType = None) -> str:
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
    scoped_value = _owner_scoped_value(entity_name, owner_id)
    return "entity-" + hashlib.md5(scoped_value.encode()).hexdigest()
