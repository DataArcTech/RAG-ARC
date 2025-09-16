def shared_module(class_):
    """
    This decorator is used to mark a class as a shared module. 
    1. When parameters set forth in Config class are the same, the SAME instance will be returned.
    2. When parameters are different, a NEW instance will be returned.
    For examples, see module_test.py
    """
    instances = {}
    
    def getinstance(*args, **kwargs):
        config_key = None
        if args and hasattr(args[0], 'model_dump'): 
            config_dict = args[0].model_dump()
            config_key = make_hashable(config_dict)
        elif args and hasattr(args[0], '__dict__'):
            config_dict = args[0].__dict__
            config_key = make_hashable(config_dict)
        
        # Ensure config_key is hashable
        if config_key is None:
            config_key = ()
        
        # Make kwargs hashable as well to avoid "unhashable type: 'dict'"
        kwargs_key = make_hashable(kwargs)
        
        key = (class_, config_key, kwargs_key)
        if key not in instances:
            instances[key] = class_(*args, **kwargs)
        return instances[key]
    
    # Keep the original class's attributes and methods
    getinstance.__name__ = class_.__name__
    getinstance.__doc__ = class_.__doc__
    getinstance.__module__ = class_.__module__
    return getinstance

# def make_hashable(obj):
#     """Recursively convert objects to hashable types"""
#     if isinstance(obj, dict):
#         return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
#     elif isinstance(obj, list):
#         return tuple(make_hashable(item) for item in obj)
#     elif isinstance(obj, set):
#         return tuple(sorted(make_hashable(item) for item in obj))
#     else:
#         return obj


def make_hashable(obj):
    """Recursively convert complex objects (including Pydantic BaseModel) to hashable types."""
    # Handle Pydantic BaseModel explicitly
    try:
        from pydantic import BaseModel  # pydantic v2
        if isinstance(obj, BaseModel):
            return make_hashable(obj.model_dump())
    except Exception:
        # If pydantic is not available or any issue occurs, fall through to other handlers
        pass

    # Dict: normalize keys (stringify for stable sorting) and recurse on values
    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            try:
                key_hashable = k if isinstance(k, (str, int, float, bool, bytes)) else repr(k)
            except Exception:
                key_hashable = repr(k)
            items.append((key_hashable, make_hashable(v)))
        items.sort(key=lambda kv: kv[0])
        return tuple(items)

    # List/Tuple: recurse and convert to tuple
    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(x) for x in obj)

    # Set: recurse and sort by repr for deterministic order
    if isinstance(obj, set):
        return tuple(sorted((make_hashable(x) for x in obj), key=lambda x: repr(x)))

    # Dataclass support
    try:
        from dataclasses import is_dataclass, asdict
        if is_dataclass(obj):
            return make_hashable(asdict(obj))
    except Exception:
        pass

    # Numpy ndarray or scalar support (best-effort without hard dependency)
    try:
        import numpy as np  # project already uses numpy; ignore if not available
        if isinstance(obj, np.ndarray):
            return ("np.ndarray", tuple(make_hashable(x) for x in obj.tolist()))
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
    except Exception:
        pass

    # Byte-like types
    if isinstance(obj, bytearray):
        return ("bytearray", bytes(obj))
    if isinstance(obj, memoryview):
        return ("memoryview", bytes(obj))

    # Objects exposing model_dump but not necessarily BaseModel (custom models)
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return make_hashable(obj.model_dump())
        except Exception:
            pass

    # Generic Python objects: use public attributes
    if hasattr(obj, "__dict__"):
        try:
            return make_hashable({k: v for k, v in vars(obj).items() if not k.startswith("_")})
        except Exception:
            pass

    # If already hashable, return as-is; otherwise, fallback to repr
    try:
        hash(obj)
        return obj
    except TypeError:
        return repr(obj)