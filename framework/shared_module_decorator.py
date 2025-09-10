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
        
        key = (class_, config_key, frozenset(kwargs.items()))
        if key not in instances:
            instances[key] = class_(*args, **kwargs)
        return instances[key]
    
    # Keep the original class's attributes and methods
    getinstance.__name__ = class_.__name__
    getinstance.__doc__ = class_.__doc__
    getinstance.__module__ = class_.__module__
    return getinstance

def make_hashable(obj):
    """Recursively convert objects to hashable types"""
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return tuple(sorted(make_hashable(item) for item in obj))
    else:
        return obj