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
            # If it is a Pydantic model, convert to a dictionary and sort
            config_dict = args[0].model_dump()
            config_key = tuple(sorted(list(config_dict.items())))
        elif args and hasattr(args[0], '__dict__'):
            # If it is a normal object, convert to a dictionary and sort
            config_dict = args[0].__dict__
            config_key = tuple(sorted(list(config_dict.items())))
        
        key = (class_, config_key, frozenset(kwargs.items()))
        if key not in instances:
            instances[key] = class_(*args, **kwargs)
        return instances[key]
    
    # Keep the original class's attributes and methods
    getinstance.__name__ = class_.__name__
    getinstance.__doc__ = class_.__doc__
    getinstance.__module__ = class_.__module__
    return getinstance