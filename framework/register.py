import json
import os
import re
import logging
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from framework.config import AbstractConfig
from framework.module import AbstractModule
from framework.singleton_decorator import singleton

_UNRESOLVED_ENV_PLACEHOLDER = object()

try:  # Centralized placeholder policy lives in config/
    from config.env_placeholder_policy import ENV_DEFAULTS as _ENV_DEFAULTS
    from config.env_placeholder_policy import SILENT_MISSING_ENV_VARS as _SILENT_MISSING_ENV_VARS
except Exception:  # pragma: no cover - defensive for minimal runtimes
    _ENV_DEFAULTS = {}
    _SILENT_MISSING_ENV_VARS = set()

@singleton
class Register:
    """
    This is used to register all kinds of applications, which can be used by FastAPI or other frameworks.
    """
    def __init__(self):
        self.registrations = {}

    def _substitute_env_vars(self, obj: Any):
        """Recursively substitute environment variables in config data"""
        if isinstance(obj, dict):
            resolved: dict[str, Any] = {}
            for key, value in obj.items():
                substituted = self._substitute_env_vars(value)
                if substituted is _UNRESOLVED_ENV_PLACEHOLDER:
                    continue
                resolved[key] = substituted
            return resolved
        elif isinstance(obj, list):
            items = [self._substitute_env_vars(item) for item in obj]
            return [None if item is _UNRESOLVED_ENV_PLACEHOLDER else item for item in items]
        elif isinstance(obj, str):
            # If the whole value is a single placeholder, allow unresolved values to be omitted so
            # downstream config defaults can take over (see test/framework/test_register_env_substitution.py).
            whole = re.fullmatch(r'\$\{([^}]+)\}', obj.strip())
            if whole:
                var_name = whole.group(1)
                env_value = os.getenv(var_name)
                if not env_value:
                    default_value = _ENV_DEFAULTS.get(var_name)
                    if default_value is not None:
                        return default_value
                    if var_name in _SILENT_MISSING_ENV_VARS:
                        return _UNRESOLVED_ENV_PLACEHOLDER
                    logger.warning(
                        "Environment variable '%s' is not set, omitting placeholder value %r",
                        var_name,
                        obj,
                    )
                    return _UNRESOLVED_ENV_PLACEHOLDER
                return env_value

            # Replace ${VAR_NAME} with environment variable values (leave embedded placeholders intact when missing).
            def replace_env_var(match):
                var_name = match.group(1)
                env_value = os.getenv(var_name)
                if not env_value:
                    default_value = _ENV_DEFAULTS.get(var_name)
                    if default_value is not None:
                        return default_value
                    if var_name in _SILENT_MISSING_ENV_VARS:
                        return match.group(0)
                    # Environment variable not set - log warning and return original
                    logger.warning(f"Environment variable '{var_name}' is not set, using placeholder '{match.group(0)}'")
                    return match.group(0)  # Return original placeholder
                return env_value
            return re.sub(r'\$\{([^}]+)\}', replace_env_var, obj)
        else:
            return obj

    def register(self, config_path: str, app_name: str, config_type: AbstractConfig):
        logger.info(f"Registering {app_name} with config path {config_path}")
        with open(config_path, "r") as f:
            try:
                json_str = f.read()
                config_data = json.loads(json_str)
                # Substitute environment variables
                config_data = self._substitute_env_vars(config_data)
                config = config_type(**config_data)
                self.registrations[app_name] = config.build()
                logger.info(f"Successfully registered {app_name}")
            except Exception as e:
                logger.error(f"Error registering {app_name}, the config file is not valid\n {e}")
                raise e

    def get_object(self, app_name: str) -> AbstractModule:
        return self.registrations[app_name]
