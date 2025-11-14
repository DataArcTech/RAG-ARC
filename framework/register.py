import json
import os
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from framework.config import AbstractConfig
from framework.module import AbstractModule
from framework.singleton_decorator import singleton

@singleton
class Register:
    """
    This is used to register all kinds of applications, which can be used by FastAPI or other frameworks.
    """
    def __init__(self):
        self.registrations = {}

    def _substitute_env_vars(self, obj):
        """Recursively substitute environment variables in config data"""
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Replace ${VAR_NAME} with environment variable values
            def replace_env_var(match):
                var_name = match.group(1)
                env_value = os.getenv(var_name)
                if env_value is None:
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