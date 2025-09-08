import json

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

    def register(self, config_path: str, app_name: str, config_type: AbstractConfig):
        with open(config_path, "r") as f:
            try:
                json_str = f.read()
                config_data = json.loads(json_str)
                config = config_type(**config_data)
                self.registrations[app_name] = config.build()
            except Exception as e:
                print(f"Error registering {app_name}, the config file is not valid\n {e}")

    def get_object(self, app_name: str) -> AbstractModule:
        return self.registrations[app_name]