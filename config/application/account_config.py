from framework.config import AbstractConfig
from typing import Literal
from config.core.file_management.storage.user_storage import UserStorageConfig
from application.account.user import Account


class AccountConfig(AbstractConfig):
    type: Literal["account"] = "account"
    user_storage_config: UserStorageConfig

    def build(self) -> Account:
        return Account(config=self)