import json
import unittest
from typing import Annotated, TypeVar, Literal
from pydantic import Field, ValidationError
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.config import AbstractConfig
from framework.module import AbstractModule

# Type definitions
NewType = TypeVar("NewType")
OperationType = Literal["read", "write", "append"]


class AConfig(AbstractConfig):
    """Config class for module A"""

    type: Literal["A"] = "A"
    param1: int | None = None
    param2: NewType
    param3: str
    param4: OperationType

    def build(self) -> "A":
        return A(self)


class A(AbstractModule):
    """Module A implementation"""

    config: AConfig

    def __call__(self, param):
        print(f"Module A called with param: {param}")
        print(f"Config param1: {self.config.param1}")
        print(f"Config param2: {self.config.param2}")
        print(f"Config param3: {self.config.param3}")
        print(f"Config param4: {self.config.param4}")
        return f"A processed: {param}"


class CConfig(AbstractConfig):
    """Config class for module C"""

    type: Literal["C"] = "C"
    param1: int
    param2: str
    param3: NewType
    sub_config: AConfig

    def build(self) -> "C":
        return C(self)


class C(AbstractModule):
    """Module C implementation"""

    config: CConfig
    
    def __call__(self, params):
        print(f"Module C called with params: {params}")
        print(f"Config param1: {self.config.param1}")
        print(f"Config param2: {self.config.param2}")
        print(f"Config param3: {self.config.param3}")
        a_module = self.config.sub_config.build()
        result = a_module("test_param")
        print(f"Module A result: {result}")
        return f"C processed: {params}, A result: {result}"

class BConfig(AbstractConfig):
    """Config class for module B"""

    type: Literal["B"] = "B"
    param1: int
    param2: str
    param3: NewType
    sub_config: Annotated[AConfig | CConfig, Field(discriminator="type")]

    def build(self) -> "B":
        return B(self)


class B(AbstractModule):
    """Module B implementation"""

    config: BConfig

    def __call__(self, params):
        print(f"Module B called with params: {params}")
        print(f"Config param1: {self.config.param1}")
        print(f"Config param2: {self.config.param2}")
        print(f"Config param3: {self.config.param3}")
        a_module = self.config.sub_config.build()
        result = a_module("test_param")
        print(f"Module A result: {result}")
        return f"B processed: {params}, A result: {result}"


class TestConfigFromJson(unittest.TestCase):
    """Test creating configs from JSON"""

    def test_create_b_and_a_config_from_json_(self):
        json_str = """
        {
            "type": "B",
            "param1": 456,
            "param2": "baz",
            "param3": "qux",
            "sub_config": {
                "type": "A",
                "param1": 789,
                "param2": "nested",
                "param3": "seq",
                "param4": "write"
            }
        }
        """
        config_data = json.loads(json_str)
        config = BConfig(**config_data)
        b = config.build()
        self.assertIsInstance(b, B)
        self.assertEqual(b.config.type, "B")
        self.assertEqual(b.config.param1, 456)
        self.assertEqual(b.config.param2, "baz")
        self.assertEqual(b.config.param3, "qux")
        self.assertIsInstance(b.config.sub_config, AConfig)
        self.assertEqual(b.config.sub_config.type, "A")
        sub_config = b.config.sub_config.build()
        self.assertIsInstance(sub_config, A)
        self.assertEqual(sub_config.config.type, "A")
        self.assertEqual(sub_config.config.param1, 789)
        self.assertEqual(sub_config.config.param2, "nested")
        self.assertEqual(sub_config.config.param3, "seq")
        self.assertEqual(sub_config.config.param4, "write")

    def test_create_b_and_c_config_from_json(self):
        json_str = """
        {
            "type": "B",
            "param1": 456,
            "param2": "baz",
            "param3": "qux",
            "sub_config": {
                "type": "C",
                "param1": 789,
                "param2": "nested",
                "param3": "seq",
                "sub_config": {
                    "type": "A",
                    "param1": 123,
                    "param2": "foo",
                    "param3": "bar",
                    "param4": "read"
                }
            }
        }
        """
        config_data = json.loads(json_str)
        config = BConfig(**config_data)
        b = config.build()
        self.assertIsInstance(b, B)
        self.assertEqual(b.config.type, "B")
        self.assertEqual(b.config.param1, 456)
        self.assertEqual(b.config.param2, "baz")
        self.assertEqual(b.config.param3, "qux")
        self.assertIsInstance(b.config.sub_config, CConfig)
        self.assertEqual(b.config.sub_config.type, "C")
        sub_config = b.config.sub_config.build()
        self.assertIsInstance(sub_config, C)
        self.assertEqual(sub_config.config.type, "C")
        self.assertEqual(sub_config.config.param1, 789)
        self.assertEqual(sub_config.config.param2, "nested")
        self.assertEqual(sub_config.config.param3, "seq")
        sub_sub_config = sub_config.config.sub_config.build()
        self.assertIsInstance(sub_sub_config, A)
        self.assertEqual(sub_sub_config.config.type, "A")
        self.assertEqual(sub_sub_config.config.param1, 123)
        self.assertEqual(sub_sub_config.config.param2, "foo")
        self.assertEqual(sub_sub_config.config.param3, "bar")
        self.assertEqual(sub_sub_config.config.param4, "read")

    def test_invalid_aconfig_from_json(self):
        json_str = """
        {
            "type": "A",
            "param2": "foo",
            "param3": "bar",
            "param4": "not_a_valid_operation"
        }
        """
        config_data = json.loads(json_str)
        with self.assertRaises(ValidationError):
            AConfig(**config_data)

    def test_invalid_bconfig_nested_json(self):
        json_str = """
        {
            "type": "B",
            "param1": 1,
            "param2": "b",
            "param3": "c",
            "sub_config": {
                "type": "A",
                "param2": "foo",
                "param3": "bar",
                "param4": "not_a_valid_operation"
            }
        }
        """
        config_data = json.loads(json_str)
        with self.assertRaises(ValidationError):
            BConfig(**config_data)


if __name__ == "__main__":
    unittest.main()
