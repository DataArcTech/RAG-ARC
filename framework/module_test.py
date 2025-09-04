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

class DConfig(AbstractConfig):
    """Config class for module D"""
    type: Literal["D"] = "D"
    param1: int
    param2: str
    param3: NewType
    sub_config: list[Annotated[AConfig | CConfig, Field(discriminator="type")]]
    def build(self) -> "D":
        return D(self)

class D(AbstractModule):
    """Module D implementation"""

    config: DConfig
    
    def __call__(self, params):
        print(f"Module D called with params: {params}")
        print(f"Config param1: {self.config.param1}")
        print(f"Config param2: {self.config.param2}")
        print(f"Config param3: {self.config.param3}")
        return f"D processed: {params}"

class TestConfigFromJson(unittest.TestCase):
    """Test creating configs from JSON"""

    def test_create_b_and_a_config_from_json(self):
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

    def test_create_d_config_from_json(self):
        json_str = """
        {
            "type": "D",
            "param1": 42,
            "param2": "dparam",
            "param3": "dval",
            "sub_config": [
                {
                    "type": "A",
                    "param1": 100,
                    "param2": "a1",
                    "param3": "a2",
                    "param4": "append"
                },
                {
                    "type": "C",
                    "param1": 200,
                    "param2": "c1",
                    "param3": "c2",
                    "sub_config": {
                        "type": "A",
                        "param1": 300,
                        "param2": "a3",
                        "param3": "a4",
                        "param4": "read"
                    }
                }
            ]
        }
        """
        config_data = json.loads(json_str)
        config = DConfig(**config_data)
        d = config.build()
        self.assertIsInstance(d, D)
        self.assertEqual(d.config.type, "D")
        self.assertEqual(d.config.param1, 42)
        self.assertEqual(d.config.param2, "dparam")
        self.assertEqual(d.config.param3, "dval")
        self.assertEqual(len(d.config.sub_config), 2)
        # First sub_config is AConfig
        sub0 = d.config.sub_config[0]
        self.assertIsInstance(sub0, AConfig)
        self.assertEqual(sub0.type, "A")
        self.assertEqual(sub0.param1, 100)
        self.assertEqual(sub0.param2, "a1")
        self.assertEqual(sub0.param3, "a2")
        self.assertEqual(sub0.param4, "append")
        # Second sub_config is CConfig
        sub1 = d.config.sub_config[1]
        self.assertIsInstance(sub1, CConfig)
        self.assertEqual(sub1.type, "C")
        self.assertEqual(sub1.param1, 200)
        self.assertEqual(sub1.param2, "c1")
        self.assertEqual(sub1.param3, "c2")
        # sub_config of CConfig is AConfig
        sub1_sub = sub1.sub_config
        self.assertIsInstance(sub1_sub, AConfig)
        self.assertEqual(sub1_sub.type, "A")
        self.assertEqual(sub1_sub.param1, 300)
        self.assertEqual(sub1_sub.param2, "a3")
        self.assertEqual(sub1_sub.param3, "a4")
        self.assertEqual(sub1_sub.param4, "read")
        # Test build chain
        d_module = config.build()
        self.assertIsInstance(d_module, D)
        self.assertEqual(d_module.config.type, "D")

if __name__ == "__main__":
    unittest.main()
