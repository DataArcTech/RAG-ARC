import json
import unittest
from typing import Annotated, TypeVar, Literal
from pydantic import Field, ValidationError
import sys
import os

# Add the project root to Python path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.config import AbstractConfig
from framework.module import AbstractModule
from framework.shared_module_decorator import shared_module

# Type definitions
NewType = TypeVar("NewType")
OperationType = Literal["read", "write", "append"]


class AConfig(AbstractConfig):
    """Config class for module A"""

    type: Literal["A"] = "A"
    param1: int
    param2: NewType
    param3: str = "test_string"
    param4: OperationType = "read"

    def build(self) -> "A":
        return A(self)


class A(AbstractModule):
    """Module A implementation"""

    config: AConfig | None = None

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
    param2: str = "default_string"  # Default value
    param3: NewType = "default_newtype"  # Default value
    sub_config: AConfig | None = None  # Optional with default None

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
        if self.config.sub_config is not None:
            a_module = self.config.sub_config.build()
            result = a_module("test_param")
            print(f"Module A result: {result}")
            return f"C processed: {params}, A result: {result}"
        else:
            return f"C processed: {params} (no sub_config)"

class BConfig(AbstractConfig):
    """Config class for module B"""

    type: Literal["B"] = "B"
    param1: int
    param2: str = "default_string"
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
    param2: str = "default_string_d"
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

    def test_create_c_config_with_defaults(self):
        """Test creating CConfig with minimal data, using defaults"""
        json_str = """
        {
            "type": "C",
            "param1": 42
        }
        """
        config_data = json.loads(json_str)
        config = CConfig(**config_data)
        c = config.build()
        self.assertIsInstance(c, C)
        self.assertEqual(c.config.type, "C")
        self.assertEqual(c.config.param1, 42)  # Default value
        self.assertEqual(c.config.param2, "default_string")  # Default value
        self.assertEqual(c.config.param3, "default_newtype")  # Default value
        self.assertIsNone(c.config.sub_config)  # Default value
        
        # Test calling the module with defaults
        result = c("test_input")
        self.assertIn("C processed: test_input (no sub_config)", result)

    def test_create_c_config_with_partial_overrides(self):
        """Test creating CConfig with some overrides and some defaults"""
        json_str = """
        {
            "type": "C",
            "param1": 100,
            "param2": "custom_string"
        }
        """
        config_data = json.loads(json_str)
        config = CConfig(**config_data)
        c = config.build()
        self.assertIsInstance(c, C)
        self.assertEqual(c.config.type, "C")
        self.assertEqual(c.config.param1, 100)  # Overridden
        self.assertEqual(c.config.param2, "custom_string")  # Overridden
        self.assertEqual(c.config.param3, "default_newtype")  # Default value
        self.assertIsNone(c.config.sub_config)  # Default value

# -------------------------------- Shared module decorator test --------------------------------

class KConfig(AbstractConfig):
    """Config class for module K"""
    type: Literal["K"] = "K"
    param1: int
    param2: str
    param3: NewType

    def build(self) -> "K":
        return K(self)

@shared_module
class K(AbstractModule):
    """Module K implementation"""

    config: KConfig

    def __call__(self, params):
        print(f"Module K called with params: {params}")
        print(f"Config param1: {self.config.param1}")
        print(f"Config param2: {self.config.param2}")
        print(f"Config param3: {self.config.param3}")
        return f"K processed: {params}"

class FConfig(AbstractConfig):
    """Config class for shared module F"""
    type: Literal["F"] = "F"
    param1: int
    param2: str
    param3: NewType
    sub_config: KConfig

    def build(self) -> "F":
        return F(self)


@shared_module
class F(AbstractModule):
    """Shared module F implementation"""
    
    config: FConfig

    def __call__(self, param):
        print(f"Module F called with param: {param}")
        print(f"Config param1: {self.config.param1}")
        print(f"Config param2: {self.config.param2}")
        print(f"Config param3: {self.config.param3}")
        if self.config.sub_config is not None:
            k_module = self.config.sub_config.build()
            result = k_module("test_param_from_F")
            print(f"Module K result from F: {result}")
            return f"F processed: {param}, K result: {result}"
        else:
            return f"F processed: {param}"

class GConfig(AbstractConfig):
    """Config class for module G"""
    type: Literal["G"] = "G"
    param1: int
    param2: str
    param3: NewType
    sub_config: FConfig

    def build(self) -> "G":
        return G(self)


class G(AbstractModule):
    """Module G implementation that uses shared module F"""
    
    config: GConfig

    def __call__(self, params):
        print(f"Module G called with params: {params}")
        print(f"Config param1: {self.config.param1}")
        print(f"Config param2: {self.config.param2}")
        print(f"Config param3: {self.config.param3}")
        f_module = self.config.sub_config.build()
        result = f_module("test_param_from_G")
        print(f"Module F result from G: {result}")
        return f"G processed: {params}, F result: {result}"


class HConfig(AbstractConfig):
    """Config class for module H"""
    type: Literal["H"] = "H"
    param1: int
    param2: str
    param3: NewType
    sub_config: FConfig

    def build(self) -> "H":
        return H(self)


class H(AbstractModule):
    """Module H implementation that also uses shared module F"""
    
    config: HConfig

    def __call__(self, params):
        print(f"Module H called with params: {params}")
        print(f"Config param1: {self.config.param1}")
        print(f"Config param2: {self.config.param2}")
        print(f"Config param3: {self.config.param3}")
        f_module = self.config.sub_config.build()
        result = f_module("test_param_from_H")
        print(f"Module F result from H: {result}")
        return f"H processed: {params}, F result: {result}"

class IConfig(AbstractConfig):
    """Config class for module I"""
    type: Literal["I"] = "I"
    param1: int
    param2: str
    param3: NewType
    sub_config: list[Annotated[GConfig | HConfig, Field(discriminator="type")]]

    def build(self) -> "I":
        return I(self)

class I(AbstractModule):
    """Module I implementation"""

    config: IConfig
    
    def __call__(self, params):
        print(f"Module I called with params: {params}")
        print(f"Config param1: {self.config.param1}")
        print(f"Config param2: {self.config.param2}")
        print(f"Config param3: {self.config.param3}")
        # Call all submodules and collect their results
        results = []
        for sub in self.config.sub_config:
            mod = sub.build()
            if hasattr(mod, '__call__'):
                results.append(mod(f"from_I_{params}"))
        return f"I processed: {params}, submodules: {results}"

class TestSharedModule(unittest.TestCase):
    """Test shared module functionality"""

    def test_shared_module_same_config(self):
        json_str = """
        {
            "type": "I",
            "param1": 1,
            "param2": "iparam",
            "param3": "ival",
            "sub_config": [
                {
                    "type": "G",
                    "param1": 2,
                    "param2": "gparam",
                    "param3": "gval",
                    "sub_config": {
                        "type": "F",
                        "param1": 10,
                        "param2": "fparam",
                        "param3": "fval",
                        "sub_config": {
                            "type": "K",
                            "param2": "kparam",
                            "param1": 10,
                            "param3": "kval"
                        }
                    }
                },
                {
                    "type": "H",
                    "param1": 3,
                    "param2": "hparam",
                    "param3": "hval",
                    "sub_config": {
                        "type": "F",
                        "param1": 10,
                        "param2": "fparam",
                        "param3": "fval",
                        "sub_config": {
                            "type": "K",
                            "param1": 10,
                            "param2": "kparam",
                            "param3": "kval"
                        }
                    }
                }
            ]
        }
        """
        config_data = json.loads(json_str)
        config = IConfig(**config_data)
        i_module = config.build()
        self.assertIsInstance(i_module, I)
        self.assertEqual(i_module.config.type, "I")
        self.assertEqual(len(i_module.config.sub_config), 2)
        g_config = i_module.config.sub_config[0]
        h_config = i_module.config.sub_config[1]
        g_module = g_config.build()
        h_module = h_config.build()
        f_from_g = g_module.config.sub_config.build()
        f_from_h = h_module.config.sub_config.build()
        self.assertIs(f_from_g, f_from_h)
        # Both F modules should have the same K sub_config
        if f_from_g.config.sub_config is not None and f_from_h.config.sub_config is not None:
            k_from_g = f_from_g.config.sub_config.build()
            k_from_h = f_from_h.config.sub_config.build()
            self.assertIs(k_from_g, k_from_h)

    def test_shared_module_different_k_config(self):
        json_str = """
        {
            "type": "I",
            "param1": 1,
            "param2": "iparam",
            "param3": "ival",
            "sub_config": [
                {
                    "type": "G",
                    "param1": 2,
                    "param2": "gparam",
                    "param3": "gval",
                    "sub_config": {
                        "type": "F",
                        "param1": 10,
                        "param2": "fparam",
                        "param3": "fval",
                        "sub_config": {
                            "type": "K",
                            "param1": 10,
                            "param2": "kparam",
                            "param3": "kval"
                        }
                    }
                },
                {
                    "type": "H",
                    "param1": 3,
                    "param2": "hparam",
                    "param3": "hval",
                    "sub_config": {
                        "type": "F",
                        "param1": 10,
                        "param2": "fparam",
                        "param3": "fval",
                        "sub_config": {
                            "type": "K",
                            "param1": 20,
                            "param2": "kparam",
                            "param3": "kval"
                        }
                    }
                }
            ]
        }
        """
        config_data = json.loads(json_str)
        config = IConfig(**config_data)
        i_module = config.build()
        self.assertIsInstance(i_module, I)
        self.assertEqual(i_module.config.type, "I")
        self.assertEqual(len(i_module.config.sub_config), 2)
        g_config = i_module.config.sub_config[0]
        h_config = i_module.config.sub_config[1]
        g_module = g_config.build()
        h_module = h_config.build()
        f_from_g = g_module.config.sub_config.build()
        f_from_h = h_module.config.sub_config.build()
        self.assertIsNot(f_from_g, f_from_h)
        # F modules have different K sub_configs, so K modules should be different
        if f_from_g.config.sub_config is not None and f_from_h.config.sub_config is not None:
            k_from_g = f_from_g.config.sub_config.build()
            k_from_h = f_from_h.config.sub_config.build()
            self.assertIsNot(k_from_g, k_from_h)

    def test_shared_module_different_config(self):
        json_str = """
        {
            "type": "I",
            "param1": 1,
            "param2": "iparam",
            "param3": "ival",
            "sub_config": [
                {
                    "type": "G",
                    "param1": 2,
                    "param2": "gparam",
                    "param3": "gval",
                    "sub_config": {
                        "type": "F",
                        "param1": 10,
                        "param2": "fparam",
                        "param3": "fval",
                        "sub_config": {
                            "type": "K",
                            "param1": 10,
                            "param2": "kparam",
                            "param3": "kval"
                        }
                    }
                },
                {
                    "type": "H",
                    "param1": 3,
                    "param2": "hparam",
                    "param3": "hval",
                    "sub_config": {
                        "type": "F",
                        "param1": 20,
                        "param2": "fparam",
                        "param3": "fval",
                        "sub_config": {
                            "type": "K",
                            "param1": 20,
                            "param2": "kparam",
                            "param3": "kval"
                        }
                    }
                }
            ]
        }
        """
        config_data = json.loads(json_str)
        config = IConfig(**config_data)
        i_module = config.build()
        self.assertIsInstance(i_module, I)
        self.assertEqual(i_module.config.type, "I")
        self.assertEqual(len(i_module.config.sub_config), 2)
        g_config = i_module.config.sub_config[0]
        h_config = i_module.config.sub_config[1]
        g_module = g_config.build()
        h_module = h_config.build()
        f_from_g = g_module.config.sub_config.build()
        f_from_h = h_module.config.sub_config.build()
        self.assertIsNot(f_from_g, f_from_h)

if __name__ == "__main__":
    unittest.main()