import json
import unittest
from pydantic import ValidationError
from framework.config import AbstractConfig
from typing import Literal, Annotated
from pydantic import Field


class AConfig(AbstractConfig):
    type: Literal["A"] = "A"
    a_1: int
    a_2: str

    def build(self):
        # This is just for testing - in real usage you'd return the actual module
        return None


class BConfig(AbstractConfig):
    type: Literal["B"] = "B"
    b_1: str
    b_2: int
    b_3: float

    def build(self):
        # This is just for testing - in real usage you'd return the actual module
        return None


class CConfig(AbstractConfig):
    type: Literal["C"] = "C"
    model: Annotated[AConfig | BConfig, Field(discriminator="type")]
    c_1: int

    def build(self):
        # This is just for testing - in real usage you'd return the actual module
        return None


class TestConfigValidation(unittest.TestCase):
    """Unit tests for AbstractConfig and its subclasses."""

    def test_cconfig_with_aconfig_model(self):
        """Test CConfig with AConfig as model."""
        json_str = """
        {
            "type": "C",
            "model": {
                "type": "A",
                "a_1": 123,
                "a_2": "foo"
            },
            "c_1": 100
        }
        """
        data = json.loads(json_str)
        c_a = CConfig.model_validate(data)
        self.assertIsInstance(c_a, CConfig)
        self.assertEqual(c_a.type, "C")
        self.assertIsInstance(c_a.model, AConfig)
        self.assertEqual(c_a.model.a_1, 123)
        self.assertEqual(c_a.model.a_2, "foo")
        self.assertEqual(c_a.c_1, 100)

    def test_cconfig_with_bconfig_model(self):
        """Test CConfig with BConfig as model."""
        json_str = """
        {
            "type": "C",
            "model": {
                "type": "B",
                "b_1": "bar",
                "b_2": 456,
                "b_3": 3.14
            },
            "c_1": 200
        }
        """
        data = json.loads(json_str)
        c_b = CConfig.model_validate(data)
        self.assertIsInstance(c_b, CConfig)
        self.assertEqual(c_b.type, "C")
        self.assertIsInstance(c_b.model, BConfig)
        self.assertEqual(c_b.model.b_1, "bar")
        self.assertEqual(c_b.model.b_2, 456)
        self.assertEqual(c_b.model.b_3, 3.14)
        self.assertEqual(c_b.c_1, 200)

    def test_type_validation(self):
        """Test that type field validation works correctly."""
        # Valid type should work
        a_config = AConfig(a_1=1, a_2="test")
        self.assertEqual(a_config.type, "A")

        # Invalid type should raise ValidationError
        with self.assertRaises(ValidationError) as cm:
            AConfig.model_validate({"type": "INVALID", "a_1": 1, "a_2": "test"})
        self.assertIn("Input should be 'A'", str(cm.exception))

    def test_dconfig_missing_type_field(self):
        """Test that DConfig raises TypeError due to missing type field."""
        # DConfig is already defined in the module, so we need to test that it would fail
        # We can do this by creating a new class that mimics DConfig's structure
        with self.assertRaises(TypeError) as cm:

            class TestConfig(AbstractConfig):
                test_1: int
                test_2: str
                # Missing: type: Literal["TEST"] = "TEST"

        self.assertIn("must declare `type: Literal['TAG'] = 'TAG'`", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
