import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from typing import Literal

# Add the project root to Python path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.register import Register
from framework.module import AbstractModule
from framework.config import AbstractConfig
from framework.singleton_decorator import singleton


class TestConfig(AbstractConfig):
    """Test configuration class for testing Register functionality"""
    type: Literal["test"] = "test"
    name: str
    value: int

    def build(self) -> "TestModule":
        return TestModule(self)


class TestModule(AbstractModule):
    """Test module class for testing Register functionality"""
    config: TestConfig

    def __init__(self, config: TestConfig):
        self.config = config

    def __call__(self, *args, **kwargs):
        return f"TestModule called with config: {self.config.name}"


class TestRegister(unittest.TestCase):
    """Test cases for the Register class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.register = Register()
        # Clear any existing registrations to ensure clean tests
        self.register.registrations.clear()

    def test_register_initialization(self):
        """Test that Register initializes with empty registrations dictionary."""
        self.assertEqual(self.register.registrations, {})
        # Note: There's a typo in the original code - it should be 'registrations' not 'registations'

    def test_register_with_valid_config_file(self):
        """Test registering a module with a valid JSON config file."""
        # Create a temporary config file
        config_data = {
            "type": "test",
            "name": "test_module",
            "value": 42
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            # Register the module
            self.register.register(config_path, "test_app", TestConfig)
            
            # Verify registration
            self.assertIn("test_app", self.register.registrations)
            registered_module = self.register.registrations["test_app"]
            self.assertIsInstance(registered_module, TestModule)
            self.assertEqual(registered_module.config.name, "test_module")
            self.assertEqual(registered_module.config.value, 42)
        finally:
            # Clean up temporary file
            os.unlink(config_path)

    def test_register_with_invalid_json_file(self):
        """Test registering with an invalid JSON file."""
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            config_path = f.name

        try:
            # Mock print to capture error output
            with patch('builtins.print') as mock_print:
                self.register.register(config_path, "invalid_app", TestConfig)
                
                # Verify error was printed
                mock_print.assert_called_once()
                error_message = mock_print.call_args[0][0]
                self.assertIn("Error registering invalid_app", error_message)
                self.assertIn("config file is not valid", error_message)
                
            # Verify no registration occurred
            self.assertNotIn("invalid_app", self.register.registrations)
        finally:
            # Clean up temporary file
            os.unlink(config_path)

    def test_register_with_missing_file(self):
        """Test registering with a non-existent config file."""
        non_existent_path = "/path/that/does/not/exist.json"
        
        with patch('builtins.print') as mock_print:
            with self.assertRaises(FileNotFoundError):
                self.register.register(non_existent_path, "missing_app", TestConfig)
            
        # Verify no registration occurred
        self.assertNotIn("missing_app", self.register.registrations)

    def test_register_with_invalid_config_data(self):
        """Test registering with valid JSON but invalid config data."""
        # Create a temporary config file with invalid data
        config_data = {
            "type": "test",
            "name": "test_module"
            # Missing required 'value' field
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch('builtins.print') as mock_print:
                self.register.register(config_path, "invalid_data_app", TestConfig)
                
                # Verify error was printed
                mock_print.assert_called_once()
                error_message = mock_print.call_args[0][0]
                self.assertIn("Error registering invalid_data_app", error_message)
                
            # Verify no registration occurred
            self.assertNotIn("invalid_data_app", self.register.registrations)
        finally:
            # Clean up temporary file
            os.unlink(config_path)

    def test_get_object_existing_registration(self):
        """Test getting an object that has been registered."""
        # First register a module
        config_data = {
            "type": "test",
            "name": "retrieved_module",
            "value": 100
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            self.register.register(config_path, "retrieval_app", TestConfig)
            
            # Get the registered object
            retrieved_object = self.register.get_object("retrieval_app")
            
            # Verify it's the correct object
            self.assertIsInstance(retrieved_object, TestModule)
            self.assertEqual(retrieved_object.config.name, "retrieved_module")
            self.assertEqual(retrieved_object.config.value, 100)
        finally:
            # Clean up temporary file
            os.unlink(config_path)

    def test_get_object_non_existent_registration(self):
        """Test getting an object that hasn't been registered."""
        with self.assertRaises(KeyError):
            self.register.get_object("non_existent_app")

    def test_multiple_registrations(self):
        """Test registering multiple modules."""
        # Register first module
        config_data_1 = {
            "type": "test",
            "name": "module_1",
            "value": 1
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
            json.dump(config_data_1, f1)
            config_path_1 = f1.name

        # Register second module
        config_data_2 = {
            "type": "test",
            "name": "module_2",
            "value": 2
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
            json.dump(config_data_2, f2)
            config_path_2 = f2.name

        try:
            # Register both modules
            self.register.register(config_path_1, "app_1", TestConfig)
            self.register.register(config_path_2, "app_2", TestConfig)
            
            # Verify both registrations
            self.assertEqual(len(self.register.registrations), 2)
            self.assertIn("app_1", self.register.registrations)
            self.assertIn("app_2", self.register.registrations)
            
            # Verify individual modules
            module_1 = self.register.get_object("app_1")
            module_2 = self.register.get_object("app_2")
            
            self.assertEqual(module_1.config.name, "module_1")
            self.assertEqual(module_1.config.value, 1)
            self.assertEqual(module_2.config.name, "module_2")
            self.assertEqual(module_2.config.value, 2)
        finally:
            # Clean up temporary files
            os.unlink(config_path_1)
            os.unlink(config_path_2)

    def test_register_overwrite_existing(self):
        """Test that registering with the same app_name overwrites the previous registration."""
        # Register first module
        config_data_1 = {
            "type": "test",
            "name": "original_module",
            "value": 1
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
            json.dump(config_data_1, f1)
            config_path_1 = f1.name

        # Register second module with same app_name
        config_data_2 = {
            "type": "test",
            "name": "overwritten_module",
            "value": 2
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
            json.dump(config_data_2, f2)
            config_path_2 = f2.name

        try:
            # Register first module
            self.register.register(config_path_1, "same_app", TestConfig)
            original_module = self.register.get_object("same_app")
            self.assertEqual(original_module.config.name, "original_module")
            
            # Register second module with same app_name
            self.register.register(config_path_2, "same_app", TestConfig)
            overwritten_module = self.register.get_object("same_app")
            
            # Verify the module was overwritten
            self.assertEqual(overwritten_module.config.name, "overwritten_module")
            self.assertEqual(overwritten_module.config.value, 2)
            
            # Verify only one registration exists
            self.assertEqual(len(self.register.registrations), 1)
        finally:
            # Clean up temporary files
            os.unlink(config_path_1)
            os.unlink(config_path_2)

    def test_register_with_different_module_types(self):
        """Test registering different types of modules."""
        # Create a second test config class
        class AnotherTestConfig(AbstractConfig):
            type: Literal["another_test"] = "another_test"
            title: str
            count: int

            def build(self) -> "AnotherTestModule":
                return AnotherTestModule(self)

        class AnotherTestModule(AbstractModule):
            config: AnotherTestConfig

            def __init__(self, config: AnotherTestConfig):
                self.config = config

        # Register first type
        config_data_1 = {
            "type": "test",
            "name": "test_module",
            "value": 42
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
            json.dump(config_data_1, f1)
            config_path_1 = f1.name

        # Register second type
        config_data_2 = {
            "type": "another_test",
            "title": "Another Module",
            "count": 10
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
            json.dump(config_data_2, f2)
            config_path_2 = f2.name

        try:
            # Register both types
            self.register.register(config_path_1, "test_app", TestConfig)
            self.register.register(config_path_2, "another_app", AnotherTestConfig)
            
            # Verify both registrations
            test_module = self.register.get_object("test_app")
            another_module = self.register.get_object("another_app")
            
            self.assertIsInstance(test_module, TestModule)
            self.assertIsInstance(another_module, AnotherTestModule)
            
            self.assertEqual(test_module.config.name, "test_module")
            self.assertEqual(another_module.config.title, "Another Module")
        finally:
            # Clean up temporary files
            os.unlink(config_path_1)
            os.unlink(config_path_2)

    def test_register_with_empty_json_file(self):
        """Test registering with an empty JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{}")
            config_path = f.name

        try:
            with patch('builtins.print') as mock_print:
                self.register.register(config_path, "empty_app", TestConfig)
                
                # Verify error was printed
                mock_print.assert_called_once()
                error_message = mock_print.call_args[0][0]
                self.assertIn("Error registering empty_app", error_message)
                
            # Verify no registration occurred
            self.assertNotIn("empty_app", self.register.registrations)
        finally:
            # Clean up temporary file
            os.unlink(config_path)

    def test_register_with_malformed_json(self):
        """Test registering with malformed JSON content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"type": "test", "name": "test", "value": }')  # Missing value
            config_path = f.name

        try:
            with patch('builtins.print') as mock_print:
                self.register.register(config_path, "malformed_app", TestConfig)
                
                # Verify error was printed
                mock_print.assert_called_once()
                error_message = mock_print.call_args[0][0]
                self.assertIn("Error registering malformed_app", error_message)
                
            # Verify no registration occurred
            self.assertNotIn("malformed_app", self.register.registrations)
        finally:
            # Clean up temporary file
            os.unlink(config_path)


if __name__ == "__main__":
    unittest.main()
