import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from typing import Literal

from pydantic import ValidationError

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
            with patch('framework.register.logger.error') as mock_log:
                with self.assertRaises(json.decoder.JSONDecodeError):
                    self.register.register(config_path, "invalid_app", TestConfig)

                # Verify error was printed
                mock_log.assert_called_once()
                error_message = mock_log.call_args[0][0]
                self.assertIn("Error registering invalid_app", error_message)
                self.assertIn("config file is not valid", error_message)
                
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
            with patch('framework.register.logger.error') as mock_log:
                with self.assertRaises(ValidationError):
                    self.register.register(config_path, "invalid_data_app", TestConfig)
                
                # Verify error was logged
                mock_log.assert_called_once()
                error_message = mock_log.call_args[0][0]
                self.assertIn("Error registering invalid_data_app", error_message)
                
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
            with patch('framework.register.logger.error') as mock_log:
                with self.assertRaises(ValidationError):
                    self.register.register(config_path, "empty_app", TestConfig)
                
                # Verify error was printed
                mock_log.assert_called_once()
                error_message = mock_log.call_args[0][0]
                self.assertIn("Error registering empty_app", error_message)

        finally:
            # Clean up temporary file
            os.unlink(config_path)

    def test_register_with_malformed_json(self):
        """Test registering with malformed JSON content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"type": "test", "name": "test", "value": }')  # Missing value
            config_path = f.name

        try:
            with self.assertRaises(json.decoder.JSONDecodeError):
                self.register.register(config_path, "malformed_app", TestConfig)
            # No print patching: error is logged, not printed
        finally:
            # Clean up temporary file
            os.unlink(config_path)

    def test_substitute_env_vars_simple_string(self):
        """Test environment variable substitution in simple string."""
        # Set up environment variable
        os.environ['TEST_VAR'] = 'test_value'
        
        try:
            result = self.register._substitute_env_vars('${TEST_VAR}')
            self.assertEqual(result, 'test_value')
        finally:
            # Clean up environment variable
            if 'TEST_VAR' in os.environ:
                del os.environ['TEST_VAR']

    def test_substitute_env_vars_missing_variable(self):
        """Test environment variable substitution when variable is missing."""
        result = self.register._substitute_env_vars('${MISSING_VAR}')
        self.assertEqual(result, '${MISSING_VAR}')  # Should return original string

    def test_substitute_env_vars_no_substitution(self):
        """Test string without environment variable syntax."""
        result = self.register._substitute_env_vars('regular_string')
        self.assertEqual(result, 'regular_string')

    def test_substitute_env_vars_multiple_variables(self):
        """Test environment variable substitution with multiple variables."""
        os.environ['VAR1'] = 'value1'
        os.environ['VAR2'] = 'value2'
        
        try:
            result = self.register._substitute_env_vars('${VAR1}_${VAR2}')
            self.assertEqual(result, 'value1_value2')
        finally:
            # Clean up environment variables
            for var in ['VAR1', 'VAR2']:
                if var in os.environ:
                    del os.environ[var]

    def test_substitute_env_vars_nested_dict(self):
        """Test environment variable substitution in nested dictionary."""
        os.environ['NESTED_VAR'] = 'nested_value'
        
        try:
            config_data = {
                'name': '${NESTED_VAR}',
                'nested': {
                    'value': '${NESTED_VAR}_suffix'
                }
            }
            result = self.register._substitute_env_vars(config_data)
            
            expected = {
                'name': 'nested_value',
                'nested': {
                    'value': 'nested_value_suffix'
                }
            }
            self.assertEqual(result, expected)
        finally:
            if 'NESTED_VAR' in os.environ:
                del os.environ['NESTED_VAR']

    def test_substitute_env_vars_nested_list(self):
        """Test environment variable substitution in nested list."""
        os.environ['LIST_VAR'] = 'list_value'
        
        try:
            config_data = ['${LIST_VAR}', 'regular_item', {'key': '${LIST_VAR}'}]
            result = self.register._substitute_env_vars(config_data)
            
            expected = ['list_value', 'regular_item', {'key': 'list_value'}]
            self.assertEqual(result, expected)
        finally:
            if 'LIST_VAR' in os.environ:
                del os.environ['LIST_VAR']

    def test_substitute_env_vars_mixed_types(self):
        """Test environment variable substitution with mixed data types."""
        os.environ['MIXED_VAR'] = 'mixed_value'
        
        try:
            config_data = {
                'string': '${MIXED_VAR}',
                'number': 42,
                'boolean': True,
                'null': None,
                'list': ['${MIXED_VAR}', 123, False],
                'nested': {
                    'deep': {
                        'value': '${MIXED_VAR}_deep'
                    }
                }
            }
            result = self.register._substitute_env_vars(config_data)
            
            expected = {
                'string': 'mixed_value',
                'number': 42,
                'boolean': True,
                'null': None,
                'list': ['mixed_value', 123, False],
                'nested': {
                    'deep': {
                        'value': 'mixed_value_deep'
                    }
                }
            }
            self.assertEqual(result, expected)
        finally:
            if 'MIXED_VAR' in os.environ:
                del os.environ['MIXED_VAR']

    def test_substitute_env_vars_partial_missing(self):
        """Test environment variable substitution with some missing variables."""
        os.environ['EXISTING_VAR'] = 'existing_value'
        
        try:
            config_data = {
                'existing': '${EXISTING_VAR}',
                'missing': '${MISSING_VAR}',
                'mixed': '${EXISTING_VAR}_${MISSING_VAR}'
            }
            result = self.register._substitute_env_vars(config_data)
            
            expected = {
                'existing': 'existing_value',
                'missing': '${MISSING_VAR}',
                'mixed': 'existing_value_${MISSING_VAR}'
            }
            self.assertEqual(result, expected)
        finally:
            if 'EXISTING_VAR' in os.environ:
                del os.environ['EXISTING_VAR']

    def test_register_with_env_vars_in_config(self):
        """Test registering a module with environment variables in config file."""
        os.environ['APP_NAME'] = 'env_test_app'
        os.environ['APP_VALUE'] = '999'
        
        try:
            # Create config file with environment variables
            config_data = {
                "type": "test",
                "name": "${APP_NAME}",
                "value": 999  # Use actual integer value, env var substitution happens after JSON parsing
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                config_path = f.name

            try:
                # Register the module
                self.register.register(config_path, "env_app", TestConfig)
                
                # Verify registration
                self.assertIn("env_app", self.register.registrations)
                registered_module = self.register.registrations["env_app"]
                self.assertIsInstance(registered_module, TestModule)
                self.assertEqual(registered_module.config.name, "env_test_app")
                self.assertEqual(registered_module.config.value, 999)
            finally:
                # Clean up temporary file
                os.unlink(config_path)
        finally:
            # Clean up environment variables
            for var in ['APP_NAME', 'APP_VALUE']:
                if var in os.environ:
                    del os.environ[var]

    def test_register_with_env_vars_string_values(self):
        """Test registering with environment variables that should remain as strings."""
        os.environ['STRING_VAR'] = 'string_value'
        
        try:
            # Create config file with environment variables in string format
            config_data = {
                "type": "test",
                "name": "${STRING_VAR}",
                "value": 42
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                config_path = f.name

            try:
                # Register the module
                self.register.register(config_path, "string_env_app", TestConfig)
                
                # Verify registration
                self.assertIn("string_env_app", self.register.registrations)
                registered_module = self.register.registrations["string_env_app"]
                self.assertIsInstance(registered_module, TestModule)
                self.assertEqual(registered_module.config.name, "string_value")
                self.assertEqual(registered_module.config.value, 42)
            finally:
                # Clean up temporary file
                os.unlink(config_path)
        finally:
            if 'STRING_VAR' in os.environ:
                del os.environ['STRING_VAR']

    def test_register_with_nested_env_vars(self):
        """Test registering with nested environment variables in config."""
        os.environ['BASE_URL'] = 'https://api.example.com'
        os.environ['API_VERSION'] = 'v1'
        
        try:
            # Create config file with nested environment variables
            config_data = {
                "type": "test",
                "name": "nested_env_test",
                "value": 100,
                "metadata": {
                    "url": "${BASE_URL}/api/${API_VERSION}",
                    "endpoints": [
                        "${BASE_URL}/users",
                        "${BASE_URL}/posts"
                    ]
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                config_path = f.name

            try:
                # Register the module
                self.register.register(config_path, "nested_env_app", TestConfig)
                
                # Verify registration
                self.assertIn("nested_env_app", self.register.registrations)
                registered_module = self.register.registrations["nested_env_app"]
                self.assertIsInstance(registered_module, TestModule)
                self.assertEqual(registered_module.config.name, "nested_env_test")
                self.assertEqual(registered_module.config.value, 100)
            finally:
                # Clean up temporary file
                os.unlink(config_path)
        finally:
            # Clean up environment variables
            for var in ['BASE_URL', 'API_VERSION']:
                if var in os.environ:
                    del os.environ[var]

    def test_substitute_env_vars_edge_cases(self):
        """Test environment variable substitution edge cases."""
        # Test empty string
        result = self.register._substitute_env_vars('')
        self.assertEqual(result, '')
        
        # Test string with only ${}
        result = self.register._substitute_env_vars('${}')
        self.assertEqual(result, '${}')
        
        # Test string with malformed syntax
        result = self.register._substitute_env_vars('${VAR')
        self.assertEqual(result, '${VAR')
        
        # Test string with nested braces
        result = self.register._substitute_env_vars('${VAR${NESTED}}')
        self.assertEqual(result, '${VAR${NESTED}}')
        
        # Test non-string types (should return as-is)
        self.assertEqual(self.register._substitute_env_vars(123), 123)
        self.assertEqual(self.register._substitute_env_vars(True), True)
        self.assertEqual(self.register._substitute_env_vars(None), None)

    def test_register_with_env_vars_failure_handling(self):
        """Test that environment variable substitution doesn't break error handling."""
        # Create config file with invalid data but valid env var syntax
        config_data = {
            "type": "test",
            "name": "${VALID_VAR}",
            # Missing required 'value' field
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch('framework.register.logger.error') as mock_log:
                with self.assertRaises(ValidationError):
                    self.register.register(config_path, "env_failure_app", TestConfig)
                
                # Verify error was printed
                mock_log.assert_called_once()
                error_message = mock_log.call_args[0][0]
                self.assertIn("Error registering env_failure_app", error_message)
                
        finally:
            # Clean up temporary file
            os.unlink(config_path)


if __name__ == "__main__":
    unittest.main()
