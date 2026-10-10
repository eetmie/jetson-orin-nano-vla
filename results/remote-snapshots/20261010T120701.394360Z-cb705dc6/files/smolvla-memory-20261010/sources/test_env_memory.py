"""Environment fingerprinting must not load unused inference frameworks."""
import builtins
import importlib.metadata
import sys
import types
import unittest
from unittest.mock import patch

from bench.runner import _package_versions


class EnvMemoryTests(unittest.TestCase):
    def test_versions_preserve_loaded_runtime_and_skip_framework_imports(self):
        modules = ['torch', 'onnxruntime', 'tensorrt', 'transformers', 'lerobot', 'numpy']
        installed = {'torch':'2.11', 'onnxruntime-gpu':'1.25', 'tensorrt-cu13':'10.16',
                     'transformers':'5.5', 'numpy':'2.3'}
        def version(name):
            if name not in installed:
                raise importlib.metadata.PackageNotFoundError(name)
            return installed[name]
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in modules:
                raise AssertionError(f'environment collection imported {name}')
            return original_import(name, *args, **kwargs)
        loaded = {name:None for name in modules}
        loaded['tensorrt'] = types.SimpleNamespace(__version__='10.16.actual')
        with patch.dict(sys.modules, loaded), \
             patch('bench.runner.importlib.metadata.version', side_effect=version), \
             patch('builtins.__import__', side_effect=guarded_import):
            result = _package_versions()
        self.assertEqual(result, {'torch':'2.11', 'onnxruntime':'1.25',
            'tensorrt':'10.16.actual', 'transformers':'5.5', 'numpy':'2.3'})


if __name__ == '__main__':
    unittest.main()
