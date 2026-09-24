"""CPU-only regressions for PEP 660 contamination of archived source trees."""

import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import common


class SourceImportTests(unittest.TestCase):
    def setUp(self):
        self.original_meta_path = list(sys.meta_path)
        self.original_modules = dict(sys.modules)

    def tearDown(self):
        sys.meta_path[:] = self.original_meta_path
        for name in list(sys.modules):
            if name == "sglang" or name.startswith("sglang.") or name == "external_fixture":
                if name in self.original_modules:
                    sys.modules[name] = self.original_modules[name]
                else:
                    sys.modules.pop(name, None)

    def test_missing_generated_version_uses_pinned_fallback_without_foreign_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            package = root / "python/sglang"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("from .version import __version__\n")
            # Exercise the actual pinned fallback rather than reproducing it.
            pinned = Path(__file__).resolve().parent / "fixtures/pinned_version.py"
            (package / "version.py").write_text(pinned.read_text())
            (package / "selected.py").write_text("origin = 'selected'\n")
            namespace = package / "nested"
            namespace.mkdir()
            (namespace / "child.py").write_text("origin = 'nested selected'\n")
            foreign_version = root / "foreign_version.py"
            foreign_version.write_text("__version__ = 'foreign-editable'\n__version_tuple__ = ('foreign',)\n")
            external = root / "external.py"
            external.write_text("origin = 'external dependency'\n")
            seen = []

            class EditableFinder:
                def find_spec(self, fullname, path=None, target=None):
                    seen.append(fullname)
                    if fullname == "sglang._version":
                        return importlib.util.spec_from_file_location(fullname, foreign_version)
                    if fullname == "external_fixture":
                        return importlib.util.spec_from_file_location(fullname, external)
                    return None

            # This finder would fill a missing module even though sglang's
            # package __path__ points to the correct source, as in the image.
            sys.meta_path.insert(0, EditableFinder())
            common.isolate_sglang_imports(root)
            with patch.object(importlib.metadata, "version", return_value="metadata-fallback") as metadata:
                loaded = importlib.import_module("sglang")
            self.assertEqual(loaded.__version__, "metadata-fallback")
            metadata.assert_called_once_with("sglang")
            self.assertEqual(Path(loaded.__file__).resolve(), package / "__init__.py")
            self.assertNotIn("sglang._version", sys.modules)
            self.assertEqual(importlib.import_module("sglang.selected").origin, "selected")
            self.assertEqual(importlib.import_module("sglang.nested.child").origin, "nested selected")
            with self.assertRaises(ModuleNotFoundError) as error:
                importlib.import_module("sglang._version")
            self.assertEqual(error.exception.name, "sglang._version")
            self.assertFalse(any(name == "sglang" or name.startswith("sglang.") for name in seen))
            self.assertEqual(importlib.import_module("external_fixture").origin, "external dependency")
            self.assertIn("external_fixture", seen)

    def test_symlink_outside_selected_tree_cannot_supply_module(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            package = root / "python/sglang"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            foreign = root / "foreign.py"
            foreign.write_text("origin = 'foreign'\n")
            (package / "escape.py").symlink_to(foreign)
            common.isolate_sglang_imports(root)
            with self.assertRaisesRegex(ImportError, "Source isolation violation"):
                importlib.import_module("sglang.escape")

    def test_preloaded_namespace_is_rejected(self):
        with patch.dict(sys.modules, {"sglang._version": types.ModuleType("sglang._version")}):
            with self.assertRaisesRegex(RuntimeError, "fresh worker"):
                common.isolate_sglang_imports("/unused")


if __name__ == "__main__":
    unittest.main()
