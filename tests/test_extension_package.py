"""Standard-library tests: python -m unittest discover -s tests -p test_extension_package.py"""
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

spec = importlib.util.spec_from_file_location('extension', Path(__file__).resolve().parents[1] / 'scripts/package_extension.py')
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'extension'
        for name, content in {
            'README.md': b'Exact reproduction commands',
            'extension_report.pdf': b'%PDF-1.4\nfixture',
            'code/main.py': b'# student code',
            'configs/run.json': b'{}',
            'results/metrics.json': b'{}',
        }.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.out = self.root / '123_extension.zip'

    def test_round_trip_and_no_overwrite(self):
        extension.package(self.source, '123', self.out)
        self.assertEqual(extension.validate(self.out), 5)
        with self.assertRaises(ValueError):
            extension.package(self.source, '123', self.out)

    def test_missing_report(self):
        (self.source / 'extension_report.pdf').unlink()
        with self.assertRaises(ValueError):
            extension.package(self.source, '123', self.out)

    def test_unsafe_archive(self):
        with zipfile.ZipFile(self.out, 'w') as archive:
            archive.writestr('../secret', 'bad')
        with self.assertRaises(ValueError):
            extension.validate(self.out)

    def test_extra_checkpoint(self):
        (self.source / 'last.pt').write_bytes(b'checkpoint')
        with self.assertRaises(ValueError):
            extension.package(self.source, '123', self.out)

    def test_optional_checkpoint(self):
        (self.source / 'extension_best.pt').write_bytes(b'not loaded by checker')
        extension.package(self.source, '123', self.out)
        self.assertEqual(extension.validate(self.out), 6)

    def test_data_not_allowed(self):
        (self.source / 'data').mkdir()
        (self.source / 'data/train.bin').write_bytes(b'not for upload')
        with self.assertRaises(ValueError):
            extension.package(self.source, '123', self.out)


if __name__ == '__main__':
    unittest.main()
