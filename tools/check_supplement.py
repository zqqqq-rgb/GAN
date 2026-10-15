"""Validate the public reproducibility archive without inference or training."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json


def main():
    root = Path(__file__).resolve().parents[1]
    description = json.loads((root / 'reproducibility/ARCHIVE.json').read_text(encoding='utf-8'))
    archive = root / description['path']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == description['sha256']
    with ZipFile(archive) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert len(names) == len(set(names)), 'Duplicate archive entries'
        for name in names:
            assert not name.startswith(('/', '\\')) and '..' not in Path(name).parts
        index = json.loads(z.read('INDEX.json'))
        for item in index['files']:
            value = z.read(item['path'])
            assert len(value) == item['bytes'], item['path']
            assert hashlib.sha256(value).hexdigest() == item['sha256'], item['path']
        assert sum(not name.endswith('/') for name in names) == description['file_count']
    print(json.dumps({'archive_sha256': description['sha256'],
                      'indexed_files_verified': len(index['files']),
                      'archive_files': description['file_count'],
                      'training_or_inference_started': False}, indent=2))


if __name__ == '__main__':
    main()
