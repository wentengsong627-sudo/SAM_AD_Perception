"""Prepare the pinned model dependencies. No downloads occur during inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def valid(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256']


def destination(relative):
    path = ROOT / relative
    path.resolve().relative_to(ROOT)
    return path


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'SAM-AD-asset-preparer/1.0'})
    print('Downloading ' + path.name, flush=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as out:
            count = 0
            last = 0
            while block := response.read(4 * 1024 * 1024):
                out.write(block)
                count += len(block)
                if count - last >= 64 * 1024 * 1024:
                    print('  %d MiB' % (count // 1048576), flush=True)
                    last = count
        os.replace(temporary, path)
    except Exception:
        if temporary.exists(): temporary.unlink()
        raise


def prepare(backend='both', bundle=None, check=False):
    lock = json.loads((ROOT / 'assets.lock.json').read_text(encoding='utf-8'))
    rows = [x for x in lock['files'] if backend == 'both' or x['backend'] == backend]
    missing = []
    for item in rows:
        target = destination(item['path'])
        if valid(target, item): continue
        if target.exists():
            raise RuntimeError('Existing dependency differs from this release: ' + item['path'] +
                               '. It has been preserved; move it aside before retrying.')
        missing.append(item)
    if check:
        if missing: raise RuntimeError('Missing dependencies: ' + ', '.join(x['path'] for x in missing[:8]))
        return {'backend': backend, 'verified_dependencies': len(rows), 'offline_check': True}
    bundle = Path(bundle).resolve() if bundle else None
    archives = {}
    for item in missing:
        target = destination(item['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        candidate = bundle / item['path'] if bundle else None
        if candidate and valid(candidate, item):
            # Hard links avoid another 1.3 GB copy on the same disk. Inference
            # never modifies these files; every launch verifies their hashes.
            try: os.link(candidate, target)
            except OSError: shutil.copyfile(candidate, target)
        elif bundle:
            raise RuntimeError('Source bundle is missing or has a different file: ' + item['path'])
        elif 'archive' in item:
            name = item['archive']
            if name not in archives:
                cache = ROOT / '.asset-cache' / (name + '.zip')
                if not cache.is_file(): download(lock['archives'][name]['url'], cache)
                archive = zipfile.ZipFile(cache)
                roots = {p.split('/')[0] for p in archive.namelist() if '/' in p}
                if len(roots) != 1: raise RuntimeError('Unexpected upstream ZIP layout: ' + name)
                archives[name] = archive, next(iter(roots))
            archive, prefix = archives[name]
            # Never extract a supplied ZIP path. Only write a locked destination.
            data = archive.read(prefix + '/' + item['member'])
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise RuntimeError('Upstream source hash mismatch: ' + item['path'])
            target.write_bytes(data)
        else:
            cache = ROOT / '.asset-cache' / Path(item['path']).name
            if not valid(cache, item): download(item['url'], cache)
            if not valid(cache, item): raise RuntimeError('Checkpoint hash mismatch: ' + item['path'])
            shutil.copyfile(cache, target)
        if not valid(target, item): raise RuntimeError('Dependency verification failed: ' + item['path'])
    for archive, _ in archives.values(): archive.close()
    return {'backend': backend, 'verified_dependencies': len(rows), 'prepared': len(missing),
            'source': 'local verified bundle' if bundle else 'original upstream URLs'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['A', 'D', 'both'], default='both')
    parser.add_argument('--from-bundle', type=Path)
    parser.add_argument('--check', action='store_true', help='Verify offline without downloading')
    args = parser.parse_args()
    print(json.dumps(prepare(args.backend, args.from_bundle, args.check), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print('Asset preparation failed: ' + str(exc), file=sys.stderr)
        sys.exit(1)
