#!/usr/bin/env python3
"""
Download and extract the UCI HAR Dataset into the `datasets/` folder.

Usage:
    python helper_functions/download_dataset.py            # download + extract
    python helper_functions/download_dataset.py --force    # re-download even if present
    python helper_functions/download_dataset.py --zip path/to/UCI_HAR_Dataset.zip   # use a local zip

The dataset is tried from a GitHub Release mirror of this repository first and
from the official UCI repository second. Every download is verified with SHA-256.

Dataset license: use in publications must cite Anguita et al., ESANN 2013.
Commercial use is prohibited. See datasets/read_this_for_the_dataset_folder.txt.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
main_app_path = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
DATASETS_DIR = os.path.join(main_app_path, 'datasets')

DATASET_FOLDER = 'UCI HAR Dataset'
DATASET_SHA256 = '2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0'
DATASET_URLS = [
    # GitHub Release mirror (primary)
    'https://github.com/tzamalisp/Human-Activity-Recognition-with-Tensorflow2-and-Keras/'
    'releases/download/dataset-v1.0/UCI_HAR_Dataset.zip',
    # Official UCI Machine Learning Repository (fallback)
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip',
]

# Files the data loader needs; used to decide whether the dataset is present.
REQUIRED_FILES = [
    os.path.join(group, 'Inertial Signals', '{}_{}_{}.txt'.format(signal, axis, group))
    for group in ('train', 'test')
    for signal in ('total_acc', 'body_acc', 'body_gyro')
    for axis in ('x', 'y', 'z')
] + [os.path.join('train', 'y_train.txt'), os.path.join('test', 'y_test.txt')]


def dataset_is_present(data_folder=DATASET_FOLDER):
    root = os.path.join(DATASETS_DIR, data_folder)
    return all(os.path.isfile(os.path.join(root, f)) for f in REQUIRED_FILES)


def sha256sum(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, dest):
    print('Downloading: {}'.format(url))
    request = urllib.request.Request(url, headers={'User-Agent': 'har-dataset-downloader'})
    with urllib.request.urlopen(request, timeout=60) as response, open(dest, 'wb') as out:
        total = int(response.headers.get('Content-Length') or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print('\r  {:5.1f}% ({:.1f} / {:.1f} MB)'.format(100 * done / total, done / 1e6, total / 1e6),
                      end='', flush=True)
            else:
                print('\r  {:.1f} MB'.format(done / 1e6), end='', flush=True)
    print()


def _extract(zip_path, data_folder):
    """Extract the archive to a staging folder (skipping macOS metadata), then move it to datasets/<data_folder>."""
    wanted = os.path.join(DATASETS_DIR, data_folder)
    print('Extracting to: {}'.format(wanted))
    os.makedirs(DATASETS_DIR, exist_ok=True)
    staging = os.path.realpath(tempfile.mkdtemp(prefix='.extract_', dir=DATASETS_DIR))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                name = member.filename
                base = os.path.basename(name.rstrip('/'))
                if name.startswith('__MACOSX/') or base == '.DS_Store' or base.startswith('._'):
                    continue
                # guard against path traversal
                destination = os.path.realpath(os.path.join(staging, name))
                if not destination.startswith(staging + os.sep):
                    raise RuntimeError('Unsafe path in archive: {}'.format(name))
                archive.extract(member, staging)

        extracted = os.path.join(staging, DATASET_FOLDER)
        if not os.path.isdir(extracted):
            raise RuntimeError('Archive does not contain a "{}" folder'.format(DATASET_FOLDER))
        if os.path.exists(wanted):
            shutil.rmtree(wanted)
        shutil.move(extracted, wanted)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def ensure_dataset(data_folder=DATASET_FOLDER, force=False, local_zip=None):
    """Make sure the dataset exists under datasets/<data_folder>; download it if missing."""
    if not force and local_zip is None and dataset_is_present(data_folder):
        print('Dataset already present: {}'.format(os.path.join(DATASETS_DIR, data_folder)))
        return os.path.join(DATASETS_DIR, data_folder)

    if local_zip is not None:
        _extract(local_zip, data_folder)
    else:
        errors = []
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = os.path.join(tmp, 'dataset.zip')
            for url in DATASET_URLS:
                try:
                    _download(url, zip_path)
                    checksum = sha256sum(zip_path)
                    if checksum != DATASET_SHA256:
                        raise ValueError('SHA-256 mismatch (got {})'.format(checksum))
                    _extract(zip_path, data_folder)
                    break
                except Exception as err:  # try the next mirror
                    print('  failed: {}'.format(err))
                    errors.append('{} -> {}'.format(url, err))
            else:
                raise RuntimeError('Could not download the dataset from any source:\n  ' + '\n  '.join(errors))

    if not dataset_is_present(data_folder):
        raise RuntimeError('Dataset extracted but expected files are missing in datasets/{}'.format(data_folder))
    print('Dataset ready: {}'.format(os.path.join(DATASETS_DIR, data_folder)))
    return os.path.join(DATASETS_DIR, data_folder)


def main():
    parser = argparse.ArgumentParser(description='Download the UCI HAR Dataset into datasets/.')
    parser.add_argument('--folder', default=DATASET_FOLDER,
                        help='Folder name under datasets/ (must match data_folder in configs/nn_setup.yml)')
    parser.add_argument('--force', action='store_true', help='Download again even if the dataset exists')
    parser.add_argument('--zip', dest='local_zip', help='Extract from a local zip instead of downloading')
    args = parser.parse_args()
    try:
        ensure_dataset(args.folder, force=args.force, local_zip=args.local_zip)
    except Exception as err:
        print('Error: {}'.format(err))
        sys.exit(1)


if __name__ == '__main__':
    main()
