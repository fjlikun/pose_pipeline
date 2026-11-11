"""Incremental, pickle-free SAM numeric output archive. One NPZ per sample."""
import json
import os
from pathlib import Path
import numpy as np


def numeric_fields(value, prefix='', arrays=None, omitted=None):
    """Flatten mapping paths using JSON Pointer escaping; never save object arrays."""
    arrays = {} if arrays is None else arrays
    omitted = {} if omitted is None else omitted
    if isinstance(value, dict):
        for key, item in value.items():
            escaped = str(key).replace('~','~0').replace('/','~1')
            numeric_fields(item, f'{prefix}/{escaped}' if prefix else escaped, arrays, omitted)
    elif value is None:
        omitted[prefix] = 'None'
    else:
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        try:
            array = np.asarray(value)
            if array.dtype.kind not in 'biufc':
                omitted[prefix] = f'non-numeric dtype {array.dtype}'
            else:
                arrays[prefix] = array
        except (TypeError, ValueError):
            omitted[prefix] = 'non-array value'
    return arrays, omitted


def atomic_npz(path, arrays, compressed):
    temp = path.with_name(path.name + '.tmp')
    try:
        with temp.open('xb') as handle:
            (np.savez_compressed if compressed else np.savez)(handle, **arrays)
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()


class NPZArchive:
    def __init__(self, directory, estimator, metadata, compressed=True):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)  # Never mix runs or overwrite raw archives.
        self.compressed = compressed
        self.count = 0
        self.header = dict(archive_version='1.0', complete=False, compression=compressed,
                           metadata=metadata, assets=None,
                           frame_records='frames.jsonl', key_paths='JSON Pointer escaping without leading slash')
        faces = getattr(estimator, 'faces', None)
        if faces is not None:
            arrays, omitted = numeric_fields({'faces':faces})
            atomic_npz(self.directory/'assets.npz', arrays, compressed)
            self.header['assets'] = 'assets.npz'
        self._write_header()
        (self.directory/'frames.jsonl').touch()

    def _write_header(self):
        tmp = self.directory/'manifest.json.tmp'
        tmp.write_text(json.dumps(self.header, ensure_ascii=False, allow_nan=False, indent=2),encoding='utf-8')
        os.replace(tmp,self.directory/'manifest.json')

    def append(self, entry, prediction):
        record = {key:value for key,value in entry.items() if key != 'pred_keypoints_3d'}
        record['npz_file'] = None
        record['omitted_fields'] = {}
        if prediction is not None:
            arrays, omitted = numeric_fields(prediction)
            name = f"frame_{entry['frame_index']:09d}.npz"
            # Context names reserved outside SAM fields, no object/pickle payloads.
            arrays['__frame_index__'] = np.asarray(entry['frame_index'],dtype=np.int64)
            arrays['__time_seconds__'] = np.asarray(entry['time_seconds'],dtype=np.float64)
            atomic_npz(self.directory/name, arrays, self.compressed)
            record.update(npz_file=name, omitted_fields=omitted,
                          fields={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()})
        with (self.directory/'frames.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            handle.flush()
        self.count += 1

    def finish(self, metadata):
        self.header.update(complete=True, sampled_frames=self.count, metadata=metadata)
        self._write_header()
