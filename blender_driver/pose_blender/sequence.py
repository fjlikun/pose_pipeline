import json
import math
from pathlib import Path


def finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load_sequence(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') != '1.0':
        raise ValueError('Expected schema_version 1.0')
    if not finite_number(data.get('fps')) or data['fps'] <= 0:
        raise ValueError('fps must be positive and finite')
    coordinates = data.get('coordinates', {})
    if coordinates.get('translation_included') is not False:
        raise ValueError('Expected translation_included=false; global root motion is unsupported')
    matrix = coordinates.get('axis_transform_to_blender')
    if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(r, list) or len(r) != 3 or not all(map(finite_number, r)) for r in matrix):
        raise ValueError('Expected finite 3x3 axis_transform_to_blender')
    for i in range(3):
        for k in range(3):
            if abs(sum(matrix[i][j]*matrix[k][j] for j in range(3)) - (i == k)) > 1e-5:
                raise ValueError('Coordinate transform must be orthonormal')
    frames = data.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('frames must be a nonempty array')
    previous_index = previous_time = -1
    good = 0
    for frame in frames:
        index, time = frame.get('frame_index'), frame.get('time_seconds')
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index <= previous_index:
            raise ValueError('frame_index must be strictly increasing nonnegative integers')
        if not finite_number(time) or time < 0 or time <= previous_time:
            raise ValueError('time_seconds must be strictly increasing nonnegative numbers')
        previous_index, previous_time = index, time
        if frame.get('status') not in ('ok', 'missing', 'error'):
            raise ValueError('Unknown frame status')
        points = frame.get('pred_keypoints_3d')
        if frame['status'] == 'ok':
            if not isinstance(points, list) or len(points) != 70 or any(not isinstance(p, list) or len(p) != 3 or not all(map(finite_number, p)) for p in points):
                raise ValueError('ok frame requires finite 70x3 pred_keypoints_3d')
            good += 1
        elif points is not None:
            raise ValueError('missing/error frame must have null keypoints')
    if not good:
        raise ValueError('No usable pose frames')
    return data


def load_mapping(path):
    mapping = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(mapping.get('armature'), str) or not mapping['armature']:
        raise ValueError('mapping requires armature object name')
    names = set()
    for item in mapping.get('bones', []):
        if not isinstance(item.get('bone'), str) or item['bone'] in names:
            raise ValueError('Mapping bone names must be unique strings')
        names.add(item['bone'])
        for key in ('head', 'tail'):
            _indices(item.get(key))
        if 'x_hint' in item:
            if len(item['x_hint']) != 2:
                raise ValueError('x_hint is [positive indices, negative indices]')
            for indices in item['x_hint']:
                _indices(indices)
    if not names:
        raise ValueError('Mapping has no bones')
    return mapping


def _indices(indices):
    if not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < 70 for i in indices):
        raise ValueError('Each landmark selector must be a nonempty list of indices 0..69')


def transformed_points(frame, matrix):
    return [tuple(sum(row[k]*point[k] for k in range(3)) for row in matrix) for point in frame['pred_keypoints_3d']]
