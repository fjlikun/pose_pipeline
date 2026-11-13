"""Wall-clock stage times, with explicit CUDA/ROCm completion at boundaries."""
from contextlib import contextmanager
from time import perf_counter


STAGES = ('decode', 'rgb_conversion', 'detection', 'bbox_dedup', 'tracking',
          'sam', 'scene_position', 'raw_archive_write', 'sample_index_write')


def model_synchronizer(*models):
    devices = sorted({str(model.device) for model in models
                      if str(getattr(model, 'device', '')).split(':')[0] == 'cuda'})
    if not devices:
        return lambda: None
    import torch

    def synchronize():
        for device in devices:
            torch.cuda.synchronize(device)

    return synchronize


class FrameTiming:
    def __init__(self, synchronize, started_at):
        self.synchronize = synchronize
        self.started_at = started_at
        self.stages = dict.fromkeys(STAGES, 0.0)
        self.sam_calls = []

    @contextmanager
    def measure(self, stage):
        self.synchronize()
        started = perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.stages[stage] += perf_counter() - started

    def record(self, frame_index, time_seconds, detection_count):
        self.synchronize()
        return dict(frame_index=frame_index, time_seconds=time_seconds, units='seconds',
                    detection_count=detection_count, stages_seconds=dict(self.stages),
                    sam_calls=list(self.sam_calls), frame_total_seconds=perf_counter()-self.started_at)
