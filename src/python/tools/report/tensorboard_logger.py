# taking reference from torch.utils.tensorboard
from pathlib import Path
import numpy as np
import time
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto import event_pb2

class tensorboard_logger(object):

    def __init__(self, log_directory):
        if not Path(log_directory).exists():
            raise RuntimeError("Error: log directory not found")
        else:
            self.log_directory = log_directory
        
        self.file_writer = None
        self._set_file_writer()

    def _set_file_writer(self):
        if self.file_writer is None:
            self.file_writer = FileWriter(self.log_directory)
        return self.file_writer

    def add_scalar(self, tag, scalar_value, global_step):
        scalar_value = self._make_np(scalar_value)
        assert scalar_value.squeeze().ndim == 0, "scalar should be 0D"
        scalar_value = float(scalar_value)

        current_summary = Summary(value=[Summary.Value(tag=tag, simple_value=scalar_value)])
        self.file_writer.add_summary(current_summary, global_step, None)

    def _make_np(self, x):
        if isinstance(x, np.ndarray):
            return x
        if np.isscalar(x):
            return np.array([x])

class FileWriter(object):
    def __init__(self, log_directory):
        self.event_writer = EventFileWriter(log_directory)

    def add_event(self, event, step=None, walltime=None):
        event.wall_time = time.time() if walltime is None else walltime
        if step is not None:
            event.step = int(step)
        self.event_writer.add_event(event)

    def add_summary(self, summary, global_step=None, walltime=None):
        event = event_pb2.Event(summary=summary)
        self.add_event(event, global_step, walltime)