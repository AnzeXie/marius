# taking reference from torch.utils.tensorboard
from pathlib import Path
import numpy as np
import time
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto import event_pb2

class tensorboard_logger(object):

    def __init__(self, log_directory = "./"):
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

    def add_batch_timing_stats(self, scalars, global_step):
        self.add_scalar("wait_for_batch", scalars[0], global_step)
        self.add_scalar("sample_edges", scalars[1], global_step)
        self.add_scalar("sample_negatives", scalars[2], global_step)
        self.add_scalar("sample_neighbors", scalars[3], global_step)
        self.add_scalar("set_uniques_edges", scalars[4], global_step)
        self.add_scalar("set_uniques_neighbors", scalars[5], global_step)
        self.add_scalar("set_eval_filter", scalars[6], global_step)
        self.add_scalar("load_node_data_cpu", scalars[7], global_step)
        self.add_scalar("batch_host_queue", scalars[8], global_step)
        self.add_scalar("h2d_transfer", scalars[9], global_step)
        self.add_scalar("batch_device_queue", scalars[10], global_step)
        self.add_scalar("load_node_data_gqu", scalars[11], global_step)
        self.add_scalar("perform_map", scalars[12], global_step)
        self.add_scalar("forward_encoder", scalars[13], global_step)
        self.add_scalar("prepare_batch", scalars[14], global_step)
        self.add_scalar("forward_decoder", scalars[15], global_step)
        self.add_scalar("loss", scalars[16], global_step)
        self.add_scalar("backward", scalars[17], global_step)
        self.add_scalar("step", scalars[18], global_step)
        self.add_scalar("accumulate_gradients", scalars[19], global_step)
        self.add_scalar("gradient_device_queue", scalars[20], global_step)
        self.add_scalar("d2h_transfer", scalars[21], global_step)
        self.add_scalar("gradient_host_queue", scalars[22], global_step)
        self.add_scalar("update_embeddings", scalars[23], global_step)
        self.add_scalar("end_to_end", scalars[24], global_step)


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