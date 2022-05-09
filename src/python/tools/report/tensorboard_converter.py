from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from pathlib import Path
import argparse
import numpy as np


def batch_timing_appender(tag, data, index, writer):
    writer.add_scalar(f"{tag}",data, index)
    pass

def batch_timing_converter(path, experiment_name, writer):
    batch_timing_data = pd.read_csv(path)
    for batch_idx in range(batch_timing_data["id"].size):
        for col in batch_timing_data.columns:
            if col == "id":
                continue
            writer.add_scalar(f"{path.stem}_{col}({experiment_name})", batch_timing_data[col][batch_idx], batch_idx)
    writer.flush()

def mem_usage_converter(path, experiment_name, writer):
    mem_usage_data = pd.read_csv(path)
    for t in range(mem_usage_data.shape[0]):
        for col in mem_usage_data.columns:
            writer.add_scalar(f"{path.stem}_{col}({experiment_name})", mem_usage_data[col][t], t)
    writer.flush()

def format_converter(log_dir, experiment_name, writer):
    for file in log_dir.iterdir():
        if "batch_timing" in file.name:
            batch_timing_converter(file, experiment_name, writer)
        elif "mem_usage" in file.name:
            mem_usage_converter(file, experiment_name, writer)

def set_args():
    parser = argparse.ArgumentParser(
        description="A converter that converts logged training information into Tensorboard formats",
        prog='tensorboard_converter')
    parser.add_argument('--log_directory', '-ld',
                        type=str,
                        metavar="log_directory",
                        help="Specifies the directory containing logged training infomation.")
    parser.add_argument('--experiment_name', '-en',
                        type=str,
                        metavar="experiment_name",
                        help="Specifies the experiment name.")
    parser.add_argument('--output_directory', '-o',
                        type=str,
                        metavar="output_directory",
                        help="Specifies the directory to save converted training information.")

    return parser

def set_summarywriter(output_dir):
    writer = SummaryWriter(log_dir=output_dir)
    return writer

def main():
    parser = set_args()
    args = parser.parse_args()
    
    log_dir = Path(args.log_directory)
    output_dir = Path(args.output_directory)

    assert(log_dir.exists())
    if not output_dir.exists():
        output_dir.mkdir()
    
    writer = set_summarywriter(output_dir)
    
    format_converter(log_dir, args.experiment_name, writer)

    writer.close()

if __name__ == "__main__":
    main()
