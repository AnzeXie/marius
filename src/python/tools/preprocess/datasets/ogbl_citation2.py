from pathlib import Path
from marius.tools.preprocess.dataset import LinkPredictionDataset
from marius.tools.preprocess.utils import download_url, extract_file
import numpy as np
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.converters.spark_converter import SparkEdgeListConverter
import torch
import os


class OGBLCitation2(LinkPredictionDataset):

    def __init__(self, output_directory: Path, spark=False):

        super().__init__(output_directory, spark)

        self.dataset_name = "ogbl_citation2"
        self.dataset_url = "http://snap.stanford.edu/ogb/data/linkproppred/citation-v2.zip"

    def download(self, overwrite=False):

        self.input_train_edges_file = self.output_directory / Path("train.pt")
        self.input_valid_edges_file = self.output_directory / Path("valid.pt")
        self.input_test_edges_file = self.output_directory / Path("test.pt")

        download = False
        if not self.input_train_edges_file.exists():
            download = True
        if not self.input_valid_edges_file.exists():
            download = True
        if not self.input_test_edges_file.exists():
            download = True

        if download:
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=False)

            for file in (self.output_directory / Path("citation-v2/split/time")).iterdir():
                file.rename(self.output_directory / Path(file.name))

    def preprocess(self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False):
        train_idx = torch.load(self.input_train_edges_file)
        valid_idx = torch.load(self.input_valid_edges_file)
        test_idx = torch.load(self.input_test_edges_file)

        train_list = np.array([train_idx.get("source_node"),
                               train_idx.get("target_node")]).T
        valid_list = np.array([valid_idx.get("source_node"),
                               valid_idx.get("target_node")]).T
        test_list = np.array([test_idx.get("source_node"),
                              test_idx.get("target_node")]).T

        converter = SparkEdgeListConverter if self.spark else TorchEdgeListConverter
        converter = converter(
            output_dir=self.output_directory,
            train_edges=train_list,
            valid_edges=valid_list,
            test_edges=test_list,
            num_partitions=num_partitions,
            remap_ids=remap_ids,
            known_node_ids=[torch.arange(2927963)], # not all nodes appear in the edges, need to supply all node ids for the mapping to be correct
            format="numpy",
            partitioned_evaluation=partitioned_eval
        )

        return converter.convert()
