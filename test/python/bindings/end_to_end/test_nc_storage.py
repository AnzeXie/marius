import unittest
import shutil
from pathlib import Path
import pytest
import os
import marius as m
import torch

from test.python.constants import TMP_TEST_DIR, TESTING_DATA_DIR
from test.test_data.generate import generate_random_dataset
from test.test_configs.generate_test_configs import generate_configs_for_dataset


def run_configs(directory, partitioned_eval=False):
    for filename in os.listdir(directory):
        if filename.startswith("M-"):
            config_file = directory / Path(filename)
            print("|||||||||||||||| RUNNING CONFIG ||||||||||||||||")
            print(config_file)
            config = m.config.loadConfig(config_file.__str__(), True)

            if partitioned_eval:
                config.storage.full_graph_evaluation = False

            m.manager.marius_train(config)


class TestNCStorage(unittest.TestCase):
    output_dir = TMP_TEST_DIR / Path("storage")

    @classmethod
    def setUp(self):
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

    @classmethod
    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_no_valid(self):
        num_nodes = 500
        num_rels = 10
        num_edges = 10000

        name = "no_valid"
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                splits=[.9, .1],
                                feature_dim=10,
                                task="nc")

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer"],
                                     storage_names=["in_memory"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_only_train(self):
        num_nodes = 500
        num_rels = 10
        num_edges = 10000

        name = "only_train"
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                feature_dim=10,
                                task="nc")

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer"],
                                     storage_names=["in_memory"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_no_valid_no_relations(self):
        num_nodes = 500
        num_rels = 1
        num_edges = 10000

        name = "no_valid_no_relations"
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                splits=[.9, .1],
                                feature_dim=10,
                                task="nc")

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer"],
                                     storage_names=["in_memory"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_only_train_no_relations(self):
        num_nodes = 500
        num_rels = 1
        num_edges = 10000

        name = "only_train_no_relations"
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                feature_dim=10,
                                task="nc")

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer"],
                                     storage_names=["in_memory"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_no_valid_buffer(self):
        num_nodes = 500
        num_rels = 10
        num_edges = 10000

        name = "no_valid_buffer"
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                splits=[.9, .1],
                                feature_dim=10,
                                num_partitions=8,
                                partitioned_eval=True,
                                task="nc")

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer"],
                                     storage_names=["part_buffer"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name), partitioned_eval=True)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_only_train_buffer(self):
        num_nodes = 500
        num_rels = 10
        num_edges = 10000

        name = "only_train_buffer"
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                feature_dim=10,
                                num_partitions=8,
                                task="nc")

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer"],
                                     storage_names=["part_buffer"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_no_valid_buffer_no_relations(self):
        num_nodes = 500
        num_rels = 1
        num_edges = 10000

        name = "no_valid_buffer_no_relations"
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                splits=[.9, .1],
                                num_partitions=8,
                                partitioned_eval=True,
                                feature_dim=10,
                                task="nc")

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer"],
                                     storage_names=["part_buffer"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name), partitioned_eval=True)

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_only_train_buffer_no_relations(self):
        num_nodes = 500
        num_rels = 1
        num_edges = 10000

        name = "only_train_buffer_no_relations"
        generate_random_dataset(output_dir=self.output_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                num_partitions=8,
                                feature_dim=10,
                                task="nc")

        generate_configs_for_dataset(self.output_dir / Path(name),
                                     model_names=["gs_1_layer"],
                                     storage_names=["part_buffer"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="nc")

        run_configs(self.output_dir / Path(name))
