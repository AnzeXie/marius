//
// Created by Jason Mohoney on 2/15/22.
//

#include "common/pybind_headers.h"

#include "nn/layers/reduction/concat.h"

void init_concat_reduction_layer(py::module &m) {
    py::class_<ConcatReduction, ReductionLayer, std::shared_ptr<ConcatReduction>>(m, "ConcatReduction")
            .def(py::init<shared_ptr<LayerConfig>, torch::Device>(),
                 py::arg("layer_config"),
                 py::arg("device"))
            .def(py::init([](int input_dim,
                             int output_dim,
                             std::optional<torch::Device> device,
                             InitConfig init,
                             bool bias,
                             InitConfig bias_init,
                             string activation) {

                     auto layer_config = std::make_shared<LayerConfig>();
                     layer_config->input_dim = input_dim;
                     layer_config->output_dim = output_dim;
                     layer_config->type = LayerType::GNN;

                     auto layer_options = std::make_shared<ReductionLayerOptions>();
                     layer_options->type = ReductionLayerType::CONCAT;
                     layer_config->options = layer_options;

                     layer_config->init = std::make_shared<InitConfig>(init);
                     layer_config->bias = bias;
                     layer_config->bias_init = std::make_shared<InitConfig>(bias_init);
                     layer_config->optimizer = nullptr;
                     layer_config->activation = getActivationFunction(activation);

                     torch::Device torch_device = torch::kCPU;
                     if (device.has_value()) {
                         torch_device = device.value();
                     }

                     return std::make_shared<ConcatReduction>(layer_config, torch_device);

                 }), py::arg("input_dim"),
                 py::arg("output_dim"),
                 py::arg("device") = py::none(),
                 py::arg("init") = InitConfig(InitDistribution::GLOROT_UNIFORM, nullptr),
                 py::arg("bias") = false,
                 py::arg("bias_init") = InitConfig(InitDistribution::ZEROS, nullptr),
                 py::arg("activation") = "none")
            .def("forward", &ConcatReduction::forward, py::arg("input"))
            .def("reset", &ConcatReduction::reset);
}