#include "include/inference.h"

namespace py = pybind11;

FEATPREC ReLU(FEATPREC x){
    return x < 0.0 ? 0.0 : x > 32.0 ? 32.0 : x;
}

Eigen::MatrixXd infer_basic(const Eigen::SparseMatrix<int> features, const std::vector<Edge> &edges, long num_neurons, int num_features, int num_images)
{

    FEATPREC *neuron_values = new FEATPREC[num_neurons * BATCH_SIZE];
    Eigen::MatrixXd  result(num_images,num_features);

    for(int i = 0; i < num_images; i += BATCH_SIZE){
        memset(neuron_values, 0, sizeof(FEATPREC) * num_neurons * BATCH_SIZE);
        for(int k = 0; k < num_features; k++){
            for(int j = 0; j < BATCH_SIZE; j++){
                if(i + j >= num_images) continue;
                neuron_values[(k * BATCH_SIZE) + j] = (FEATPREC) features.coeff(i+j, k);
            }
        }
        for(Edge e : edges) {
            if(e.activation){
                for(int j = 0 ; j < BATCH_SIZE; j++) {
                    neuron_values[(e.dest * BATCH_SIZE) + j] = ReLU( \
                            neuron_values[(e.dest * BATCH_SIZE) + j] \
                            + (neuron_values[(e.source * BATCH_SIZE) + j]  * e.weight)
                        );
                }
            } else {
                for(int j = 0 ; j < BATCH_SIZE; j++) {
                    neuron_values[(e.dest * BATCH_SIZE) + j] += neuron_values[(e.source * BATCH_SIZE) + j]  * e.weight;
                }
            }
        }

        for(int k = 1; k <= num_features; k++) {
            for(int j = 0; j < BATCH_SIZE; j++) {
                int neuron = (num_neurons - k);
                int feature = num_features - k;
                if(i + j >= num_images) continue;
                result(i+j, feature) = neuron_values[(neuron * BATCH_SIZE) + j];
            }
        }
    }

    delete[] neuron_values;
    return result;
}

PYBIND11_MODULE(spdnn, m)
{
    m.doc() = "A module for performing inference of sparse networks";
    py::class_<Edge>(m, "Edge")
        .def(py::init<const int, const int, const FEATPREC, const bool>())
        .def_readwrite("source", &Edge::source)
        .def_readwrite("dest", &Edge::dest)
        .def_readwrite("weight", &Edge::weight);

    m.def("infer_basic", &infer_basic);

}