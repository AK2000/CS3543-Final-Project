#include "include/inference.h"

namespace py = pybind11;

FEATPREC ReLU(FEATPREC x){
    return x < 0.0 ? 0.0 : x > 32.0 ? 32.0 : x;
 }

Eigen::MatrixXd infer_basic(const Eigen::SparseMatrix<int> features, const std::vector<Edge> &edges, long num_neurons, int num_features, int num_images, int batch_size)
{

    FEATPREC *neuron_values = new FEATPREC[num_neurons * batch_size];
    Eigen::MatrixXd  result(num_images,num_features);

    FEATPREC source_val;
    FEATPREC* partial_sum;

    for(int i = 0; i < num_images; i += batch_size){
        memset(neuron_values, 0, sizeof(FEATPREC) * num_neurons * batch_size);

        for(Edge e : edges) {
            for(int j = 0; j < batch_size; j++){
                if(e.source < num_features){
                    source_val = (FEATPREC) features.coeff(i+j, e.source);
                } else{
                    source_val = neuron_values[(e.source * batch_size) + j];
                }
                
                partial_sum = &neuron_values[(e.dest * batch_size) + j];
                *partial_sum += (source_val * e.weight);

                if(e.activation){
                    *partial_sum = ReLU(*partial_sum);
                }
            }
        }

        for(int k = 1; k <= num_features; k++)
        {
            for(int j = 0; j < batch_size; j++){
                int neuron = (num_neurons - k);
                int feature = num_features - k;
                result(i+j, feature) = neuron_values[(neuron * batch_size) + j];
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
        .def(py::init<const int, const int, const FEATPREC, const bool>());
    m.def("infer_basic", &infer_basic);

}