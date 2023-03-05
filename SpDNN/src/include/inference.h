#ifndef SPDNN_INFERENCE
#define SPDNN_INFERENCE

#ifndef INDPREC
#define INDPREC int
#endif
#ifndef VALPREC
#define VALPREC float
#endif
#ifndef FEATPREC
#define FEATPREC float
#endif

#define BATCH_SIZE 64

#include <vector>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

struct Edge {
    int source;
    int dest;
    FEATPREC weight;
    bool activation;

    Edge(const int source, const int dest, const FEATPREC weight, 
            const bool activation) : 
        source(source), dest(dest), weight(weight), activation(activation) {}
};

#endif