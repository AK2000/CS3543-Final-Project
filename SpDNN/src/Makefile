CXX = clang++
CXXFLAGS = -std=c++11 -DINDPREC=int -DVALPREC=float -DFEATPREC=float -Wall -shared -fPIC -I../../extern/pybind11/include -I../../extern/eigen-3.4.0
PYTHON_INCLUDES=$$(python3-config --includes)
OPTFLAGS = -O3 -g

TARGETS = inference

# ----- Make Rules -----

all:	$(TARGETS)

inference:  inference.cpp include/inference.h
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $(PYTHON_INCLUDES) $< -o spdnn$$(python3-config --extension-suffix)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt *.bin core *.html *.xml *.so