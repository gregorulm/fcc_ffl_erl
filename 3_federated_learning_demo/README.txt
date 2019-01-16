Functional Federated Learning
=============================
2017-2018 Fraunhofer-Chalmers Research Centre for Industrial Mathematics

Demo and benchmarking code supporting the paper by G. Ulm, E. Gustavsson,
M. Jirstrand


Implementation:
---------------
Gregor Ulm     (gregor.ulm@fcc.chalmers.se)
Adrian Nilsson (adrian.nilsson@fcc.chalmers.se)
Simon Smith    (simon.smith@fcc.chalmers.se)


Content:
--------
/erl         system completely based on (sequential) Erlang
/erl_dist    system completely based on distributed Erlang
/erl_cnode   system using C nodes for computations
/erl_nif     system using C codes as NIFs for computations
/foundation  initial implementation of our Federated Learning framework
/other       script for benchmarking, native compile guide


Licence:
--------
Code contribution by FCC are released under the MIT license. Part of the
implementation uses the C library FANN, which is under the GNU Lesser
General Public License v2.1.


FANN vs Erlang Implementation:
-----------------------------
There are some subtle differences between our Erlang implementation of
an ANN and the implementation provided by the FANN library. We do not
consider these differences to be significant, but we want to record them
anyway.

FANN is using the preset 'FANN_TRAIN_BATCH', which is gradient decent
using standard backpropagation. Erlang is implemented as SGD, sometimes
also referred to as incremental gradient decent, with a mini-batch size
of 1. To use batch training in Erlang, one simply has to compute the
gradient for every training sample and then update the ANN with the mean
of the gradients.

Both approaches use the mean-squared error as the loss/cost function for
training.

Since the Erlang implementation does not use a learning rate, the
learning rate in FANN is explicitly set to 1. Its default is 0.7.

Both Erlang and FANN use the sigmoid activation function. In FANN, this
corresponds to the parameter 'FANN_SIGMOID' set for every layer.
However, FANN uses a possibly awkward definition of the sigmoid and its
derivative, which we believe to not be entirely correct: Since FANN's
steepness parameter, which has a default value of 0.5, is only present
in the derivative of the sigmoid, the constant factor of 2 in the
exponent of the sigmoid function is questionable. This can be seen in
the FANN source code available at the following link:

https://github.com/libfann/fann/blob
/d71d54788bee56ba4cf7522801270152da5209d7/src/include
/fann_activation.h#L39-L42

Our response to this problem is to set the steepness parameter of FANN
to 1, which makes the sigmoid implementation coherent. Furthermore, we
modified the Erlang code to use the corresponding sigmoid, i.e.
1/(1 + e^(-2*x)). As a consequence, both implementations should have
equivalent activation functions.
