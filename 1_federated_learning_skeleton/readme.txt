Federated Learning: Skeleton with Toy Example
(c) 2017 Fraunhofer-Chalmers Research Centre for Industrial
         Mathematics, Department of Systems and Data Analysis

Research and development:
Gregor Ulm      - gregor.ulm@fcc.chalmers.se
Emil Gustavsson - emil.gustavsson@fcc.chalmers.se



1) Overview

Federated Learning (cf. McMahan et al., 2017) is a decentralized, i.e.
distributed, approach to Machine Learning. This implementation of the
general idea behind Federated Learning is one of if not the first
publicly available one, and also the first public implementation in
a functional programming language (Erlang).

Federated Learning consists of the following steps:
 - select a subset of clients
 - send the current model to each client
 - for each client, update the provided model based on local data
 - for each client, send updated model to server
 - aggregate the client models, for instance by averaging, in order to
   construct an improved global model

This demo illustrates Federated Learning with 25 clients, using a
contrived example. However, it is a sound foundation for performing
practically useful machine learning tasks.

The source code below was created at the Fraunhofer-Chalmers Centre
and constitutes an exploratory prototype. It was the foundation for
subsequent work that solves practical problems in distributed data
analytics for our industrial clients.



2) Execution

Launch the Erlang/OTP 18 shell with 'erl', compile the source with
'c(demo).', and execute it with 'demo:main()'.



3) Licence

The code in this repository is released unter the MIT License.
