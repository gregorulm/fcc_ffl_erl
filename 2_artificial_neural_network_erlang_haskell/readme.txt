Artificial Neural Network in a Purely Functional Programming Style
(c) 2017 Fraunhofer-Chalmers Research Centre for Industrial
         Mathematics, Department of Systems and Data Analysis

Research and development:
Gregor Ulm      - gregor.ulm@fcc.chalmers.se
Emil Gustavsson - emil.gustavsson@fcc.chalmers.se



1) Overview
------------------------------------------------------------------------
This is a research implementation of an artificial neural network
(ANN) in a purely functional programming style, using Erlang.

The function 'main()' contains an example specifications of a
three-layer ANN. Our code is fully generic, however, meaning that ANNs
with arbitrary numbers of nodes in their input, hidden, and output
layers can be specified.

The goal of this implementation was the exploration of the suitability
of Erlang for performing computationally non-trivial distributed machine
learning tasks instead of delegating those tasks to 'C nodes'.



2) Content
------------------------------------------------------------------------
/Erlang
ann_basis.erl: bare-bones implementation using one input value
ann_batch.erl: extended version that performs training on a batch of
               inputs
/Haskell
ann_basic.hs: original Haskell prototype
ann_batch.hs: original Haskell prototype



3) Execution
------------------------------------------------------------------------
Launch the Erlang/OTP 18 shell with 'erl', compile the source with
'c(pf_ann).', and execute it with 'pf_ann:main()'.

The Haskell code can most easily be executed by launching 'ghci'
and loading the file with the command ':l', followed by the file
name, e.g. ':l ann_basic.hs'.



4) Licence
------------------------------------------------------------------------
The code in this repository is released unter the MIT License.
