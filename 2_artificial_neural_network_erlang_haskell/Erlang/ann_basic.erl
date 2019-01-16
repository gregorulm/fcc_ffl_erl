-module(ann_basic).
-compile([export_all]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Artificial Neural Network in a Purely Functional Programming Style
% (c) 2017 Fraunhofer-Chalmers Research Centre for Industrial
%          Mathematics, Department of Systems and Data Analysis
%
% Research and development:
% Gregor Ulm      - gregor.ulm@fcc.chalmers.se
% Emil Gustavsson - emil.gustavsson@fcc.chalmers.se
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% In the hardcoded example in main() an artificial neural network (ANN)
% with two input neurons, two hidden neurons, and three output neurons
% is defined.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Objective function
sigmoid(X) ->
    1.0 / (1.0 + math:exp(-X)).


% In the forward pass the dot product of the input values and the
% weights of the outgoing edges is computed
forward(_    , []      , Acc) -> lists:reverse(Acc);
forward(Input, [W | Ws], Acc) ->
    Val = lists:sum(
            lists:zipwith(
                fun(X, Y) -> X * Y end, Input, W)),

    forward(Input, Ws, [Val | Acc]).


% Compute error in output layer
output_error(Vals, Targets) ->
    lists:zipwith(
        fun(X, Y) -> X * (1.0 - X) * (X - Y) end, Vals, Targets).


% Compute new weights of output layer
% args: input values, errors of output layer, weights, accumulator
backprop(_ , []    , []      , Acc) -> lists:reverse(Acc);
backprop(In, [E|Es], [Ws|Wss], Acc) ->
    A = lists:zipwith(
            fun(W, I) -> W - (E * I) end, Ws, In),

    backprop(In, Es, Wss, [A|Acc]).


% Compute errors of hidden layer neurons
% args: hidden layer output, output errors, output layer weights, acc
errors_hidden([]    , _         , _      , Acc) -> lists:reverse(Acc);
errors_hidden([H|Hs], Output_Err, Weights, Acc) ->
    % take first element of output weights for backpropagation;
    % results in a list of weights of outgoing edges for each hidden
    % layer neuron
    Outgoing = lists:map(fun(X) -> hd(X) end, Weights),

    % remaining weights for next iteration
    Rest = lists:map(fun(X) -> tl(X) end, Weights),

    % compute error for current hidden layer neuron
    TMP  = lists:zipwith(fun(X, E) -> E * X end, Outgoing, Output_Err),
    A    = lists:sum(TMP) * H * (1.0 - H),

    errors_hidden(Hs, Output_Err, Rest, [A|Acc]).


ann({Input, Weights, Targets}) ->

    {W_Input, W_Bias, W_Hidden} = Weights,

    % Forward Pass
    Hidden_In  = forward(Input, W_Input, []),

    % add bias; W_Bias is a list of lists with one element each
    Hidden_In_ = lists:zipwith(
                    fun(X, Y) -> X + hd(Y) end, Hidden_In, W_Bias),


    Hidden_Out = lists:map(fun(X) -> sigmoid(X) end, Hidden_In_),

    Output_In  = forward(Hidden_Out, W_Hidden, []),
    Output_Out = lists:map(fun(X) -> sigmoid(X) end, Output_In),

    % Target vs Output; provides information for the user
    io:format("Target values..~p~n", [Targets]),
    io:format("Output values .~p~n", [Output_Out]),
    Delta = lists:zipwith(fun(X, Y) -> X - Y end, Targets, Output_Out),
    io:format("Deviations~p~n", [Delta]),

    % Reverse pass
    Output_Errors = output_error(Output_Out, Targets),

    % Update weights for output layer
    W_Hidden_     = backprop(
                        Hidden_Out, Output_Errors, W_Hidden, []),

    Hidden_Errors = errors_hidden(
                        Hidden_Out, Output_Errors, W_Hidden_, []),

    W_Input_ = backprop(Input, Hidden_Errors, W_Input, []),

    % update weights of bias node
    W_Bias_  = backprop([1], Hidden_Errors, W_Bias, []),

    Weights_ = {W_Input_, W_Bias_, W_Hidden_},

    { Output_Errors, {Input, Weights_, Targets}}.


% Training: iterate N times
iterate_N(0, _)   -> ok;
iterate_N(N, ANN) ->
    {Errors, ANN_} = ann(ANN),
    io:format("Errors in output layer..~p~n", [Errors]),
    io:format("Iterations left: ~p~n", [N - 1]),
    io:format("................................................~n", []),

    iterate_N(N - 1, ANN_).


% Training: iterate until error below a given threshold is reached
iterate_eps(Eps, ANN, N) ->
    {Errors, ANN_} = ann(ANN),
    io:format("Errors in output layer..~p~n", [Errors]),
    io:format("End of iteration ~p~n", [N]),
    io:format("................................................~n", []),

    case lists:all(fun(X) -> abs(X) < Eps end, Errors) of
        true -> io:format("Done (Iteration ~p)!~n", [N]),
                ok;

        false -> iterate_eps(Eps, ANN_, N + 1)
    end.


% Defines the ANN
main() ->
    % Hardcoded input: 2 input, 3 hidden, 2 output neurons
    Input    = [0.50, 0.45],

    % Weights from 2 input to 3 hidden nodes
    W_Input  = [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]],

    % Weight from bias node (connects with hidden layer)
    W_Bias = [[0.1], [0.1], [0.1]],

    % Weights from 3 hidden to 2 output nodes
    W_Hidden = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],

    % Alternatively, randomize weights to a small number

    Targets  = [0.05, 0.95],
    % Of course the numbers are arbitrary

    Weights  = {W_Input, W_Bias, W_Hidden},
    Config   = {Input, Weights, Targets},

   %iterate_N(100, Config).
   iterate_eps(0.00000001, Config, 0).
