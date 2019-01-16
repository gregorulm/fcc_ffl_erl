-module(ann_batch).
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
        fun(X, Y) -> X * (1.0 - X) * (Y - X) end, Vals, Targets).


% Compute new weights of output layer
% args: input values, errors of output layer, weights, accumulator
backprop(_ , []    , []      , Acc) -> lists:reverse(Acc);
backprop(In, [E|Es], [Ws|Wss], Acc) ->
    A = lists:zipwith(
            fun(W, I) -> W + (E * I) end, Ws, In),

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


ann(Input, Weights, Targets) ->
    {W_Input, W_Bias, W_Hidden} = Weights,

    % Forward Pass
    Hidden_In  = forward(Input, W_Input, []),
    % add bias; W_Bias is a list of lists with one element each
    Hidden_In_ = lists:zipwith(
                    fun(X, Y) -> X + hd(Y) end, Hidden_In, W_Bias),
    Hidden_Out = lists:map(fun(X) -> sigmoid(X) end, Hidden_In_),

    Output_In  = forward(Hidden_Out, W_Hidden, []),
    Output_Out = lists:map(fun(X) -> sigmoid(X) end, Output_In),

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

    { Output_Errors, Weights_}.


% keep weights, iterate over input/target pairs
wrap_ann([]    , Weights, []    , Errors) ->
    {lists:reverse(Errors), Weights};

wrap_ann([I|Is], Weights, [T|Ts], Errors) ->
    { Error, Weights_ } = ann(I, Weights, T),
    wrap_ann(Is, Weights_, Ts, [Error | Errors]).


% Training: iterate until error below a given threshold is reached
iterate_eps(Eps, Config, N) ->
    {Inputs, Weights, Targets} = Config,
    Errors          = [],
    {Err, Weights_} = wrap_ann(Inputs, Weights, Targets, Errors),

    case N rem 50000 of
        0 ->
             io:format("Errors in output layer..~p~n", [Err]),
             io:format("End of iteration ~p~n", [N]),
             io:format("....................................~n", []);

        _ -> ok
    end,

    case check_xss(Err, Eps) of
        true -> io:format("Done (Iteration ~p)!~n", [N]),
                ok;

        false -> iterate_eps(Eps, {Inputs, Weights_, Targets}, N + 1)
    end.


% input: list of lists of errors; checks if error for all inputs is
% below threshold
check_xss([]         , _  ) -> true;
check_xss([Errors|Es], Eps) ->
    case lists:all(fun(X) -> abs(X) < Eps end, Errors) of
        true  -> check_xss(Es, Eps);
        false -> false
    end.


% Training: iterates N times over batch of input data
iterate_N(0, _     ) -> ok;
iterate_N(N, Config) ->
    {Inputs, Weights, Targets} = Config,

    Errors = [],
    {Err, Weights_} = wrap_ann(Inputs, Weights, Targets, Errors),

    io:format("Errors in output layer..~p~n", [Err]),
    io:format("Iterations left: ~p~n", [N - 1]),
    io:format("................................................~n", []),

    iterate_N(N - 1, {Inputs, Weights_, Targets} ).


% Defines the ANN
main() ->
    % Hardcoded input
    Inputs    = [[0.4, 0.4], [0.3, 0.2], [0.2, 0.2], [0.3, 0.3], [0.2, 0.1]],

    % ANN structure: 2 input, 3 hidden, 2 output neurons

    % Weights from 2 input to 4 hidden nodes
    W_Input  = [[0.01, 0.01], [0.01, 0.01], [0.01, 0.01]],

    % Weight from bias node (connects with hidden layer)
    W_Bias = [[0.1], [0.1], [0.1]],

    % Weights from 3 hidden to 1 output nodes
    W_Hidden = [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]],

    % Alternatively, randomize weights to a small number

    Targets  = [[0.0, 0.8], [0.1, 0.5], [0.0, 0.4], [0.0, 0.6], [0.1, 0.3]],

    %toy example: f(x, y) -> x - y, x + y

    Weights  = {W_Input, W_Bias, W_Hidden},
    Config   = {Inputs, Weights, Targets},

   % iterate_N(10000, Config).
   iterate_eps(0.01, Config, 0).

