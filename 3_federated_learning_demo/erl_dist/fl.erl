-module(fl).
-export([main/0, client/0, init_client/1]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Federated Learning: Artifician Neural Network
% (c) 2017 Fraunhofer-Chalmers Research Centre for Industrial
%          Mathematics, Department of Systems and Data Analysis
%
% Research and development:
% Gregor Ulm      - gregor.ulm@fcc.chalmers.se
% Emil Gustavsson - emil.gustavsson@fcc.chalmers.se
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Federated Learning (cf. McMahan et al., 2017) is a decentralized, i.e.
% distributed, approach to Machine Learning. This implementation of the
% general idea behind Federated Learning is one of if not the first
% publicly available one, and also the first public implementation in
% a functional programming language (Erlang).
%
% Federated Learning consists of the following steps:
% - select a subset of clients
% - send the current model to each client
% - for each client, update the provided model based on local data
% - for each client, send updated model to server
% - aggregate the client models, for instance by averaging, in order to
%   construct an improved global model
%
% This demo illustrates Federated Learning with 25 clients, where each
% clent uses an artificial neural network.
%
% Execution:
% Launch the Erlang/OTP 18 shell with 'erl', compile the source with
% 'c(demo).', and execute it with 'ann:main()'.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dotProduct(X, Y)
  when length(X) == length(Y) -> dotProduct(X, Y, 0);

dotProduct(_, _) -> erlang:error("Length mismatch").

dotProduct([X|Xs], [Y|Ys], Acc) -> dotProduct(Xs, Ys, Acc + X * Y);
dotProduct([]    , []    , Acc) -> Acc.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Client Process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% data generation: with every new iteration of the client process,
% a new set of training data is generated
% N: number of data points
generate_data(0, Acc) -> Acc;
generate_data(N, Acc) ->
  X      = rand:uniform(),
  Y      = rand:uniform(),
  Input  = [X, Y],
  Target_1 = math:sqrt(X * Y),
  Target_2 = math:sqrt(Target_1),
  Target = [Target_1, Target_2],
  generate_data(N-1, [{Input, Target}|Acc]).


% objective function
sigmoid(X) -> 1.0 / (1.0 + math:exp(-2*X)).



% multiply list of inputs with list of lists of weights
forward(_    , []      , Acc) -> lists:reverse(Acc);
forward(Input, [W | Ws], Acc) ->

  Val = dotProduct(Input, W),

  forward(Input, Ws, [Val | Acc]).



% compute error in output layer
delta_output_error(Vals, Targets) ->
  lists:zipwith(
    fun(X, Y) -> 2 * X * (1.0 - X) * (X - Y) end, Vals, Targets).



% Compute new weights of output layer
% args: input values, errors of output layer, weights, accumulator
backprop(_ , []    , []      , Acc) -> lists:reverse(Acc);
backprop(In, [E|Es], [Ws|Wss], Acc) ->

  A = lists:zipwith(fun(W, I) -> W - (E * I) end, Ws, In),
  backprop(In, Es, Wss, [A|Acc]).



% compute errors of hidden layer neurons
% args: hidden layer output, output errors, output layer weights, acc
errors_hidden([]    , _         , _      , Acc) -> lists:reverse(Acc);
errors_hidden([H|Hs], Output_Err, Weights, Acc) ->

  % take first element of output weights for backpropagation,
  % result in a list of weights of outgoing edges for each hidden
  % layer neuron
  Outgoing = [ hd(X) || X <- Weights ],

  % remaining weights for next iteration
  Rest = [ tl(X) || X <- Weights ],

  % compute error for current hidden layer neuron
  TMP  = lists:zipwith(fun(X, E) -> E * X end, Outgoing, Output_Err),
  A    = lists:sum(TMP) * 2 * H * (1.0 - H),

  errors_hidden(Hs, Output_Err, Rest, [A|Acc]).



% Forward Pass
compute_forward(Input, Weights, Targets) ->

  {W_Input, [W_B_In, W_B_Out], W_Hidden} = Weights,
  Hidden_In  = forward(Input, W_Input, []),

  % add bias to hidden layer
  Hidden_In_ = lists:zipwith(
                 fun(X, Y) -> X + Y end, Hidden_In, W_B_In),

  Hidden_Out    = [ sigmoid(X) || X <- Hidden_In_ ],
  Output_In     = forward(Hidden_Out, W_Hidden, []),
  % add bias to output layer
  Output_In_ = lists:zipwith(fun(X, Y) -> X + Y end, Output_In, W_B_Out),
  Output_Out    = [ sigmoid(X) || X <- Output_In_ ],
  Output_Errors = delta_output_error(Output_Out, Targets),

  {Hidden_Out, Output_Errors}.



ann(Input, Weights, _Targets) ->

  {W_Input, [W_B_In, W_B_Out], W_Hidden} = Weights,

  % Forward Pass
  {Hidden_Out, Delta_Output_Errors} =
    compute_forward(Input, Weights, _Targets),

  % Update weights for output layer
  W_Hidden_     = backprop(Hidden_Out, Delta_Output_Errors, W_Hidden, []),

  Delta_Hidden_Errors = errors_hidden(
                    Hidden_Out, Delta_Output_Errors, W_Hidden, []),

  W_Input_ = backprop(Input, Delta_Hidden_Errors, W_Input, []),

  % update weights of bias node
  W_B_Out_  = lists:zipwith(fun(B, E) -> B - E end, W_B_Out, Delta_Output_Errors),
  W_B_In_  = lists:zipwith(fun(B, E) -> B - E end, W_B_In, Delta_Hidden_Errors),
  {W_Input_, [W_B_In_, W_B_Out_], W_Hidden_}.



client() ->

  receive
  % receive current model from server
  { assignment, Model_Ws, Server_Pid } ->

    Data = generate_data(250, []),
    Val = wrap_ann(Data, Model_Ws),
    Server_Pid ! { update, self(), Val },
    client()

  end.



% keep weights, iterate over input/target pairs
wrap_ann([]    , Weights) -> Weights;
wrap_ann([D|Ds], Weights) ->

  {Input, Target} = D,
  Weights_        = ann(Input, Weights, Target),
  wrap_ann(Ds, Weights_).





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Server process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


sum_up([]        , []        , Acc) -> lists:reverse(Acc);
sum_up([Xs | Xss], [Ys | Yss], Acc) ->

  A = lists:zipwith(fun(X, Y) -> X + Y end, Xs, Ys),
  sum_up(Xss, Yss, [A | Acc]).



fold_lists([]        , Acc) -> Acc;
fold_lists([Xs | Xss], Acc) ->
  Acc_ = sum_up(Xs, Acc, []),
  fold_lists(Xss, Acc_).



scale_nested_list(XXs, N) -> [ [ X * N || X <- Xs ] || Xs <- XXs ].

%server(Client_Pids_Nums, Data, Data_List, Model_Ws, N, StartTime, Stats) ->
server(Client_Pids_Nums, Data, Model_Ws, N, StartTime, Stats) ->
% Data: validation set

  % send model to arbitrary subset of clients
  %Subset = random_subset_guard(Client_Pids_Nums),

  Subset      = Client_Pids_Nums, % modified for benchmarking: selecting all clients
  Subset_pids = [ X || {X, _Num} <- Subset ],

  % send assignment
  [ X ! { assignment, Model_Ws, self() } || X <- Subset_pids ],


  % receive locally updated models from clients
  Vals = [
    receive { update, Pid, Val } -> Val end || Pid <- Subset_pids ],

  % separate received data
  {W_In, W_B, W_Out} = lists:unzip3(Vals),

  % Accumulators (second argument of fold_lists) starts with head of L
  Sum_Local_Ws = [ fold_lists(tl(L), hd(L)) || L <- [W_In, W_B, W_Out] ],

  % aggregate and average:

  % existing weight: curent model
  Total        = length(Client_Pids_Nums),
  Num_Active   = length(Subset),
  Num_Inactive = Total - Num_Active,

  % scale up: Weights * Num_Inactive
  Scaled_Ws = [ scale_nested_list(X, Num_Inactive) || X <- tuple_to_list(Model_Ws) ],

  Total_Ws_Sum = [ fold_lists([X], Y) || {X, Y} <- lists:zip(Sum_Local_Ws, Scaled_Ws) ],

  % scale down: divide by length(Client_Pids)
  Factor = 1.0 / length(Client_Pids_Nums),

  Model_Ws_ = list_to_tuple([ scale_nested_list(X, Factor) || X <- Total_Ws_Sum ]),

  % compute errors
  Total_Errors = error_forward(Model_Ws_, Data, []),

  Error_Sum = lists:sum(Total_Errors) / length(Total_Errors), % Mean of the square errors


  Eps = 0.00015, % pick a smaller value to extend training duration

  case lists:all(fun(E) -> abs(E) < Eps end, Total_Errors) of
    % training completed:
    true -> io:format("Done!~n", []),
            Stats;

    false ->

      case N rem 500 == 0 of

        false -> server(Client_Pids_Nums, Data,
                        Model_Ws_, N+1, StartTime, Stats);

        true -> io:format("Mean error:~p~n", [Error_Sum]),
                io:format("N: ~p~n~n", [N]),

                {MegaSec, Sec, Micro} = os:timestamp(),
                Time = (MegaSec*1000000 + Sec)*1000 + round(Micro/1000),

                Val = {Time, Error_Sum},
                Stats_ = maps:put(N, Val, Stats),

                case os:system_time(seconds) - StartTime >= 10*60 of % yields enough data for paper!
                  true  -> Stats_;
                  false -> server(Client_Pids_Nums, Data,
                                  Model_Ws_, N+1, StartTime, Stats_)
                end
      end

    end.



error_forward(_    , []    , Acc) -> Acc;
error_forward(Model, [D|Ds], Acc) ->

  {Input, Target}             = D,
  {W_Input, [W_B_In, W_B_Out], W_Hidden} = Model,

  Hidden_In  = forward(Input, W_Input, []),

  % add bias to hidden
  Hidden_In_ =
    lists:zipwith(fun(X, Y) -> X + Y end, Hidden_In, W_B_In),

  Hidden_Out = [ sigmoid(X) || X <- Hidden_In_ ],

  Output_In  = forward(Hidden_Out, W_Hidden, []),
  % add bias to output
  Output_In_ = lists:zipwith(fun(X, Y) -> X + Y end, Output_In, W_B_Out),
  Output_Out = [ sigmoid(X) || X <- Output_In_ ],

  A = error_function(Output_Out, Target),
  error_forward(Model, Ds, [A|Acc]).


error_function(Output, Target)
  when length(Output) == length(Target) ->
    error_function(Output, Target, 0).


error_function([], [], Acc) ->
  Acc/2; % divide by 2 to make the derivative nicer
error_function([Output|Os], [Target|Ts], Acc) ->
  error_function(Os, Ts, Acc + math:pow((Target - Output), 2)).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Entry point
% run "demo:main()" to start demo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


init_client(Server_Node) ->
  {s3rver, Server_Node} ! {initialized}.


start_client_node(Id)->
  StrId = integer_to_list(Id),
  Sname = "client" ++ StrId,
  Str = ("x-terminal-emulator -e \"erl -hidden -sname " ++ Sname ++
         " -eval \\\"fl:init_client('" ++ atom_to_list(node()) ++"')\\\"\""),
  %io:format(Str),
  os:cmd(Str),
  Sname.


% spawn and connect to all client nodes
spawn_clients(0, Acc) -> Acc;
spawn_clients(N, Acc) ->
  Sname = start_client_node(N),
  { ok, Hostname } = inet:gethostname(),
  Client_Node = list_to_atom(Sname ++ "@" ++ Hostname),
  receive {initialized} ->
    Pid = spawn(Client_Node, ?MODULE, client, [])
  end,
  spawn_clients(N-1, [{Pid, N} | Acc]).


main() ->

  io:format("Federated Learning Demo~n~n", []),

  io:format("Generating training data..~n~n", []),
  Num_Clients = 10,
%  {Data, Data_List} = generate_Data(Num_Clients, maps:new(), []),
  % clients retrieve data according to dict key
  % data is sent together with model, i.e. for client N, send
  % {Model, maps:get(N, Data)} from server

  register(s3rver, self()),
  io:format("Spawning clients...~n~n", []),
  Client_Pids = spawn_clients(Num_Clients, []),

  % hardcoded input: 2 input, 3 hidden, 2 output neurons
  % 1 bias node for hidden layer

  W_Input  = [[0.2, 0.1], [0.4, 0.8], [0.7, 0.6]],
  W_Bias   = [[0.1, 0.1, 0.1], [0.1, 0.1]],
  W_Hidden = [[0.3, 0.4, 0.2], [0.05, 0.20, 0.7]],
  % alternatively, randomize weights to a small number

  % train until target error rate reached
  Model_Ws = {W_Input, W_Bias, W_Hidden},

  N     = 0, % iteration count
  Stats = maps:new(), % key: Iteration, value: {time, total abs. error}

  StartTime = os:system_time(seconds),

  % Validation set:
  Data = generate_data(450, []), % 450 pairs
%  Res   = server(Client_Pids, Data, Data_List, Model_Ws, N, StartTime, Stats),

  Res   = server(Client_Pids, Data, Model_Ws, N, StartTime, Stats),

  % process and write results to file
  Content = extract_content(lists:sort(maps:keys(Res)), Res, []),
  write_results("data/output_erl_dist" ++ integer_to_list(StartTime) ++ ".csv", Content).



write_results(File, Xs) ->
  {ok, S} = file:open(File, write),
  lists:foreach(
    fun({X, Y, Z}) -> io:format(S, "~p,~p,~p~n",[X, Y, Z]) end, Xs),
  file:close(S).



extract_content([]    , _  , Acc) -> lists:reverse(Acc);
extract_content([K|Ks], Map, Acc) ->
  {Time, Error} = maps:get(K, Map),
  extract_content(Ks, Map, [{K, Time, Error}|Acc]).
