-module(normalize).
-compile([export_all]).

% normalization of training data

generate_data(0, Acc) -> Acc;
generate_data(N, Acc) ->
  X      = random:uniform(),
  Y      = random:uniform(),
  Input  = {X, Y},
  Target_1 = math:sqrt(X * Y),
  Target_2 = math:sqrt(Target_1),
  Target = {Target_1, Target_2},
  generate_data(N-1, [{Input, Target}|Acc]).


normalize_data(Data) ->

  {Input, Target} = lists:unzip(Data),

  {I1, I2}   = lists:unzip(Input),
  {T1, T2}   = lists:unzip(Target),
  [I1_, I2_] = lists:map(fun(X) -> normalize(X) end, [I1 ,I2]),
  [T1_, T2_] = lists:map(fun(X) -> normalize(X) end, [T1, T2]),
  Input_     = lists:zip(I1_, I2_),
  Target_    = lists:zip(T1_, T2_),

  % turn into listt of tuples into list of lists:
  A = lists:map(fun({X, Y}) -> [X, Y] end, Input_ ),
  B = lists:map(fun({X, Y}) -> [X, Y] end, Target_),

  % list of tuples
  lists:zip(A, B).


normalize(Data) ->
  Min = lists:min(Data),
  Max = lists:max(Data),
  D   = Max - Min,
  lists:map(fun(X) -> ((X - Min)/ D) end, Data).


divide(0          , _, []       , Acc       ) -> Acc;
divide(Num_Clients, N, Data_List, Data_Map) ->
  {Vals, Rest} = lists:split(N, Data_List),
  Data_Map_    = maps:put(Num_Clients, Vals, Data_Map),
  divide(Num_Clients - 1, N, Rest, Data_Map_).


main() ->
  Num_Clients = 2,
  Data_Client = 5, % number of {input, target} pairs for each client
  Total = Num_Clients * Data_Client,

  % all data, corresponds to 'Data_List' in fl.erl
  Data_List  = normalize_data(generate_data(Total, [])),
  io:format("B: ~p~n~n", [Data_List]),

  % divide data among clients
  % 'Data_Maps' corresponds to 'Data' in fl.erl
  Data_Map = divide(Num_Clients, Data_Client, Data_List, maps:new()),
  io:format("B: ~p~n~n", [Data_Map]).

  % clients retrieve data according to dict key
  % data is sent together with model, i.e. for client N, send
  % {Model, maps:get(N, Data)} from server
