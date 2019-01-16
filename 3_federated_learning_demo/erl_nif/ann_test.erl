-module(ann_test).
-export([test/0, train/2, init_ann/0, free_ann/0]).
-on_load(init/0).

init() ->
    ok = erlang:load_nif("./ann_nif", 0).

generate_helper(0, Acc) -> Acc;
generate_helper(N, Acc) ->
  X      = random:uniform(),
  Y      = random:uniform(),
  Input  = [X, Y],
  Target_1 = math:sqrt(X * Y),
  Target_2 = math:sqrt(Target_1),
  Target = [Target_1, Target_2],
  generate_helper(N-1, [{Input, Target}|Acc]).

test() ->
  W_Input  = [[0.2, 0.1], [0.4, 0.8], [0.7, 0.6]],
  W_Bias   = [[0.1, 0.1, 0.1], [0.1, 0.1]],
  W_Hidden = [[0.3, 0.4, 0.2], [0.05, 0.20, 0.7]],
  Model = {W_Input, W_Bias, W_Hidden},
  Data = generate_helper(250, []),
  %%io:format("Training data: ~p~n", [Data]),
  T = os:system_time(micro_seconds),
  New_Weights = train(Model, Data),
  Time = os:system_time(micro_seconds) - T,
  io:format("NIF time: ~p~n", [Time]),
  New_Weights.

train(_ModelW, _Data) ->
    exit(nif_library_not_loaded).
