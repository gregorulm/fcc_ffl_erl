-module(demo).
-compile([export_all]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Federated Learning: Skeleton with Toy Example
% (c) 2017 Fraunhofer-Chalmers Research Centre for Industrial
%          Mathematics, Department of Systems and Data Analysis
%
% Research and development:
% Gregor Ulm      - gregor.ulm@fcc.chalmers.se
% Emil Gustavsson - emil.gustavsson@fcc.chalmers.se
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Client Process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% produce a random integer between 1 and 10, inclusive
update_int() ->
    rand:uniform(10).



client() ->

    receive

    % receive current model from server
    { assignment, Model, Server_Pid } ->

        % simple computation that takes current model into account
        % '1' is added to avoid division-by-zero error
        Val = Model, % ((update_int() + trunc(Model)) rem 10) + 1,
        %io:format("Received model: ~p~n", [Model]),
        % send update to the server
        Server_Pid ! { update, self(), Val },
        client()

    end.





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Server process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% pick a random subset of clients;
% every client Pid has an even chance of being selected
random_subset([]     , Acc) -> Acc;
random_subset([H | T], Acc) ->
    case rand:uniform(2) of
        1 -> random_subset(T, Acc);
        2 -> random_subset(T, [H | Acc])
    end.



server(Client_Pids, Model) ->

    io:format("Client pids: ~p~n", [Client_Pids]),

    % send model to arbitrary subset of clients
    Subset = random_subset(Client_Pids, []),
    io:format("Subset of clients for update:~n ~p~n", [Subset]),
    io:format("Sending assignment to clients...~n", []),

    % send assignment
    TrainingData = [{[1.1, 1.2],[4.1, 4.2]}, {[2.1, 2.2],[5.1, 5.2]}, {[3.1, 3.2],[6.1, 6.2]}],
    lists:map(fun(X) -> call_cnode(self(), X, {assignment, Model, TrainingData}) end, Subset),

    % receive values
    Vals = [ receive { update, Val } -> Val end || _ <- Subset ],
    io:format("Received values: ~p~n", [Vals]),

    % update model by averaging new values

    % simple:
    {W_Inputs, W_Hiddens} = lists:unzip(Vals),

    Elem_Add = fun(X, Acc) -> element_wise_add(X, Acc) end,
    W_Input_Sum = lists:foldr( Elem_Add, [], W_Inputs ),
    W_Hidden_Sum = lists:foldr( Elem_Add, [], W_Hiddens ),

    N_Clients = length(Subset),
    New_W_Input = element_wise_div(W_Input_Sum, N_Clients),
    New_W_Hidden = element_wise_div(W_Hidden_Sum, N_Clients),

    Model_ = {New_W_Input, New_W_Hidden},

    % more contrived:
    % weighted average:
    % Total  = length(Client_Pids),
    % Sub    = length(Subset),
    % A      = Model * (Total - Sub),
    % B      = lists:sum(Vals),
    % Model_ = (A + B) / Total,

    io:format("Old Model: ~p, New Model: ~p~n", [Model, Model_]),

    % wait, then do another round
    io:format("Pausing for five seconds...~n~n~n~n", []),
    timer:sleep(5000),

    io:format("Enter next round~n", []),
    server(Client_Pids, Model_).

element_wise_div(M, Divisor) ->
  [[ X / Divisor || X <- Xs] || Xs <- M].

element_wise_add(M, []) -> M;
element_wise_add(M1, M2) ->
  F = fun(Xs, Ys) -> lists:zipwith(fun(X, Y) -> X + Y end, Xs, Ys) end,
  lists:zipwith(F, M1, M2).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Entry point
% run "demo:main()" to start demo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

spawn_clients(0, Acc) -> Acc;
spawn_clients(N, Acc) ->
    Pid = spawn(?MODULE, client, []),
    spawn_clients(N-1, [Pid | Acc]).

random_matix(Rows, Cols) ->
  [[random:uniform() || _ <- lists:seq(1, Cols)] || _ <- lists:seq(1, Rows)].



call_client_ann(Id) ->
  StrId = integer_to_list(Id),
  ServerNode = atom_to_list( node() ),
  os:cmd("x-terminal-emulator -e ./client_ann " ++ ServerNode ++ " " ++ StrId),
  { ok, Hostname } = inet:gethostname(),
  "c" ++ StrId ++ "@" ++ Hostname.


spawn_c_clients(0, Acc) -> Acc;
spawn_c_clients(N, Acc) ->
  Sname = call_client_ann(N),
  spawn_c_clients(N-1, [Sname | Acc]).

% Spawning a C Node
call_cnode(From, Sname, Msg) ->
    {any, list_to_atom(Sname)} ! { call, From, Msg }.

main() ->

    io:format("Federated Learning Demo~n~n", []),
    io:format("Spawning clients...~n~n", []),
    % spawn 25 clients
    Client_Pids = spawn_c_clients(25, []),

    {W_Input, W_Hidden} = {random_matix(3, 2), random_matix(2, 3)},

    % spawn server
    Model = {W_Input, W_Hidden},
    server(Client_Pids, Model).
