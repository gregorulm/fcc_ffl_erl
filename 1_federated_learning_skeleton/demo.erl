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
        Val = ((update_int() + trunc(Model)) rem 10) + 1,

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
    lists:map(fun(X) -> X ! { assignment, Model, self()} end, Subset),

    % receive values
    Vals = [ receive { update, Pid, Val } -> Val end || Pid <- Subset ],
    io:format("Received values: ~p~n", [Vals]),

    % update model by averaging new values

    % simple:
    % Model_ = Vals / length(Clients_Subset)

    % more contrived:
    % weighted average:
    Total  = length(Client_Pids),
    Sub    = length(Subset),
    A      = Model * (Total - Sub),
    B      = lists:sum(Vals),
    Model_ = (A + B) / Total,

    io:format("Old Model: ~p, New Model: ~p~n", [Model, Model_]),

    % wait, then do another round
    io:format("Pausing for five seconds...~n~n~n~n", []),
    timer:sleep(5000),

    io:format("Enter next round~n", []),
    server(Client_Pids, Model_).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Entry point
% run "demo:main()" to start demo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

spawn_clients(0, Acc) -> Acc;
spawn_clients(N, Acc) ->
    Pid = spawn(?MODULE, client, []),
    spawn_clients(N-1, [Pid | Acc]).


main() ->

    io:format("Federated Learning Demo~n~n", []),
    io:format("Spawning clients...~n~n", []),
    % spawn 25 clients
    Client_Pids = spawn_clients(25, []),

    % spawn server
    Model = 5,
    server(Client_Pids, Model).
