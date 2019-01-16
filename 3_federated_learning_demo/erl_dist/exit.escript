#!/usr/bin/env escript
%%! -sname exitScript

main([ClientStrNum]) ->
  ClientNumbers = lists:seq(1, list_to_integer(ClientStrNum)),
  {ok, Host} = inet:gethostname(),

  ClientStr = ["client" ++ integer_to_list(Num) ++ "@" ++ Host
         || Num <- ClientNumbers],

  Clients = [list_to_atom(Client) || Client <- ClientStr],
  [exit_node(Client) || Client <- Clients],

  ok;

main(_) ->
  usage().


exit_node(Node) ->
  case net_adm:ping(Node) of
    pong ->
      case rpc:call(Node, init, stop, []) of
        _Res ->
          io:format("Exited: ~p~n", [Node]);
        { badrpc, Reason } ->
          io:format("RPC to ~p failed: ~p~n", [Node, Reason])
      end;
    pang ->
      io:format("exit_node: failed to ping ~p~n", [Node])
  end.


usage() ->
  io:format("usage: rpc with an exit signal to all nodes in the simulation~n"),
  halt(1). % return non-zero exit code
