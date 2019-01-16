-module(tests).
-export([test_2d_list/3]).
-on_load(init/0).

init() ->
    ok = erlang:load_nif("./nif_tests", 0).

test_2d_list(_L, _r, _c) ->
    exit(nif_library_not_loaded).
