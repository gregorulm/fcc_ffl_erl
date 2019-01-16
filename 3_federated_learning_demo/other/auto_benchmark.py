
import sys
import time
from os import system, chdir


'''
Here is a list of bools corresponding to each benchmark.
Set those benchmarks you want to use to True.
'''
BENCHMARK_C_NODE = True
BENCHMARK_C_NODE_NATIVE = True
BENCHMARK_CONCURRENT_ERL = True
BENCHMARK_DISTRIBUTED_ERL = True
BENCHMARK_CONCURRENT_NATIVE_ERL = True
BENCHMARK_DISTRIBUTED_NATIVE_ERL = True
BENCHMARK_NIFS = True

NUM_OF_RUNS = 5


if BENCHMARK_C_NODE:
    print("Benchmarking C Node:")
    chdir("./erl_cnode/")

    print("Compiling")
    system("gcc client_ann.c ei_utils.c fann_utils.c -o client_ann -lerl_interface -lei -lpthread -ldoublefann -lm")
    system("erlc fl.erl")

    for i in range(NUM_OF_RUNS):
        print("\nRun #" + str(i+1))
        system("erl -sname server -noshell -eval 'fl:main().' -eval 'init:stop().'")
    chdir("../")


if BENCHMARK_C_NODE_NATIVE:
    print("Benchmarking C Node:")
    chdir("./erl_cnode/")

    print("Compiling")
    system("gcc client_ann.c ei_utils.c fann_utils.c -o client_ann -lerl_interface -lei -lpthread -ldoublefann -lm")
    system("erlc +native fl.erl")

    for i in range(NUM_OF_RUNS):
        print("\nRun #" + str(i+1))
        system("erl -sname server -noshell -eval 'fl:main().' -eval 'init:stop().'")
    chdir("../")


if BENCHMARK_CONCURRENT_ERL:
    print("\n\nBenchmarking concurrent erlang:")
    chdir("./erl/")

    print("Compiling")
    system("erlc fl.erl")

    for i in range(NUM_OF_RUNS):
        print("\nRun #" + str(i+1))
        system("erl -sname server -noshell -eval 'fl:main().' -eval 'init:stop().'")
    chdir("../")


if BENCHMARK_DISTRIBUTED_ERL:
    print("\n\nBenchmarking distributed erlang:")
    chdir("./erl_dist/")

    print("Compiling")
    system("erlc fl.erl")

    for i in range(NUM_OF_RUNS):
        print("\nRun #" + str(i+1))
        system("erl -sname server -noshell -eval 'fl:main().' -eval 'init:stop().'")
        system("./exit.escript 10")
        time.sleep(1)
    chdir("../")


if BENCHMARK_CONCURRENT_NATIVE_ERL:
    print("\n\nBenchmarking concurrent erlang with native:")
    chdir("./erl/")

    print("Compiling")
    system("erlc +native fl.erl")

    for i in range(NUM_OF_RUNS):
        print("\nRun #" + str(i+1))
        system("erl -sname server -noshell -eval 'fl:main().' -eval 'init:stop().'")
    chdir("../")


if BENCHMARK_DISTRIBUTED_NATIVE_ERL:
    print("\n\nBenchmarking distributed erlang with native:")
    chdir("./erl_dist/")

    print("Compiling")
    system("erlc +native fl.erl")

    for i in range(NUM_OF_RUNS):
        print("\nRun #" + str(i+1))
        system("erl -sname server -noshell -eval 'fl:main().' -eval 'init:stop().'")
        system("./exit.escript 10")
        time.sleep(1)
    chdir("../")


if BENCHMARK_NIFS:
    print("\n\nBenchmarking erlang with NIFs:")
    chdir("./erl_nif/")

    print("Compiling")
    system("gcc -o ann_nif -fpic -shared ann_nif.c nif_utils.c fann_utils.c -ldoublefann -lm")
    system("erlc fl.erl")

    for i in range(NUM_OF_RUNS):
        print("\nRun #" + str(i+1))
        system("erl -noshell -eval 'fl:main().' -eval 'init:stop().'")
    chdir("../")


print("\nDone benchmarking.")
