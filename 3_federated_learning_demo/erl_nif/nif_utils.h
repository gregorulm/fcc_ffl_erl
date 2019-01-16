
#include <erl_nif.h>
#include <stdio.h>

/**
 * From Erl_NIF list to C array.
**/
double *list_to_array(ErlNifEnv* env, ERL_NIF_TERM list);

/**
* From C array to Erl_NIF list.
**/
ERL_NIF_TERM array_to_list(ErlNifEnv* env, double *array, int size);

/**
 *  convert an Erlang list of lists of Floats to a
 *  2D C-array of doubles.
**/
double **list2D_to_array2D(ErlNifEnv* env, ERL_NIF_TERM list2D);

/**
 *  convert a 2D C-array of doubles to an
 *  Erlang list of lists of Floats.
**/
ERL_NIF_TERM array2D_to_list2D(ErlNifEnv* env, double ** array2D, int n_rows, int n_cols);

/**
 * Data should have the type of list of tuples of lists(size=2) => [{[i1, i2],[u1, u2]}]
**/
int train_data_to_arrays(ErlNifEnv* env, ERL_NIF_TERM data, double *input, double *output);
