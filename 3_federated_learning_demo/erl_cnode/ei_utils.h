
#ifndef _ERL_INTERFACE_H
#include "erl_interface.h"
#endif

#ifndef EI_H
#include "ei.h"
#endif


/**
 * create C-array from erlang list. The array has to be freed if returned.
 * NULL can be returned in three ways:
 * * input is not a proper list,
 * * input is the empty list,
 * * input list contains non-float elements
**/
double *list_to_array(ETERM* list);


/**
 *  convert an Erlang list of lists of Floats to a
 *  2D C-array of doubles.
**/
double **list2D_to_array2D(ETERM* list2D);


/**
 *  convert a 2D C-array of doubles to an
 *  Erlang list of lists of Floats.
**/
ETERM *array2D_to_list2D(double ** array2D, int n_rows, int n_cols);


/**
 * convert a C-array of doubles to an Erlang list
**/
ETERM *array_to_list(double *array, int size);


/**
 * returns an erlang tuple: {error, Reason}
**/
ETERM *make_error(const char *reason);
