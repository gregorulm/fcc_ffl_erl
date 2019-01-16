#include "nif_utils.h"

/**
 * Invariant test: 
 *   Convert a 2D list to an array and back to a list == Identity
 * I.e, this should be equivalent to the Identity function.
**/
static ERL_NIF_TERM test_2d_list_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){

  ERL_NIF_TERM list2D = argv[0];
  ERL_NIF_TERM n_r = argv[1];
  ERL_NIF_TERM n_c = argv[2];

  int n_rows, n_cols;
  if (!enif_get_int(env, argv[1], &n_rows)) {
    return enif_make_badarg(env);
  }
  if (!enif_get_int(env, argv[2], &n_cols)) {
    return enif_make_badarg(env);
  }

  double **arr = list2D_to_array2D(env, list2D);
  if(arr == NULL){
    return enif_make_list(env, 0);
  }

  list2D = array2D_to_list2D(env, arr, n_rows, n_cols);
  for(int i = 0; i < n_rows; i++){
    free(arr[i]);
  }
  free(arr);
  return list2D;
}

static ErlNifFunc nif_funcs[] = {
    {"test_2d_list", 3, test_2d_list_nif}
};

ERL_NIF_INIT(tests, nif_funcs, NULL, NULL, NULL, NULL)
