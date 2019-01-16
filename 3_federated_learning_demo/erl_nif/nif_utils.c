#include "nif_utils.h"


double *list_to_array(ErlNifEnv* env, ERL_NIF_TERM list){
  ERL_NIF_TERM elem;
  double c_elem;
  unsigned int n;

  if(!enif_get_list_length(env, list, &n)){
    return NULL;
  }

  double *arr =  n > 0 ? malloc(n*sizeof(double)) : NULL;

  int i = 0;
  while(enif_get_list_cell(env, list, &elem, &list)){
    // convert weights to C data-types
    if(!enif_get_double(env, elem, &c_elem)){
      free(arr);
      return NULL;
    }
    arr[i] = c_elem;
    i++;
  }
  return arr;
}


ERL_NIF_TERM array_to_list(ErlNifEnv* env, double *array, int size){
  ERL_NIF_TERM list[size];

  for(int i = 0; i < size; i++){
    list[i]  = enif_make_double(env, array[i]);
  }

  ERL_NIF_TERM erl_list = enif_make_list_from_array(env, list, size);

  return erl_list;
}


double **list2D_to_array2D(ErlNifEnv* env, ERL_NIF_TERM list2D){
  ERL_NIF_TERM inner_list;
  ERL_NIF_TERM elem;
  unsigned int n;

  if(!enif_get_list_length(env, list2D, &n)){
    return NULL;
  }

  double **arr = n > 0 ? malloc(n*sizeof(double)) : NULL;

  int i = 0;
  while(enif_get_list_cell(env, list2D, &inner_list, &list2D)){
    int inner_len;
    if(!enif_get_list_length(env, inner_list, &inner_len)){
      free(arr);
      return NULL;
    }

    arr[i] = list_to_array(env, inner_list);
    if(!arr[i]){
      free(arr);
      return NULL;
    }
    i++;
  }
  return arr;
}


ERL_NIF_TERM array2D_to_list2D(ErlNifEnv* env, double ** array2D, int n_rows, int n_cols){
  ERL_NIF_TERM list2D = enif_make_list(env, 0);
  ERL_NIF_TERM inner_list[n_cols];

  for(int i = n_rows-1; i >= 0 ; i--){
    for(int j = 0; j < n_cols; j++){
      inner_list[j]  = enif_make_double(env, array2D[i][j]);
    }
    ERL_NIF_TERM head = enif_make_list_from_array(env, inner_list, n_cols);

    list2D = enif_make_list_cell(env, head, list2D);
  }
  return list2D;
}


/**
 * Data should have the type of list of tuples of lists(size=2) => [{[i1, i2],[u1, u2]}]
**/
int train_data_to_arrays(ErlNifEnv* env, ERL_NIF_TERM data, double *input, double *output){
  unsigned int n;
  if(!enif_get_list_length(env, data, &n)){
    return 0;
  }

  ERL_NIF_TERM elem;
  int arity;
  const ERL_NIF_TERM *tuple;
  int i = 0;
  while(enif_get_list_cell(env, data, &elem, &data)){
    if(!enif_get_tuple(env, elem, &arity, &tuple) || arity != 2){ return 0; }
    ERL_NIF_TERM erl_input = tuple[0];
    ERL_NIF_TERM erl_output = tuple[1];

    ERL_NIF_TERM x1, x2;
    if(!enif_get_list_cell(env, erl_input, &x1, &erl_input)){ return 0; }
    if(!enif_get_list_cell(env, erl_input, &x2, &erl_input)){ return 0; }
    ERL_NIF_TERM y1, y2;
    if(!enif_get_list_cell(env, erl_output, &y1, &erl_output)){ return 0; }
    if(!enif_get_list_cell(env, erl_output, &y2, &erl_output)){ return 0; }

    if(!enif_get_double(env, x1, &input[i])){ return 0; }
    if(!enif_get_double(env, x2, &input[i+1])){ return 0; }
    if(!enif_get_double(env, y1, &output[i])){ return 0; }
    if(!enif_get_double(env, y2, &output[i+1])){ return 0; }
    i+=2;
  }
  return 1;
}
