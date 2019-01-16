
#include "ei_utils.h"
#include <stdlib.h>


double *list_to_array(ETERM* list){
  int n = erl_length(list);
  if (n == -1){ // assert that we have a proper list
    return NULL;
  }

  double *arr = malloc(sizeof(double)*n);

  for (int i = 0; i < n; i++){
    ETERM *hd = ERL_CONS_HEAD(list);
    if (!ERL_IS_FLOAT(hd)){
      free(arr);
      return NULL;
    }
    arr[i] = ERL_FLOAT_VALUE(hd);
    list = ERL_CONS_TAIL(list);
  }
  return arr;
}


ETERM *array2D_to_list2D(double ** array2D, int n_rows, int n_cols){
  ETERM *list2D = erl_mk_empty_list();
  ETERM *inner_list[n_cols];

  for(int i = n_rows-1; i >= 0 ; i--){
    for(int j = 0; j < n_cols; j++){
      inner_list[j]  = erl_mk_float(array2D[i][j]);
    }
    ETERM *head = erl_mk_list(inner_list, n_cols);

    list2D = erl_cons(head, list2D);
  }
  return list2D;
}

ETERM *array_to_list(double *array, int size){
  ETERM *list[size];
  for(int i = 0; i < size; i++){
    list[i]  = erl_mk_float(array[i]);
  }

  ETERM *erl_list = erl_mk_list(list, size);
  return erl_list;
}


double **list2D_to_array2D(ETERM* list2D){
  int n = erl_length(list2D);
  if (n == -1){ // assert that we have a proper list
    return NULL;
  }

  double **arr = malloc(sizeof(double*)*n);

  for (int i = 0; i < n; i++){
    ETERM *hd = ERL_CONS_HEAD(list2D);
    int inner_n = erl_length(hd);
    if (inner_n == -1){
      free(arr);
      return NULL;
    }
    arr[i] = list_to_array(hd);
    list2D = ERL_CONS_TAIL(list2D);
  }
  return arr;
}


ETERM *make_error(const char *reason){
  ETERM *errTuple[] = { erl_mk_atom("error")
  , erl_mk_string(reason) };
  ETERM *resp = erl_mk_tuple(errTuple, 2);
  erl_free_array(errTuple, 2);

  return resp;
}
