
#include "doublefann.h"
#include "nif_utils.h"
#include "fann_utils.h"

#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_NODES 2
#define HIDDEN_NODES 3
#define OUTPUT_NODES 2


void generate_data(int n, fann_type *input, fann_type *output);


static ERL_NIF_TERM train_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  // Handle input
  ERL_NIF_TERM weights = argv[0];
  int arity;
  const ERL_NIF_TERM *tuple;
  if(!enif_get_tuple(env, weights, &arity, &tuple) || arity != 3){
    fprintf(stderr, "Bad tuple.\n\r");
    return enif_make_badarg(env);
  }
  ERL_NIF_TERM erl_input_weights = tuple[0];
  ERL_NIF_TERM erl_bias_weights = tuple[1];
  ERL_NIF_TERM erl_input_bias_weights;
  if(!enif_get_list_cell(env, erl_bias_weights, &erl_input_bias_weights, &erl_bias_weights)){
    fprintf(stderr, "Bad list from input bias.\n\r");
    return enif_make_badarg(env);
  }
  ERL_NIF_TERM erl_hidden_bias_weights;
  if(!enif_get_list_cell(env, erl_bias_weights, &erl_hidden_bias_weights, &erl_bias_weights)){
    fprintf(stderr, "Bad list from hidden bias.\n\r");
    return enif_make_badarg(env);
  }
  ERL_NIF_TERM erl_hidden_weights = tuple[2];

  struct fann *ann = fann_create_standard(3, INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES);
  //fann_set_callback(ann, &test_callback);
  fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
  fann_set_learning_rate(ann, 0.01f);
  fann_set_train_error_function(ann, FANN_ERRORFUNC_LINEAR);
  fann_set_activation_function_hidden(ann, FANN_SIGMOID);
  // The derivitive will be wrong steepness is not set to 1 (fann issue)
  fann_set_activation_steepness_hidden(ann, 1.0f);
  fann_set_activation_function_output(ann, FANN_SIGMOID);
  fann_set_activation_steepness_output(ann, 1.0f);

  // decode weights
  double** input_weights = list2D_to_array2D(env, erl_input_weights);
  double** hidden_weights = list2D_to_array2D(env, erl_hidden_weights);

  double* input_bias_weights = list_to_array(env, erl_input_bias_weights);
  double* hidden_bias_weights = list_to_array(env, erl_hidden_bias_weights);

  // update weights to ann
  update_weights(ann, input_weights, hidden_weights
    , HIDDEN_NODES, INPUT_NODES
    , OUTPUT_NODES, HIDDEN_NODES
    , input_bias_weights, hidden_bias_weights);

  // train
  unsigned int input_len = 250;
  fann_type *input = malloc(sizeof(fann_type)*input_len*INPUT_NODES);
  fann_type *output = malloc(sizeof(fann_type)*input_len*OUTPUT_NODES);
  generate_data(input_len, input, output);

  struct fann_train_data *data = fann_create_train_array(input_len,
      INPUT_NODES, input, OUTPUT_NODES, output);

  const unsigned int n_epochs = 1;
  const unsigned int report_interval = 0;
  fann_train_on_data(ann, data, n_epochs, report_interval, 0);

  // extract weights from ann
  extract_weights(ann, input_weights, hidden_weights
            , HIDDEN_NODES, INPUT_NODES
            , OUTPUT_NODES, HIDDEN_NODES
            , input_bias_weights, hidden_bias_weights);

  // encode trained weights
  erl_input_weights = array2D_to_list2D(env, input_weights, HIDDEN_NODES, INPUT_NODES);
  erl_hidden_weights = array2D_to_list2D(env, hidden_weights, OUTPUT_NODES, HIDDEN_NODES);
  erl_input_bias_weights = array_to_list(env, input_bias_weights, HIDDEN_NODES);
  erl_hidden_bias_weights = array_to_list(env, hidden_bias_weights, OUTPUT_NODES);

  ERL_NIF_TERM biases = enif_make_list2(env, erl_input_bias_weights, erl_hidden_bias_weights);
  ERL_NIF_TERM result = enif_make_tuple3(env, erl_input_weights, biases, erl_hidden_weights);

  // free everything
  fann_destroy(ann);
  fann_destroy_train(data);
  for(int i = 0; i < HIDDEN_NODES; i++){
    free(input_weights[i]);
  }
  for(int i = 0; i < OUTPUT_NODES; i++){
    free(hidden_weights[i]);
  }
  free(input_weights); free(hidden_weights);
  free(input_bias_weights); free(hidden_bias_weights);
  free(input); free(output);

  return result;
}


fann_type random_float(){
  return rand()/(fann_type)RAND_MAX; // [0, 1.0]
}


void generate_data(int n, fann_type *input, fann_type *output){
  for (int i = 0; i < n*2; i+=2){
    fann_type x1 = random_float();
    fann_type x2 = random_float();
    fann_type t1 = sqrt(x1 * x2);
    fann_type t2 = sqrt(t1);

    input[i] = x1;
    input[i+1] = x2;
    output[i] = t1;
    output[i+1] = t2;
  }
}



static ErlNifFunc nif_funcs[] = {
  {"train", 1, train_nif},
};

ERL_NIF_INIT(fl, nif_funcs, NULL, NULL, NULL, NULL)
