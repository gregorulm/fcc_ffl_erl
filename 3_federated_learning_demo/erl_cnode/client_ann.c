
/* client_ann.c */

#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>

#include "erl_interface.h"
#include "ei.h"
#include "ei_utils.h"

#include "doublefann.h"
#include "fann_utils.h"

#define BUFSIZE 1000

#define INPUT_NODES 2
#define HIDDEN_NODES 3
#define OUTPUT_NODES 2

int train_data_to_arrays(ETERM *data, double *input, double *output);
void generate_data(int n, fann_type *input, fann_type *output);

/**
* Takes a short name as an argument.
*/
int main(int argc, char **argv) {
  int fd;                                  /* File Descriptor to Erlang node */

  int loop = 1;                            /* Loop flag */
  int got;                                 /* Result of receive */
  unsigned char buf[BUFSIZE];              /* Buffer for incoming message */
  ErlMessage emsg;                         /* Incoming message */

  ETERM *fromp, *tuplep, *fnp, *argp1, *resp;
  int res;

  erl_init(NULL, 0);

  int connect_id = strtoimax(argv[2], NULL, 10); // TOOD: error handling

  if (erl_connect_init(connect_id, "secretcookie", 0) == -1)
    erl_err_quit("erl_connect_init");

  if ((fd = erl_connect(argv[1])) < 0){
    fprintf(stderr, "FD: %d\n", fd);
    erl_err_quit("erl_connect");
  }

  fprintf(stderr, "Connected.\n\r");

  struct fann *ann = fann_create_standard(3, INPUT_NODES, HIDDEN_NODES,
    OUTPUT_NODES);

  fann_set_callback(ann, &test_callback);
  fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
  fann_set_learning_rate(ann, 0.01f);
  fann_set_train_error_function(ann, FANN_ERRORFUNC_LINEAR);
  fann_set_activation_function_hidden(ann, FANN_SIGMOID);
  // The derivitive will be wrong steepness is not set to 1 (fann issue)
  fann_set_activation_steepness_hidden(ann, 1.0f);
  fann_set_activation_function_output(ann, FANN_SIGMOID);
  fann_set_activation_steepness_output(ann, 1.0f);

  srand(time(NULL));   // should only be called once

  while (loop) {

    got = erl_receive_msg(fd, buf, BUFSIZE, &emsg);
    if (got == ERL_TICK) {
      /* ignore */
    }
    else if (got == ERL_ERROR) {

      fprintf(stderr, "Error received: ERL_ERROR\n");
      loop = 0;
    }
    else {

      if (emsg.type == ERL_REG_SEND) {

        fromp = erl_element(2, emsg.msg);
        tuplep = erl_element(3, emsg.msg);
        fnp = erl_element(1, tuplep);
        argp1 = erl_element(2, tuplep);

        // Match the messages
        if (strncmp(ERL_ATOM_PTR(fnp), "assignment", 10) == 0) {
          ETERM *erl_input_weights = erl_element(1, argp1);
          ETERM *erl_biases = erl_element(2, argp1);
          ETERM *erl_input_bias_weights = ERL_CONS_HEAD(erl_biases);
          ETERM *erl_hidden_bias_weights = ERL_CONS_HEAD(ERL_CONS_TAIL(erl_biases));
          ETERM *erl_hidden_weights = erl_element(3, argp1);

          // decode weights
          double** input_weights = list2D_to_array2D(erl_input_weights);
          double** hidden_weights = list2D_to_array2D(erl_hidden_weights);

          double* input_bias_weights = list_to_array(erl_input_bias_weights);
          double* hidden_bias_weights = list_to_array(erl_hidden_bias_weights);

          update_weights(ann, input_weights, hidden_weights
            , HIDDEN_NODES, INPUT_NODES
            , OUTPUT_NODES, HIDDEN_NODES
            , input_bias_weights, hidden_bias_weights);

          // train
          int input_len = 250;
          fann_type *input = malloc(sizeof(fann_type)*input_len*INPUT_NODES);
          fann_type *output = malloc(sizeof(fann_type)*input_len*OUTPUT_NODES);
          generate_data(input_len, input, output);

          struct fann_train_data *data =
            fann_create_train_array(input_len,
              INPUT_NODES, input, OUTPUT_NODES, output);


          const unsigned int n_epochs = 1;
          const unsigned int report_interval = 0;
          fann_train_on_data(ann, data, n_epochs, report_interval, 0);

          extract_weights(ann, input_weights, hidden_weights
            , HIDDEN_NODES, INPUT_NODES
            , OUTPUT_NODES, HIDDEN_NODES
            , input_bias_weights, hidden_bias_weights);

          // encode trained weights
          erl_input_weights = array2D_to_list2D(input_weights, HIDDEN_NODES, INPUT_NODES);
          erl_hidden_weights = array2D_to_list2D(hidden_weights, OUTPUT_NODES, HIDDEN_NODES);
          erl_input_bias_weights = array_to_list(input_bias_weights, HIDDEN_NODES);
          erl_hidden_bias_weights = array_to_list(hidden_bias_weights, OUTPUT_NODES);
          ETERM *result = erl_format("{~w, [~w, ~w], ~w}",
                                    erl_input_weights,
                                    erl_input_bias_weights,
                                    erl_hidden_bias_weights,
                                    erl_hidden_weights);

          // send result back to server
          resp = erl_format("{update, ~w}", result);
          erl_send(fd, fromp, resp);

          // clean-up
          for(int i = 0; i < HIDDEN_NODES; i++){
            free(input_weights[i]);
          }
          for(int i = 0; i < OUTPUT_NODES; i++){
            free(hidden_weights[i]);
          }

          free(input_weights);
          free(hidden_weights);
          free(input_bias_weights);
          free(hidden_bias_weights);
          free(input);
          free(output);
          erl_free_compound(erl_input_weights);
          erl_free_compound(erl_hidden_weights);
          erl_free_compound(erl_input_bias_weights);
          erl_free_compound(erl_hidden_bias_weights);
          erl_free_compound(result);
          fann_destroy_train(data);
        } /* end if(fnp == "assingnment") */

        erl_free_term(emsg.from);
        erl_free_term(emsg.msg);
        erl_free_term(fromp);
        erl_free_compound(resp);
        erl_free_compound(tuplep);


        /* Check if Erlang is holding on to any memory
        long allocated, freed;

        erl_eterm_statistics(&allocated,&freed);
        fprintf(stderr, "currently allocated blocks: %ld\n",allocated);
        fprintf(stderr, "length of freelist: %ld\n",freed);

        erl_eterm_release(); */

      } /* end if(emsg.type == ERL_REG_SEND) */
    }
  } /* end while */
  fann_destroy(ann);
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