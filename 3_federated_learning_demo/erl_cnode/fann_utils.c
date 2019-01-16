
#include "fann_utils.h"
#include <math.h>


int update_weights(struct fann *ann
    , double ** const input_weights
    , double ** const hidden_weights
    , const unsigned int input_rows
    , const unsigned int input_cols
    , const unsigned int hidden_rows
    , const unsigned int hidden_cols
    , double * const bias_weights_in
    , double * const bias_weights_hidden){

  if( ann == NULL ){
    return 0;
  }
  unsigned int conns = fann_get_total_connections(ann);

  size_t conn_size = sizeof(struct fann_connection) * conns;
  struct fann_connection *net = malloc(conn_size);
  fann_get_connection_array(ann, net);

  for(int i= 0; i < input_rows; i++){
    for(int j = 0; j < input_cols; j++){
      net[i*(input_cols+1) + j].weight = input_weights[i][j];
    }
    net[i*(input_cols+1) + input_cols].weight = bias_weights_in[i]; //add bias weight.
  }

  int offset = input_rows * (input_cols +1);
  for(int i= 0; i < hidden_rows; i++){
    for(int j = 0; j < hidden_cols; j++){
      net[i*(hidden_cols+1) + j + offset].weight = hidden_weights[i][j];
    }
    net[i*(hidden_cols+1) + hidden_cols + offset].weight = bias_weights_hidden[i]; //add bias weight.
  }

  fann_set_weight_array(ann, net, conns);

  free(net);
  return 1;
}


int extract_weights(struct fann *ann
    , double ** input_weights
    , double ** hidden_weights
    , const unsigned int input_rows
    , const unsigned int input_cols
    , const unsigned int hidden_rows
    , const unsigned int hidden_cols
    , double * const bias_weights_in
    , double * const bias_weights_hidden){

  if( ann == NULL) {
    return 0;
  }
  unsigned int conns = fann_get_total_connections(ann);

  size_t conn_size = sizeof(struct fann_connection) * conns;
  struct fann_connection *net = malloc(conn_size);

  fann_get_connection_array(ann, net);

  for(int i= 0; i < input_rows; i++){
    for(int j = 0; j < input_cols; j++){
      input_weights[i][j] = net[i*(input_cols+1) + j].weight;
    }
    bias_weights_in[i] = net[i*(input_cols+1) + input_cols].weight;
  }

  int offset = input_rows * (input_cols +1);
  for(int i= 0; i < hidden_rows; i++){
    for(int j = 0; j < hidden_cols; j++){
      hidden_weights[i][j] = net[i*(hidden_cols+1) + j + offset].weight;
    }
    bias_weights_hidden[i] = net[i*(hidden_cols+1) + hidden_cols + offset].weight;
  }

  free(net);
  return 1;
}


/** This test detects a bit failure if the output neuron with the
 *  largest value has an expected output value of 1.
 *  Returns the MSE value (same as fann_test_data).
**/
float fann_custom_test(struct fann *ann, struct fann_train_data *data){
  if (data->num_output <= 0 || data->num_data <= 0 || data -> num_input == 0){
    return 0; // bad training data
  }

  const double epsilon = 0.01; // used for float comparisons
  int bitfail = 0;
  float squareError = 0;

  for (int i = 0; i < data->num_data; i++){
    unsigned int maxid = 0;
    fann_type *runRes = fann_run(ann, data->input[i]);

    squareError += powf(data->output[i][0] - runRes[0], 2);
    for (int j = 1; j < data->num_output; j++){
      squareError += powf(data->output[i][j] - runRes[j], 2);

      // save index of the largest output value
      fann_type output = runRes[j];
      maxid = output > runRes[maxid] ? j : maxid;
    }
    // compare max output to expected value
    bitfail += data->output[i][maxid] + epsilon >= 1.0 ? 0 : 1;
  }

  ann->num_MSE = data->num_data;
  ann->MSE_value = squareError / data->num_output;
  ann->num_bit_fail = bitfail;
  return fann_get_MSE(ann);
}


/* Print the result of running the data through the network followed
   by the expected results */
void test_vs_expected_output(struct fann *ann, struct fann_train_data *data){

  for (int i = 0; i < data->num_data; i++){
    fann_type *runRes = fann_run(ann, data->input[i]);

    fprintf(stderr, "Expected output:\r\n");
    for (int j = 0; j < data->num_output; j++){
      fprintf(stderr, "%f ", data->output[i][j]);
    }

    fprintf(stderr, "\r\nANN output:\r\n");
    for (int j = 0; j < data->num_output; j++){
      fprintf(stderr, "%f ", runRes[j]);
    }
    fprintf(stderr, "\r\n\r\n");
  }
}


int FANN_API test_callback(struct fann *ann, struct fann_train_data *train,
  unsigned int max_epochs, unsigned int epochs_between_reports,
  float desired_error, unsigned int epochs)
{
  fprintf(stderr, "Epochs %8d. MSE: %.5f. Bit fail: %d\r\n", epochs, fann_get_MSE(ann), fann_get_bit_fail(ann));
  return 0;
}
