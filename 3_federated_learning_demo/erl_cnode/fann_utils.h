
#ifndef __doublefann_h__
#include "doublefann.h"
#endif

#define NUM_IRIS_INPUTS 4
#define NUM_IRIS_OUTPUTS 3
#define NUM_IRIS_HIDDEN 7

#define NUM_MNIST_INPUTS (28*28)
#define NUM_MNIST_OUTPUTS 10
#define NUM_MNIST_HIDDEN 7

/**
 * Updates the weights of an ANN.
**/
int update_weights(struct fann *ann
    , double ** const input_weights
    , double ** const hidden_weights
    , const unsigned int input_rows
    , const unsigned int input_cols
    , const unsigned int hidden_rows
    , const unsigned int hidden_cols
    , double * const bias_weights_in
    , double * const bias_weights_hidden);

/**
 * Get all weights, excetp bias weigts, from an ANN
**/
int extract_weights(struct fann *ann
    , double ** input_weights
    , double ** hidden_weights
    , const unsigned int input_rows
    , const unsigned int input_cols
    , const unsigned int hidden_rows
    , const unsigned int hidden_cols
    , double * const bias_weights_in
    , double * const bias_weights_hidden);

/**
 * Same MSE as fann_test_data, but uses a different
 * method to detect bit failures.
*/
float fann_custom_test(struct fann *ann
    , struct fann_train_data *data
  );

/**
 * Prints all the expected output side by side
 * with the actual output.
**/
void test_vs_expected_output(struct fann *ann
    , struct fann_train_data *data
  );

/**
 * Reports the epoch number, MSE, and bit fail.
 * Normally used with testing as a customized epoch message.
**/
int FANN_API test_callback(struct fann *ann
    , struct fann_train_data *train
    , unsigned int max_epochs
    , unsigned int epochs_between_reports
    , float desired_error, unsigned int epochs
  );
