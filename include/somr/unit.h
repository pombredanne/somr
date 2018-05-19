#pragma once
#include "data_vector.h"
#include "dataset.h"

typedef struct somr_map_t somr_map_t;

typedef struct somr_unit_t {
    /** memory vector */
    double *weights;
    /** activation value for current input vector */
    double activation;
    double error;
    /** label assigned to unit after training */
    somr_label_t label;
    somr_map_t *child;
} somr_unit_t;

void somr_unit_init(somr_unit_t *n, unsigned int features_count);
void somr_unit_init_weights(somr_unit_t *n, double *weights, unsigned int features_count);
void somr_unit_init_random_weights(somr_unit_t *n, unsigned int *rand_state, unsigned int features_count);
void somr_unit_clear(somr_unit_t *n);
/** activate unit with euclidean distance from @p vector */
void somr_unit_activate(somr_unit_t *n, somr_data_vector_t *data_vector, unsigned int features_count);
/**
brings weights of unit closer to values of input vector @p vector
@p learn: learing rate
*/
void somr_unit_learn(somr_unit_t *n, somr_data_vector_t *data_vector, unsigned int features_count, double learn_rate);
void somr_unit_add_child(somr_unit_t *n, unsigned int features_count);
