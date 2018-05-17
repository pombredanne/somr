#pragma once
#include "data_vector.h"
#include "dataset.h"

/** index of class assigned to a node or a vector */
typedef int somr_label_t;

typedef struct somr_map_t somr_map_t;

typedef struct somr_node_t {
    /** memory vector */
    double *weights;
    /** activation value for current input vector */
    double activation;
    double error;
    /** label assigned to node after training */
    somr_label_t label;
    somr_map_t *child;
} somr_node_t;

void somr_node_init(somr_node_t *n, unsigned int features_count);
void somr_node_init_weights(somr_node_t *n, double *weights, unsigned int features_count);
void somr_node_clear(somr_node_t *n);
/** activate node with euclidean distance from @p vector */
void somr_node_activate(somr_node_t *n, somr_data_vector_t *data_vector, unsigned int features_count);
/**
brings weights of node closer to values of input vector @p vector
@p learn: learing rate
*/
void somr_node_learn(somr_node_t *n, somr_data_vector_t *data_vector, unsigned int features_count, double learn_rate);
void somr_node_add_child(somr_node_t *n, unsigned int features_count);
void somr_node_compute_error(somr_node_t *n, somr_dataset_t *dataset);
