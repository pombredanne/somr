#define _GNU_SOURCE // for rand_r
#include "node.h"
#include "map.h"
#include "vector.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

void somr_node_init(somr_node_t *n, unsigned int features_count) {
    n->weights = malloc(sizeof(double) * features_count);
    n->label = SOMR_EMPTY_LABEL;
    n->child = NULL;
}

void somr_node_clear(somr_node_t *n) {
    free(n->weights);
    n->weights = NULL;

    if (n->child != NULL) {
        somr_map_clear(n->child);
        free(n->child);
        n->child = NULL;
    }
}

void somr_node_init_weights(somr_node_t *n, double *weights, unsigned int features_count) {
    memcpy(n->weights, weights, sizeof(double) * features_count);
}

void somr_node_init_random_weights(somr_node_t *n, unsigned int *rand_sate, unsigned int features_count) {
    for (unsigned int i = 0; i < features_count; i++) {
        n->weights[i] = (double) rand_r(rand_sate) / (double) RAND_MAX;
    }
}

void somr_node_activate(somr_node_t *n, somr_data_vector_t *data_vector, unsigned int features_count) {
    assert(features_count > 0);

    // sqrt omitted on purpose, not need if we only use the activation value for comparison
    n->activation = somr_vector_euclid_dist_squared(n->weights, data_vector->weights, features_count);
}

void somr_node_learn(somr_node_t *n, somr_data_vector_t *data_vector, unsigned int features_count, double learn_rate) {
    assert(features_count > 0);
    assert(learn_rate > 0.0);

    for (unsigned int i = 0; i < features_count; i++) {
        double delta = data_vector->weights[i] - n->weights[i];
        n->weights[i] += learn_rate * delta;
        assert(abs(data_vector->weights[i] - n->weights[i]) <= abs(delta));
    }
}

void somr_node_add_child(somr_node_t *n, unsigned int features_count) {
    assert(n->child == NULL);

    n->child = malloc(sizeof(somr_map_t));
    somr_map_init(n->child, features_count);
}
