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

    if (n->child != NULL) {
        somr_map_clear(n->child);
        free(n->child);
    }
}

void somr_node_init_weights(somr_node_t *n, double *weights, unsigned int features_count) {
    if (weights != NULL) {
        memcpy(n->weights, weights, sizeof(double) * features_count);
    } else {
        for (unsigned int i = 0; i < features_count; i++) {
            n->weights[i] = (double) rand() / (double) RAND_MAX;
        }
    }
}

void somr_node_activate(somr_node_t *n, somr_data_vector_t *data_vector, unsigned int features_count) {
    assert(features_count > 0);

    n->activation = somr_vector_euclid_dist(n->weights, data_vector->weights, features_count);
    // sqrt omitted on purpose, not need if we only use the activation value for comparison
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
    n->child = malloc(sizeof(somr_map_t));
    somr_map_init(n->child, features_count);
    //somr_map_init_nodes_weights(n->child, n->weights);
    // TODO
    somr_map_init_weights(n->child, NULL);
}

void somr_node_compute_error(somr_node_t *n, somr_dataset_t *dataset) {
    double sum = 0.0;
    unsigned int count = 0;

    for (unsigned int i = 0; i < dataset->size; i++) {
        somr_data_vector_t *data_vector = somr_dataset_get_vector(dataset, i);
        sum += somr_vector_euclid_dist(n->weights, data_vector->weights, dataset->features_count);
        count++;
    }

    assert(count > 0);
    assert(sum >= 0.0);
    n->error = sum;
}
