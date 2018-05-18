#include "network.h"
#include "map_grow.h"
#include "trainer.h"
#include "vector.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

static void somr_network_compute_root_error(somr_network_t *n, somr_dataset_t *dataset);

void somr_network_init(somr_network_t *n, unsigned int features_count) {
    // init root node
    somr_node_init(&n->root, features_count);

    // init map
    somr_map_init(&n->map, features_count);

    somr_list_init(&n->class_list, true);
}

void somr_network_clear(somr_network_t *n) {
    somr_list_clear(&n->class_list);
    somr_map_clear(&n->map);
    somr_node_clear(&n->root);
}

void somr_network_train(somr_network_t *n, somr_dataset_t *dataset,
    double learn_rate, double tau_1, double tau_2, unsigned int lambda, bool should_orient, unsigned int seed) {
    somr_list_clear(&n->class_list);
    somr_list_copy(&n->class_list, dataset->class_list);

    // assign data set mean/avg to root node
    somr_dataset_compute_avg(dataset, n->root.weights);

    for (int j = 0; j < n->map.features_count; j++) {
        assert(n->root.weights[j] >= 0.0 && n->root.weights[j] <= 1.0);
    }

    // compute error
    somr_network_compute_root_error(n, dataset);

    // init and run trainer with settings
    somr_trainer_settings_t settings = {
        learn_rate,
        tau_1,
        tau_2,
        lambda,
        should_orient,
        seed
    };

    // TODO check
    somr_map_init_random_weights(&n->map, &settings.rand_state);

    somr_trainer_t trainer;
    somr_trainer_init(&trainer, &n->map, dataset, n->root.error, n->root.error, &settings);
    somr_trainer_train(&trainer);
}

static void somr_network_compute_root_error(somr_network_t *n, somr_dataset_t *dataset) {
    double sum = 0.0;
    for (unsigned int i = 0; i < dataset->size; i++) {
        somr_data_vector_t *data_vector = somr_dataset_get_vector(dataset, i);
        sum += somr_vector_euclid_dist(n->root.weights, data_vector->weights, dataset->features_count);
    }
    assert(sum >= 0.0);
    n->root.error = sum;
}

somr_label_t somr_network_classify(somr_network_t *n, somr_data_vector_t *data_vector) {
    return somr_map_classify(&n->map, data_vector);
}

char *somr_network_get_class(somr_network_t *n, somr_label_t label) {
    if (label == SOMR_EMPTY_LABEL) {
        return NULL;
    }
    assert(label < n->class_list.size);
    return somr_list_get(&n->class_list, label);
}

void somr_network_write_to_img(somr_network_t *n, unsigned char *img, unsigned int img_width, unsigned int img_height, unsigned char *colors) {
    unsigned int border = somr_map_get_depth(&n->map) * 2;
    somr_map_write_to_img(&n->map, img, img_width, img_height, colors, border);
}
