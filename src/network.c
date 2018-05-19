#include "network.h"
#include "map_grow.h"
#include "trainer.h"
#include "vector.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

static void somr_network_compute_root_error(somr_network_t *n, somr_dataset_t *dataset);

void somr_network_init(somr_network_t *n, unsigned int features_count) {
    somr_unit_init(&n->root, features_count);
    somr_list_init(&n->class_list, true);
}

void somr_network_clear(somr_network_t *n) {
    somr_list_clear(&n->class_list);
    somr_unit_clear(&n->root);
}

void somr_network_train(somr_network_t *n, somr_dataset_t *dataset,
    double learn_rate, double spread_threshold, double depth_threshold, unsigned int iters_count, bool should_orient, unsigned int seed) {

    somr_list_clear(&n->class_list);
    somr_list_copy(&n->class_list, dataset->class_list);

    // assign data set mean to root unit
    somr_dataset_compute_mean_weights(dataset, n->root.weights);

    // compute error
    somr_network_compute_root_error(n, dataset);

    // init and run trainer with settings
    somr_trainer_settings_t settings = {
        learn_rate,
        spread_threshold,
        depth_threshold,
        iters_count,
        should_orient,
        seed
    };

    somr_unit_add_child(&n->root, dataset->features_count);
    somr_map_init_random_weights(n->root.child, &settings.rand_state);

    somr_trainer_t trainer;
    somr_trainer_init(&trainer, n->root.child, dataset, n->root.error, n->root.error, &settings);
    somr_trainer_train(&trainer);
}

static void somr_network_compute_root_error(somr_network_t *n, somr_dataset_t *dataset) {
    n->root.error = 0.0;
    for (unsigned int i = 0; i < dataset->size; i++) {
        somr_data_vector_t *data_vector = somr_dataset_get_vector(dataset, i);
        n->root.error += somr_vector_euclid_dist(n->root.weights, data_vector->weights, dataset->features_count);
    }
    assert(n->root.error >= 0.0);
}

somr_label_t somr_network_classify(somr_network_t *n, somr_data_vector_t *data_vector) {
    return somr_map_classify(n->root.child, data_vector);
}

char *somr_network_get_class(somr_network_t *n, somr_label_t label) {
    if (label == SOMR_EMPTY_LABEL) {
        return NULL;
    }
    assert(label < n->class_list.size);
    return somr_list_get(&n->class_list, label);
}

void somr_network_write_to_img(somr_network_t *n, unsigned char *img, unsigned int img_width, unsigned int img_height, unsigned char *colors) {
    unsigned int border = somr_map_get_depth(n->root.child) * 2;
    somr_map_write_to_img(n->root.child, img, img_width, img_height, colors, border);
}
