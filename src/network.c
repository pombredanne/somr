#include "network.h"
#include "trainer.h"
#include <assert.h>

void somr_network_init(somr_network_t *n, unsigned int features_count) {
    // init root node
    somr_node_init(&n->root, features_count);

    // init map
    somr_map_init(&n->map, features_count);
    somr_map_init_weights(&n->map, NULL);
}

void somr_network_clear(somr_network_t *n) {
    somr_list_clear(&n->class_list);
    somr_map_clear(&n->map);
    somr_node_clear(&n->root);
}

void somr_network_train(somr_network_t *n, somr_dataset_t *dataset, double learn_rate, double tau_1, double tau_2, unsigned int lambda, unsigned int max_iters_count) {
    somr_list_copy(&n->class_list, dataset->class_list);

    // assign data set mean/avg to root node
    somr_dataset_compute_avg(dataset, n->root.weights);

    for (int j = 0; j < n->map.features_count; j++) {
        assert(n->root.weights[j] >= 0.0 && n->root.weights[j] <= 1.0);
    }

    // compute error
    somr_node_compute_error(&n->root, dataset);

    // init trainer
    somr_trainer_settings_t somr_trainer_settings = { learn_rate, tau_1, tau_2, lambda, max_iters_count };
    somr_trainer_t trainer;
    somr_trainer_init(&trainer, &n->map, dataset, n->root.error, n->root.error, &somr_trainer_settings);

    somr_trainer_train(&trainer);

    somr_trainer_clear(&trainer);
}

void somr_network_print_topology(somr_network_t *n) {
    somr_map_print_topology(&n->map);
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

void somr_network_write_to_img(somr_network_t *n, unsigned char *img, unsigned int img_width, unsigned int img_height) {
    unsigned char label_colors[128];
    label_colors[0] = 255;
    label_colors[1] = 0;
    label_colors[2] = 0;

    label_colors[3] = 0;
    label_colors[4] = 255;
    label_colors[5] = 0;

    label_colors[6] = 0;
    label_colors[7] = 0;
    label_colors[8] = 255;

    somr_map_write_to_img(&n->map, img, img_width, img_height, label_colors);
}
