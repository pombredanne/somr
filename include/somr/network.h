#pragma once
#include "dataset.h"
#include "list.h"
#include "map.h"
#include "node.h"

// wrapper for level 0 node and level 1 map (and class list)
typedef struct somr_network_t {
    somr_node_t root;
    somr_map_t map;
    somr_list_t class_list;
} somr_network_t;

void somr_network_init(somr_network_t *n, unsigned int features_count);
void somr_network_clear(somr_network_t *n);
void somr_network_train(somr_network_t *n, somr_dataset_t *dataset, double learn_rate, double tau_1, double tau_2, unsigned int lambda, unsigned int max_iters_count);
somr_label_t somr_network_classify(somr_network_t *n, somr_data_vector_t *data_vector);
char *somr_network_get_class(somr_network_t *n, somr_label_t label);
void somr_network_print_topology(somr_network_t *n);
void somr_network_write_to_img(somr_network_t *n, unsigned char *img, unsigned int img_width, unsigned int img_height);
