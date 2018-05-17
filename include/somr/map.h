#pragma once
#include "data_vector.h"
#include "node.h"

typedef unsigned int somr_node_id_t;

/** Main structure for SOM map */
typedef struct somr_map_t {
    /** width of map */
    unsigned int width;
    /** height of map */
    unsigned int height;
    /** number of nodes in map (width * height, used to avoid recalculation) */
    unsigned int nodes_count;
    /** number of values per node and per input vector */
    unsigned int features_count;
    /** flat array of nodes */
    somr_node_t *nodes;
    double mean_error;
} somr_map_t;

void somr_map_init(somr_map_t *n, unsigned int features_count);
void somr_map_clear(somr_map_t *n);
void somr_map_init_weights(somr_map_t *n, double *weights);
void somr_map_activate(somr_map_t *n, somr_data_vector_t *data_vector);
/** @return first best matching unit found for @p data_vector */
somr_node_id_t somr_map_find_bmu(somr_map_t *n, somr_data_vector_t *data_vector);
/**
fills @p[out] bmus with all equally-activated best matching units for @p vector
@pre @p bmus must be allocated with enough space (ie potentially the number of nodes in map)
*/
void somr_map_find_bmus(somr_map_t *n, somr_data_vector_t *data_vector, somr_node_id_t *bmus, unsigned int *bmu_count);
/** maps input vector @p vector to a class, by returnig label of its best matching unit */
somr_label_t somr_map_classify(somr_map_t *n, somr_data_vector_t *data_vector);
void somr_map_insert_row(somr_map_t *n, unsigned int row_before);
void somr_map_insert_col(somr_map_t *n, unsigned int col_before);
void somr_map_add_child(somr_map_t *n, somr_node_id_t somr_node_id);
void somr_map_print_topology(somr_map_t *n);
void somr_map_write_to_img(somr_map_t *n, unsigned char *img, unsigned int img_width, unsigned int img_height, unsigned char *colors);
