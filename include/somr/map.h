#pragma once
#include "data_vector.h"
#include "unit.h"
#include <stdio.h>

typedef unsigned int somr_unit_id_t;

/** Main structure for SOM map */
typedef struct somr_map_t {
    /** width of map */
    unsigned int width;
    /** height of map */
    unsigned int height;
    /** number of units in map (width * height, used to avoid recalculation) */
    unsigned int units_count;
    /** number of values per unit and per input vector */
    unsigned int features_count;
    /** flat array of units */
    somr_unit_t *units;
    double mean_error;
} somr_map_t;

void somr_map_init(somr_map_t *m, unsigned int features_count);
void somr_map_clear(somr_map_t *m);
void somr_map_init_random_weights(somr_map_t *m, unsigned int *rand_state);
void somr_map_activate(somr_map_t *m, somr_data_vector_t *data_vector);
/** @return first best matching unit found for @p data_vector */
somr_unit_id_t somr_map_find_bmu(somr_map_t *m, somr_data_vector_t *data_vector);
/**
fills @p[out] bmus with all equally-activated best matching units for @p vector
@pre @p bmus must be allocated with enough space (ie potentially the number of units in map)
*/
//void somr_map_find_bmus(somr_map_t *m, somr_data_vector_t *data_vector, somr_unit_id_t *bmus, unsigned int *bmu_count);
void somr_map_teach_nbhd(somr_map_t *m, somr_unit_id_t unit_id, somr_data_vector_t *data_vector, double learn_rate, double radius);
unsigned int somr_map_get_depth(somr_map_t *m);
/** maps input vector @p vector to a class, by returnig label of its best matching unit */
somr_label_t somr_map_classify(somr_map_t *m, somr_data_vector_t *data_vector);
//void somr_map_find_error_range(somr_map_t *, double *min_error, double *max_error);
void somr_map_write_to_img(somr_map_t *m, unsigned char *img, unsigned int img_width, unsigned int img_height, unsigned char *colors, unsigned int border);
