#pragma once

/** index of class assigned to a node or a vector */
typedef int somr_label_t;
/** label value for nodes with no labels */
#define SOMR_EMPTY_LABEL -1

typedef struct somr_data_vector_t {
    double *weights;
    somr_label_t label;
} somr_data_vector_t;

void somr_data_vector_init_batch(somr_data_vector_t *batch, unsigned int batch_size, unsigned int features_count);
void somr_data_vector_clear_batch(somr_data_vector_t *batch);
void somr_data_vector_normalize(somr_data_vector_t *v, unsigned int features_count);
