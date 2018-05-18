#include "data_vector.h"
#include "vector.h"
#include <assert.h>
#include <stdlib.h>

void somr_data_vector_init_batch(somr_data_vector_t *batch, unsigned int batch_size, unsigned int features_count) {
    assert(batch_size > 0);
    assert(features_count > 0);

    double *all_weights = malloc(sizeof(double) * features_count * batch_size);
    for (unsigned int i = 0; i < batch_size; i++) {
        batch[i].weights = &all_weights[i * features_count];
    }
}

void somr_data_vector_clear_batch(somr_data_vector_t *batch, unsigned int batch_size) {
    assert(batch_size > 0);
    free(batch[0].weights);
    for (unsigned int i = 0; i < batch_size; i++) {
        batch[i].weights = NULL;
    }
}

void somr_data_vector_normalize(somr_data_vector_t *v, unsigned int features_count) {
    assert(features_count > 0);
    somr_vector_normalize(v->weights, features_count);
}
