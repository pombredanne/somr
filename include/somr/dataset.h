#pragma once
#include "data_vector.h"
#include "list.h"
#include <stdbool.h>
#include <stdio.h>

typedef struct somr_dataset_t {
    somr_data_vector_t *data_vectors;
    unsigned int size;
    unsigned int features_count;
    somr_list_t *class_list;
    /** shuffle indices used to acces input vectors in random order */
    unsigned int *indices;
    bool has_parent;
} somr_dataset_t;

void somr_dataset_init(somr_dataset_t *d, somr_data_vector_t *data_vectors, unsigned int *indices, unsigned int size, unsigned int features_count, somr_list_t *class_list);
void somr_dataset_init_from_parent(somr_dataset_t *d, somr_dataset_t *parent, unsigned int *indices, unsigned int size);
void somr_dataset_shuffle(somr_dataset_t *d, unsigned int *rand_state);
somr_data_vector_t *somr_dataset_get_vector(somr_dataset_t *t, unsigned int index);
char *somr_dataset_get_class(somr_dataset_t *d, somr_label_t label);
void somr_dataset_clear(somr_dataset_t *d);
void somr_dataset_compute_avg(somr_dataset_t *d, double *avg_weights);
void somr_dataset_init_from_file(somr_dataset_t *d, FILE *file, unsigned int size, unsigned int features_count);
void somr_dataset_normalize(somr_dataset_t *d);
