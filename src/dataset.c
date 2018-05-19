#define _GNU_SOURCE // for strtok_r and rand_r
#include "dataset.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

void somr_dataset_init(somr_dataset_t *d, somr_data_vector_t *data_vectors, unsigned int *indices, unsigned int size, unsigned int features_count, somr_list_t *class_list) {
    assert(size > 0);
    assert(features_count > 0);

    d->data_vectors = data_vectors;
    d->size = size;
    d->features_count = features_count;
    d->class_list = malloc(sizeof(somr_list_t));
    somr_list_copy(d->class_list, class_list);
    d->indices = malloc(sizeof(unsigned int) * d->size);
    memcpy(d->indices, indices, sizeof(unsigned int) * d->size);
    d->has_parent = false;
}

void somr_dataset_init_from_parent(somr_dataset_t *d, somr_dataset_t *parent, unsigned int *indices, unsigned int size) {
    assert(size > 0);

    d->data_vectors = parent->data_vectors;
    d->size = size;
    d->features_count = parent->features_count;
    d->class_list = parent->class_list;
    d->indices = malloc(sizeof(unsigned int) * d->size);
    for (unsigned int i = 0; i < size; i++) {
        d->indices[i] = parent->indices[indices[i]];
    }
    d->has_parent = true;
}

void somr_dataset_clear(somr_dataset_t *d) {
    free(d->indices);
    d->indices = NULL;
    if (!d->has_parent) {
        somr_data_vector_clear_batch(d->data_vectors, d->size);
        free(d->data_vectors);
        d->data_vectors = NULL;
        somr_list_clear(d->class_list);
        free(d->class_list);
        d->class_list = NULL;
    }
}

somr_data_vector_t *somr_dataset_get_vector(somr_dataset_t *d, unsigned int index) {
    assert(index < d->size);
    unsigned int real_index = d->indices[index];
    return &d->data_vectors[real_index];
}

char *somr_dataset_get_class(somr_dataset_t *d, somr_label_t label) {
    if (label == SOMR_EMPTY_LABEL) {
        return NULL;
    }
    assert(label < d->class_list->size);
    return somr_list_get(d->class_list, label);
}

void somr_dataset_shuffle(somr_dataset_t *d, unsigned int *rand_state) {
    for (int i = 0; i < d->size; i++) {
        unsigned int index = rand_r(rand_state) % d->size;
        unsigned int swap = d->indices[i];
        d->indices[i] = d->indices[index];
        d->indices[index] = swap;
    }
}

void somr_dataset_normalize(somr_dataset_t *d) {
    for (unsigned int i = 0; i < d->size; i++) {
        unsigned int index = d->indices[i];
        somr_data_vector_normalize(&d->data_vectors[index], d->features_count);
    }
}

void somr_dataset_compute_mean_weights(somr_dataset_t *d, double *mean_weights) {
    for (unsigned int i = 0; i < d->features_count; i++) {
        mean_weights[i] = 0.0;
    }

    // sum all vectors
    for (unsigned int i = 0; i < d->size; i++) {
        unsigned int real_index = d->indices[i];
        somr_data_vector_t *data_vector = &d->data_vectors[real_index];
        for (unsigned int j = 0; j < d->features_count; j++) {
            mean_weights[j] += data_vector->weights[j];
        }
    }

    // get average for each feature
    for (unsigned int i = 0; i < d->features_count; i++) {
        mean_weights[i] /= d->size;
        assert(mean_weights[i] >= 0.0 && mean_weights[i] <= 1.0);
    }
}

void somr_dataset_init_from_file(somr_dataset_t *d, FILE *file, unsigned int size, unsigned int features_count) {
    somr_data_vector_t *data_vectors = malloc(sizeof(somr_data_vector_t) * size);
    somr_data_vector_init_batch(data_vectors, size, features_count);

    somr_list_t class_list;
    somr_list_init(&class_list, true);

    // add new line to delimiters
    char *delims = ",\n";
    size_t max_line_length = 256;
    char *line = malloc(sizeof(char) * max_line_length);

    for (unsigned int i = 0; i < size; i++) {
        int line_length = getline(&line, &max_line_length, file);
        if (line_length == -1) {
            fprintf(stderr, "Error reading input data vector\n");
            exit(EXIT_FAILURE);
        }

        // pointer to memory for vector being read
        somr_data_vector_t *data_vector = &data_vectors[i];
        char *strtok_save;

        // read label
        // first call to strtok: feed line
        char *token = strtok_r(line, delims, &strtok_save);
        if (token == NULL) {
            fprintf(stderr, "Error reading input data label\n");
            exit(EXIT_FAILURE);
        }

        // add class if new and label vector with class index
        unsigned int class_index;
        if (!somr_list_find(&class_list, token, &class_index)) {
            class_index = class_list.size;
            somr_list_push(&class_list, token);
        }
        data_vector->label = class_index;

        // read weights
        for (unsigned int j = 0; j < features_count; j++) {
            char *token = strtok_r(NULL, delims, &strtok_save);
            if (token == NULL) {
                fprintf(stderr, "Error reading input data feature\n");
                exit(EXIT_FAILURE);
            }

            data_vector->weights[j] = atof(token);
        }
    }

    free(line);

    // init array containing indices used to acces input vectors in random order
    unsigned int *indices = malloc(sizeof(unsigned int) * size);
    for (unsigned int i = 0; i < size; i++) {
        indices[i] = i;
    }
    somr_dataset_init(d, data_vectors, indices, size, features_count, &class_list);
    somr_list_clear(&class_list);
    free(indices);
}
