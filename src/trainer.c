#include "trainer.h"
#include "vector.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void somr_trainer_init(somr_trainer_t *t, somr_map_t *map, somr_dataset_t *dataset, double root_mean_error, double parent_mean_error, somr_trainer_settings_t *settings) {
    assert(map->features_count == dataset->features_count);
    assert(root_mean_error >= 0.0);
    assert(parent_mean_error >= 0);
    assert(settings->learn_rate > 0.0 && settings->learn_rate < 1.0);
    assert(settings->tau_1 >= 0.0 && settings->tau_1 <= 1.0);
    assert(settings->tau_2 >= 0.0 && settings->tau_2 <= 1.0);
    assert(settings->lambda > 0);
    assert(settings->max_iters_count > 0);
    //assert(dataset->size >= map->nodes_count);

    t->map = map;
    t->dataset = dataset;
    t->root_mean_error = root_mean_error;
    t->parent_mean_error = parent_mean_error;
    t->features_count = map->features_count;
    t->settings = settings;
}

void somr_trainer_clear(somr_trainer_t *t) {
    return;
}

void somr_trainer_train(somr_trainer_t *t) {
    double error_threshold = t->settings->tau_1 * t->parent_mean_error;

    for (unsigned int i = 0; i < t->settings->max_iters_count; i++) {
        // TODO check best radius formula
        //unsigned int radius = sqrt(t->map->width * t->map->height) / 2;
        //unsigned int radius = floor((sqrt(t->map->nodes_count / 2.0) - 1.0) / 2.0);
        double radius = sqrt(t->map->nodes_count) + 1;

        for (unsigned int j = 0; j < t->settings->lambda; j++) {
            // linear decay of learning factor and radius
            double decay = (double) j / (double) t->settings->lambda;

            double decayed_learn_rate = t->settings->learn_rate * (1.0 - decay);
            assert(decayed_learn_rate > 0.0 && decayed_learn_rate <= t->settings->learn_rate);

            // TODO make sure radius stays >= 1 ?
            // double decayed_radius = (double) radius - ((double) radius - 1.0) * decay;
            // assert(decayed_radius >= 1.0 && decayed_radius <= radius);
            double decayed_radius = (double) radius - ((double) radius) * decay;
            assert(decayed_radius > 0.0 && decayed_radius <= radius);

            somr_trainer_run_epoch(t, decayed_radius, decayed_learn_rate);
        }

        somr_node_id_t error_node_id = somr_trainer_compute_error(t);
        // loop until we stop expanding
        if (t->map->mean_error > error_threshold) {
            somr_trainer_expand(t, error_node_id);
        } else {
            break;
        }
    }

    somr_trainer_deepen(t);
    somr_trainer_label(t);
}

void somr_trainer_run_epoch(somr_trainer_t *t, double radius, double learn_rate) {
    assert(learn_rate > 0.0 && learn_rate < 1.0);
    assert(radius > 0.0);

    // pre-allocate array that will contain list of bmus
    somr_node_id_t *bmus = malloc(sizeof(somr_node_id_t) * t->map->nodes_count);
    // randomize data set
    somr_dataset_shuffle(t->dataset);

    // find bmu for each vector in data set and teach its neighborhood
    for (unsigned int i = 0; i < t->dataset->size; i++) {
        somr_data_vector_t *data_vector = somr_dataset_get_vector(t->dataset, i);

        // find all bmus (we may found several)
        unsigned int bmu_count;
        somr_map_find_bmus(t->map, data_vector, bmus, &bmu_count);
        assert(bmu_count > 0);

        // randomly pick one bmu if we have several candidates
        somr_node_id_t bmu_id;
        if (bmu_count > 1) {
            unsigned int index = rand() % bmu_count;
            bmu_id = bmus[index];
        } else {
            bmu_id = bmus[0];
        }
        bmu_id = somr_map_find_bmu(t->map, data_vector);

        // gaussian neighborhood
        double bmu_y = bmu_id / t->map->width;
        double bmu_x = bmu_id % t->map->width;
        // TODO avoid loop on all nodes ?
        for (unsigned int y = 0; y < t->map->height; y++) {
            unsigned int row_begin = y * t->map->width;
            double dist_y = bmu_y - y;
            double dist_y_square = dist_y * dist_y;
            for (unsigned int x = 0; x < t->map->width; x++) {
                somr_node_id_t somr_node_id = row_begin + x;
                double dist_x = bmu_x - x;
                // compute euclidean distance
                double dist = sqrt(dist_x * dist_x + dist_y_square);
                // compute gaussian attenuation factor
                double nbhd_factor = learn_rate * exp(-1.0 * (dist * dist) / (2.0 * radius * radius));
                if (nbhd_factor > 0.0) {
                    // teach node
                    somr_node_learn(&t->map->nodes[somr_node_id], data_vector, t->features_count, nbhd_factor);
                }
            }
        }
    }

    free(bmus);
}

somr_node_id_t somr_trainer_compute_error(somr_trainer_t *t) {
    // reset error for all nodes
    for (somr_node_id_t i = 0; i < t->map->nodes_count; i++) {
        t->map->nodes[i].error = 0.0;
    }

    for (unsigned int i = 0; i < t->dataset->size; i++) {
        somr_data_vector_t *data_vector = somr_dataset_get_vector(t->dataset, i);
        somr_node_id_t bmu_id = somr_map_find_bmu(t->map, data_vector);
        somr_node_t *bmu = &t->map->nodes[bmu_id];
        bmu->error += somr_vector_euclid_dist(bmu->weights, data_vector->weights, t->features_count);
        assert(bmu->error >= 0.0);
    }

    double sum = 0.0;
    unsigned int count = 0;
    double max_error = 0.0;
    somr_node_id_t error_node_id = t->map->nodes_count;

    for (somr_node_id_t i = 0; i < t->map->nodes_count; i++) {
        somr_node_t *node = &t->map->nodes[i];
        assert(node->error >= 0.0);
        sum += node->error;
        count++;

        if (node->error > max_error) {
            max_error = node->error;
            error_node_id = i;
        }
    }

    assert(count > 0);
    assert(sum >= 0.0);
    t->map->mean_error = sum / count;
    assert(t->map->mean_error >= 0.0);

    assert(error_node_id != t->map->nodes_count);
    return error_node_id;
}

void somr_trainer_expand(somr_trainer_t *t, somr_node_id_t error_node_id) {
    double *error_weights = t->map->nodes[error_node_id].weights;

    // find node coordinates
    int error_node_y = error_node_id / t->map->width;
    int error_node_x = error_node_id % t->map->width;

    double max_delta = -1.0;
    somr_node_id_t nb_ids[4];
    unsigned int nbs_count = 0;

    if (error_node_x > 0) {
        nb_ids[nbs_count] = error_node_id - 1;
        nbs_count++;
    }

    if (error_node_x < t->map->width - 1) {
        nb_ids[nbs_count] = error_node_id + 1;
        nbs_count++;
    }

    if (error_node_y > 0) {
        nb_ids[nbs_count] = (error_node_y - 1) * t->map->width + error_node_x;
        nbs_count++;
    }

    if (error_node_y < t->map->height - 1) {
        nb_ids[nbs_count] = (error_node_y + 1) * t->map->width + error_node_x;
        nbs_count++;
    }

    assert(nbs_count > 0);
    somr_node_id_t max_delta_nb_id = t->map->nodes_count;
    for (unsigned int i = 0; i < nbs_count; i++) {
        somr_node_id_t nb_id = nb_ids[i];
        assert(nb_id < t->map->nodes_count);

        double *nb_weights = t->map->nodes[nb_id].weights;
        double delta = somr_vector_euclid_dist(nb_weights, error_weights, t->features_count);
        if (delta > max_delta) {
            max_delta_nb_id = nb_id;
            max_delta = delta;
        }
    }

    assert(max_delta >= 0.0);
    assert(max_delta_nb_id < t->map->nodes_count);
    if (max_delta_nb_id == error_node_id + 1) {
        assert(error_node_x < t->map->width - 1);
        somr_map_insert_col(t->map, error_node_x);
    } else if (max_delta_nb_id == error_node_id - 1) {
        assert(error_node_x > 0);
        somr_map_insert_col(t->map, error_node_x - 1);
    } else if (max_delta_nb_id == error_node_id + t->map->width) {
        assert(error_node_y < t->map->height - 1);
        somr_map_insert_row(t->map, error_node_y);
    } else {
        assert(max_delta_nb_id == error_node_id - t->map->width);
        assert(error_node_y > 0);
        somr_map_insert_row(t->map, error_node_y - 1);
    }
}

// unsigned int depth = 0;
// unsigned int max_depth = 10000;

void somr_trainer_deepen(somr_trainer_t *t) {
    // if (depth >= max_depth) {
    //     return;
    // }
    // depth++;

    // pre-alloc
    unsigned int *data_vectors_indices = malloc(sizeof(unsigned int) * t->dataset->size);

    double error_threshold = t->root_mean_error * t->settings->tau_2;

    for (somr_node_id_t i = 0; i < t->map->nodes_count; i++) {
        somr_node_t *node = &t->map->nodes[i];
        if (node->error <= error_threshold) {
            continue;
        }

        // find data vectors for node
        unsigned int data_vectors_count = 0;
        for (unsigned int j = 0; j < t->dataset->size; j++) {
            somr_data_vector_t *data_vector = somr_dataset_get_vector(t->dataset, j);
            somr_node_id_t bmu_id = somr_map_find_bmu(t->map, data_vector);
            if (bmu_id == i) {
                data_vectors_indices[data_vectors_count] = j;
                data_vectors_count++;
            }
        }
        assert(data_vectors_count > 1);
        // TODO
        // if (data_vectors_count < 4) {
        //     continue;
        // }

        somr_map_add_child(t->map, i);

        somr_dataset_t child_dataset;
        somr_dataset_init_from_parent(&child_dataset, t->dataset, data_vectors_indices, data_vectors_count);
        somr_trainer_t child_trainer;
        somr_trainer_init(&child_trainer, node->child, &child_dataset, t->root_mean_error, node->error, t->settings);

        somr_trainer_train(&child_trainer);

        somr_dataset_clear(&child_dataset);
        somr_trainer_clear(&child_trainer);
    }

    free(data_vectors_indices);
}

void somr_trainer_label(somr_trainer_t *t) {
    // init with empty labels for all nodes
    for (somr_node_id_t i = 0; i < t->map->nodes_count; i++) {
        t->map->nodes[i].label = SOMR_EMPTY_LABEL;
    }

    // find bmu for each input vector and assign vector label to bmu
    for (unsigned int i = 0; i < t->dataset->size; i++) {
        somr_data_vector_t *data_vector = somr_dataset_get_vector(t->dataset, i);
        somr_node_id_t bmu_id = somr_map_find_bmu(t->map, data_vector);
        somr_node_t *bmu = &t->map->nodes[bmu_id];
        bmu->label = data_vector->label;
    }
}
