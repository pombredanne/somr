#define _GNU_SOURCE // for rand_r
#include "trainer.h"
#include "map_grow.h"
#include "vector.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static somr_unit_id_t somr_trainer_compute_error(somr_trainer_t *t);
static void somr_trainer_spread(somr_trainer_t *t, somr_unit_id_t error_unit_id);
static void somr_trainer_deepen(somr_trainer_t *t);
static void somr_trainer_run_epoch(somr_trainer_t *t, double radius, double learn_rate);

void somr_trainer_init(somr_trainer_t *t, somr_map_t *map, somr_dataset_t *dataset, double root_mean_error, double parent_mean_error, somr_trainer_settings_t *settings) {
    assert(map->features_count == dataset->features_count);
    assert(root_mean_error >= 0.0);
    assert(parent_mean_error >= 0);
    assert(settings->learn_rate > 0.0 && settings->learn_rate < 1.0);
    assert(settings->spread_threshold >= 0.0 && settings->spread_threshold <= 1.0);
    assert(settings->depth_threshold >= 0.0 && settings->depth_threshold <= 1.0);
    assert(settings->iters_count > 0);
    //assert(dataset->size >= map->units_count);

    t->map = map;
    t->dataset = dataset;
    t->root_mean_error = root_mean_error;
    t->parent_mean_error = parent_mean_error;
    t->features_count = map->features_count;
    t->settings = settings;
}

void somr_trainer_train(somr_trainer_t *t) {
    double error_threshold = t->settings->spread_threshold * t->parent_mean_error;

    while (true) {
        // TODO check best radius formula
        double radius = sqrt(t->map->units_count) / 2;
        //double radius = floor((sqrt(t->map->units_count / 2.0) - 1.0) / 2.0);
        //double radius = sqrt(t->map->units_count) + 1;
        for (unsigned int i = 0; i < t->settings->iters_count; i++) {
            // linear decay of learning factor and radius
            double decay = (double) i / (double) t->settings->iters_count;

            double decayed_learn_rate = t->settings->learn_rate * (1.0 - decay);
            assert(decayed_learn_rate > 0.0 && decayed_learn_rate <= t->settings->learn_rate);

            // TODO make sure radius stays >= 1 ?
            // double decayed_radius = (double) radius - ((double) radius - 1.0) * decay;
            // assert(decayed_radius >= 1.0 && decayed_radius <= radius);
            double decayed_radius = (double) radius - ((double) radius) * decay;
            assert(decayed_radius > 0.0 && decayed_radius <= radius);

            somr_trainer_run_epoch(t, decayed_radius, decayed_learn_rate);
        }

        somr_unit_id_t error_unit_id = somr_trainer_compute_error(t);
        // loop until we stop spreading
        if (t->map->mean_error > error_threshold) {
            somr_trainer_spread(t, error_unit_id);
        } else {
            break;
        }
    }

    somr_trainer_deepen(t);
    somr_trainer_label(t);
}

static void somr_trainer_run_epoch(somr_trainer_t *t, double radius, double learn_rate) {
    assert(learn_rate > 0.0 && learn_rate < 1.0);
    assert(radius > 0.0);

    // pre-allocate array that will contain list of bmus
    // somr_unit_id_t *bmus = malloc(sizeof(somr_unit_id_t) * t->map->units_count);

    // randomize data set
    somr_dataset_shuffle(t->dataset, &t->settings->rand_state);

    // find bmu for each vector in data set and teach its neighborhood
    for (unsigned int i = 0; i < t->dataset->size; i++) {
        somr_data_vector_t *data_vector = somr_dataset_get_vector(t->dataset, i);

        // TODO randomly pick one if several found, is this useful?
        // // find all bmus (we may found several)
        // unsigned int bmu_count;
        // somr_map_find_bmus(t->map, data_vector, bmus, &bmu_count);
        // assert(bmu_count > 0);

        // // randomly pick one bmu if we have several candidates
        // somr_unit_id_t bmu_id;
        // if (bmu_count > 1) {;
        //     unsigned int index = rand_r(&t->settings->rand_state) % bmu_count;
        //     bmu_id = bmus[index];
        // } else {
        //     bmu_id = bmus[0];
        // }

        somr_unit_id_t bmu_id = somr_map_find_bmu(t->map, data_vector);
        somr_map_teach_nbhd(t->map, bmu_id, data_vector, learn_rate, radius);
    }

    // free(bmus);
}

static somr_unit_id_t somr_trainer_compute_error(somr_trainer_t *t) {
    // reset error for all units
    for (somr_unit_id_t i = 0; i < t->map->units_count; i++) {
        t->map->units[i].error = 0.0;
    }

    // find bmu for each data vector and add weights delta to error
    for (unsigned int i = 0; i < t->dataset->size; i++) {
        somr_data_vector_t *data_vector = somr_dataset_get_vector(t->dataset, i);
        somr_unit_id_t bmu_id = somr_map_find_bmu(t->map, data_vector);
        somr_unit_t *bmu = &t->map->units[bmu_id];
        bmu->error += somr_vector_euclid_dist(bmu->weights, data_vector->weights, t->features_count);
        assert(bmu->error >= 0.0);
    }

    // compute mean error of map, and locate unit with max error
    double sum = 0.0;
    unsigned int count = 0;
    double max_error = 0.0;
    somr_unit_id_t error_unit_id = t->map->units_count;

    for (somr_unit_id_t i = 0; i < t->map->units_count; i++) {
        somr_unit_t *unit = &t->map->units[i];
        assert(unit->error >= 0.0);
        sum += unit->error;
        count++;

        // TODO randomly pick one if several with same error?
        if (unit->error > max_error) {
            max_error = unit->error;
            error_unit_id = i;
        }
    }

    assert(count > 0);
    assert(sum >= 0.0);
    t->map->mean_error = sum / count;

    assert(error_unit_id != t->map->units_count);
    return error_unit_id;
}

static void somr_trainer_spread(somr_trainer_t *t, somr_unit_id_t error_unit_id) {
    double *error_weights = t->map->units[error_unit_id].weights;

    int error_unit_y = error_unit_id / t->map->width;
    int error_unit_x = error_unit_id % t->map->width;

    somr_unit_id_t nb_ids[4];
    unsigned int nbs_count = 0;

    // locate available immediate neighbors (UP/DOWN/LEFT/RIGHT) of error unit
    if (error_unit_x > 0) {
        nb_ids[nbs_count] = error_unit_id - 1;
        nbs_count++;
    }
    if (error_unit_x < t->map->width - 1) {
        nb_ids[nbs_count] = error_unit_id + 1;
        nbs_count++;
    }
    if (error_unit_y > 0) {
        nb_ids[nbs_count] = (error_unit_y - 1) * t->map->width + error_unit_x;
        nbs_count++;
    }
    if (error_unit_y < t->map->height - 1) {
        nb_ids[nbs_count] = (error_unit_y + 1) * t->map->width + error_unit_x;
        nbs_count++;
    }

    // find neighbor with biggest weights delta
    // TODO randomly pick one if several with same delta ?
    double max_delta = -1.0;
    assert(nbs_count > 0);
    somr_unit_id_t max_delta_nb_id = t->map->units_count;
    for (unsigned int i = 0; i < nbs_count; i++) {
        somr_unit_id_t nb_id = nb_ids[i];
        assert(nb_id < t->map->units_count);

        double *nb_weights = t->map->units[nb_id].weights;
        double delta = somr_vector_euclid_dist_squared(nb_weights, error_weights, t->features_count);
        if (delta > max_delta) {
            max_delta_nb_id = nb_id;
            max_delta = delta;
        }
    }

    assert(max_delta >= 0.0);
    assert(max_delta_nb_id < t->map->units_count);

    // insert row or column between error unit and max delta neighbor
    if (max_delta_nb_id == error_unit_id + 1) {
        assert(error_unit_x < t->map->width - 1);
        somr_map_insert_col(t->map, error_unit_x);
    } else if (max_delta_nb_id == error_unit_id - 1) {
        assert(error_unit_x > 0);
        somr_map_insert_col(t->map, error_unit_x - 1);
    } else if (max_delta_nb_id == error_unit_id + t->map->width) {
        assert(error_unit_y < t->map->height - 1);
        somr_map_insert_row(t->map, error_unit_y);
    } else {
        assert(max_delta_nb_id == error_unit_id - t->map->width);
        assert(error_unit_y > 0);
        somr_map_insert_row(t->map, error_unit_y - 1);
    }
}

static void somr_trainer_deepen(somr_trainer_t *t) {
    // pre-alloc
    unsigned int *data_vectors_indices = malloc(sizeof(unsigned int) * t->dataset->size);

    double error_threshold = t->root_mean_error * t->settings->depth_threshold;

    for (somr_unit_id_t i = 0; i < t->map->units_count; i++) {
        somr_unit_t *unit = &t->map->units[i];
        if (unit->error <= error_threshold) {
            continue;
        }

        // find data vectors for unit
        unsigned int data_vectors_count = 0;
        for (unsigned int j = 0; j < t->dataset->size; j++) {
            somr_data_vector_t *data_vector = somr_dataset_get_vector(t->dataset, j);
            somr_unit_id_t bmu_id = somr_map_find_bmu(t->map, data_vector);
            if (bmu_id == i) {
                data_vectors_indices[data_vectors_count] = j;
                data_vectors_count++;
            }
        }
        assert(data_vectors_count > 1);
        // TODO check
        // if (data_vectors_count < 4) {
        //     continue;
        // }

        somr_map_add_child(t->map, i, t->settings->should_orient, &t->settings->rand_state);

        somr_dataset_t child_dataset;
        somr_dataset_init_from_parent(&child_dataset, t->dataset, data_vectors_indices, data_vectors_count);
        somr_trainer_t child_trainer;
        somr_trainer_init(&child_trainer, unit->child, &child_dataset, t->root_mean_error, unit->error, t->settings);

        somr_trainer_train(&child_trainer);

        somr_dataset_clear(&child_dataset);
    }

    free(data_vectors_indices);
}

void somr_trainer_label(somr_trainer_t *t) {
    // init with empty labels for all units
    for (somr_unit_id_t i = 0; i < t->map->units_count; i++) {
        t->map->units[i].label = SOMR_EMPTY_LABEL;
    }

    // randomize data set
    somr_dataset_shuffle(t->dataset, &t->settings->rand_state);

    // find bmu for each input vector and assign vector label to bmu
    for (unsigned int i = 0; i < t->dataset->size; i++) {
        somr_data_vector_t *data_vector = somr_dataset_get_vector(t->dataset, i);
        somr_unit_id_t bmu_id = somr_map_find_bmu(t->map, data_vector);
        somr_unit_t *bmu = &t->map->units[bmu_id];
        bmu->label = data_vector->label;
    }
}
