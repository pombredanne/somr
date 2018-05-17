#pragma once
#include "dataset.h"
#include "map.h"

typedef struct somr_trainer_settings_t {
    double learn_rate;
    double tau_1;
    double tau_2;
    unsigned int lambda;
    unsigned int max_iters_count;
} somr_trainer_settings_t;

/** Structure responsible of the training of a SOM network */
typedef struct somr_trainer_t {
    /** network to train */
    somr_map_t *map;
    somr_dataset_t *dataset;
    unsigned int features_count;
    double root_mean_error;
    double parent_mean_error;
    somr_trainer_settings_t *settings;
} somr_trainer_t;

void somr_trainer_init(somr_trainer_t *t, somr_map_t *map, somr_dataset_t *dataset, double root_mean_error, double parent_mean_error, somr_trainer_settings_t *settings);
void somr_trainer_clear(somr_trainer_t *t);
/**
main training function
@p epoch_count: number of iterations to run (1 iteration = 1 pass on full data set)
@p radius: initial neighborhood radius
@p learn: initial learning rate
*/
void somr_trainer_train(somr_trainer_t *t);
void somr_trainer_expand(somr_trainer_t *t, somr_node_id_t error_node_id);
void somr_trainer_deepen(somr_trainer_t *t);
/** label nodes using labels of input vectors (to be run when training is over */
void somr_trainer_label(somr_trainer_t *t);
somr_node_id_t somr_trainer_compute_error(somr_trainer_t *t);
/** performs one pass of training using all input vectors */
void somr_trainer_run_epoch(somr_trainer_t *t, double radius, double learn_rate);
