#include "map_grow.h"
#include "vector.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef enum somr_octant_t {
    // preserve order so that first 4 octants matches units in new 2x2 map
    SOMR_OCTANT_UP_LEFT,
    SOMR_OCTANT_UP_RIGHT,
    SOMR_OCTANT_DOWN_LEFT,
    SOMR_OCTANT_DOWN_RIGHT,
    SOMR_OCTANT_UP,
    SOMR_OCTANT_DOWN,
    SOMR_OCTANT_LEFT,
    SOMR_OCTANT_RIGHT,
    SOMR_OCTANT_NONE
} somr_octant_t;

static void somr_map_orient_child(somr_map_t *m, somr_unit_id_t unit_id);
static void somr_map_init_up_left_weights(somr_map_t *m, somr_unit_t *parent, somr_unit_t **parent_nbs);
static void somr_map_init_up_right_weights(somr_map_t *m, somr_unit_t *parent, somr_unit_t **parent_nbs);
static void somr_map_init_down_left_weights(somr_map_t *m, somr_unit_t *parent, somr_unit_t **parent_nbs);
static void somr_map_init_down_right_weights(somr_map_t *m, somr_unit_t *parent, somr_unit_t **parent_nbs);

void somr_map_insert_row(somr_map_t *m, unsigned int row_before) {
    assert(row_before + 1 < m->height);

    unsigned int new_units_count = m->units_count + m->width;
    somr_unit_t *new_units = malloc(sizeof(somr_unit_t) * new_units_count);
    unsigned int units_count_before = (row_before + 1) * m->width;
    unsigned int units_count_after = m->units_count - units_count_before;

    memcpy(&new_units[0], &m->units[0], sizeof(somr_unit_t) * units_count_before);
    somr_unit_id_t src_unit_id = units_count_before;
    somr_unit_id_t dest_unit_id = units_count_before + m->width;

    assert(src_unit_id < m->units_count);
    assert(dest_unit_id < new_units_count);
    memcpy(&new_units[dest_unit_id], &m->units[src_unit_id], sizeof(somr_unit_t) * units_count_after);

    free(m->units);
    m->units = new_units;
    m->units_count = new_units_count;
    m->height += 1;

    // init units in inserted row with meam weights
    for (somr_unit_id_t i = src_unit_id; i < dest_unit_id; i++) {
        somr_unit_t *unit = &m->units[i];
        somr_unit_init(unit, m->features_count);

        double *weights_before = m->units[i - m->width].weights;
        double *weights_after = m->units[i + m->width].weights;
        for (unsigned int j = 0; j < m->features_count; j++) {
            unit->weights[j] = (weights_before[j] + weights_after[j]) / 2.0;
        }
    }
}

void somr_map_insert_col(somr_map_t *m, unsigned int col_before) {
    assert(col_before + 1 < m->width);

    unsigned int new_units_count = m->units_count + m->height;
    somr_unit_t *new_units = malloc(sizeof(somr_unit_t) * new_units_count);
    unsigned int cols_count_before = col_before + 1;
    unsigned int cols_count_after = m->width - cols_count_before;

    somr_unit_id_t src_unit_id = 0;
    somr_unit_id_t dest_unit_id = 0;
    while (src_unit_id < m->units_count) {
        memcpy(&new_units[dest_unit_id], &m->units[src_unit_id], sizeof(somr_unit_t) * cols_count_before);
        src_unit_id += cols_count_before;
        dest_unit_id += cols_count_before + 1;

        memcpy(&new_units[dest_unit_id], &m->units[src_unit_id], sizeof(somr_unit_t) * cols_count_after);
        src_unit_id += cols_count_after;
        dest_unit_id += cols_count_after;
    }
    assert(src_unit_id == m->units_count);
    assert(dest_unit_id == new_units_count);

    free(m->units);
    m->units = new_units;
    m->units_count = new_units_count;
    m->width += 1;

    // init units in inserted column with mean weights
    for (somr_unit_id_t i = col_before + 1; i < m->units_count; i += m->width) {
        somr_unit_t *unit = &m->units[i];
        somr_unit_init(unit, m->features_count);

        double *weights_before = m->units[i - 1].weights;
        double *weights_after = m->units[i + 1].weights;
        for (unsigned int j = 0; j < m->features_count; j++) {
            unit->weights[j] = (weights_before[j] + weights_after[j]) / 2.0;
        }
    }
}

void somr_map_add_child(somr_map_t *m, somr_unit_id_t unit_id, bool should_orient, unsigned int *rand_state) {
    somr_unit_t *unit = &m->units[unit_id];
    somr_unit_add_child(unit, m->features_count);

    if (should_orient) {
        somr_map_orient_child(m, unit_id);
    } else {
        somr_map_init_random_weights(unit->child, rand_state);
    }
}

static void somr_map_orient_child(somr_map_t *m, somr_unit_id_t unit_id) {
    unsigned int unit_y = unit_id / m->width;
    unsigned int unit_x = unit_id % m->width;

    // locate available neighbors in parent map
    somr_unit_t *nbs[8] = { NULL };

    if (unit_y > 0) {
        nbs[SOMR_OCTANT_UP] = &m->units[unit_id - m->width];
        if (unit_x > 0) {
            nbs[SOMR_OCTANT_UP_LEFT] = &m->units[unit_id - m->width - 1];
        }
        if (unit_x < m->width - 1) {
            nbs[SOMR_OCTANT_UP_RIGHT] = &m->units[unit_id - m->width + 1];
        }
    }

    if (unit_y < m->height - 1) {
        nbs[SOMR_OCTANT_DOWN] = &m->units[unit_id + m->width];
        if (unit_x > 0) {
            nbs[SOMR_OCTANT_DOWN_LEFT] = &m->units[unit_id + m->width - 1];
        }
        if (unit_x < m->width - 1) {
            nbs[SOMR_OCTANT_DOWN_RIGHT] = &m->units[unit_id + m->width - 1];
        }
    }

    if (unit_x > 0) {
        nbs[SOMR_OCTANT_LEFT] = &m->units[unit_id - 1];
    }
    if (unit_x < m->width - 1) {
        nbs[SOMR_OCTANT_RIGHT] = &m->units[unit_id + 1];
    }

    somr_unit_t *parent = &m->units[unit_id];
    // init child map with weights from neighbors of parent in parent map
    somr_map_init_up_left_weights(m->units[unit_id].child, parent, nbs);
    somr_map_init_up_right_weights(m->units[unit_id].child, parent, nbs);
    somr_map_init_down_left_weights(m->units[unit_id].child, parent, nbs);
    somr_map_init_down_right_weights(m->units[unit_id].child, parent, nbs);
}

static void somr_map_init_up_left_weights(somr_map_t *m, somr_unit_t *parent, somr_unit_t **parent_nbs) {
    assert(m->units_count == 4);
    somr_unit_t *unit = &m->units[SOMR_OCTANT_UP_LEFT];

    double *orientation_weights[5] = {
        parent->weights,
        NULL, NULL, NULL, NULL
    };
    unsigned int orientation_weights_count = 1;

    if (parent_nbs[SOMR_OCTANT_UP_LEFT] != NULL) {
        orientation_weights[orientation_weights_count] = parent_nbs[SOMR_OCTANT_UP_LEFT]->weights;
        orientation_weights_count++;
    }
    if (parent_nbs[SOMR_OCTANT_UP] != NULL) {
        orientation_weights[orientation_weights_count] = parent_nbs[SOMR_OCTANT_UP]->weights;
        orientation_weights_count++;
    }
    if (parent_nbs[SOMR_OCTANT_LEFT] != NULL) {
        orientation_weights[orientation_weights_count] = parent_nbs[SOMR_OCTANT_LEFT]->weights;
        orientation_weights_count++;
    }
    //assert(orientation_weights_count > 1);
    somr_vectors_mean(orientation_weights, orientation_weights_count, m->features_count, unit->weights);
}

static void somr_map_init_up_right_weights(somr_map_t *m, somr_unit_t *parent, somr_unit_t **parent_nbs) {
    assert(m->units_count == 4);
    somr_unit_t *unit = &m->units[SOMR_OCTANT_UP_RIGHT];

    double *orientation_weights[5] = {
        parent->weights,
        NULL, NULL, NULL, NULL
    };
    unsigned int orientation_weights_count = 1;

    if (parent_nbs[SOMR_OCTANT_UP_RIGHT] != NULL) {
        orientation_weights[orientation_weights_count] = parent_nbs[SOMR_OCTANT_UP_RIGHT]->weights;
        orientation_weights_count++;
    }
    if (parent_nbs[SOMR_OCTANT_UP] != NULL) {
        orientation_weights[orientation_weights_count] = parent_nbs[SOMR_OCTANT_UP]->weights;
        orientation_weights_count++;
    }
    if (parent_nbs[SOMR_OCTANT_RIGHT] != NULL) {
        orientation_weights[orientation_weights_count] = parent_nbs[SOMR_OCTANT_RIGHT]->weights;
        orientation_weights_count++;
    }
    //assert(orientation_weights_count > 1);
    somr_vectors_mean(orientation_weights, orientation_weights_count, m->features_count, unit->weights);
}

static void somr_map_init_down_left_weights(somr_map_t *m, somr_unit_t *parent, somr_unit_t **parent_nbs) {
    assert(m->units_count == 4);
    somr_unit_t *unit = &m->units[SOMR_OCTANT_DOWN_LEFT];

    double *orientation_weights[5] = {
        parent->weights,
        NULL, NULL, NULL, NULL
    };
    unsigned int orientation_weights_count = 1;

    orientation_weights_count = 1;
    if (parent_nbs[SOMR_OCTANT_DOWN_RIGHT] != NULL) {
        orientation_weights[orientation_weights_count] = parent_nbs[SOMR_OCTANT_DOWN_RIGHT]->weights;
        orientation_weights_count++;
    }

    if (parent_nbs[SOMR_OCTANT_DOWN] != NULL) {
        orientation_weights[orientation_weights_count] = parent_nbs[SOMR_OCTANT_DOWN]->weights;
        orientation_weights_count++;
    }

    if (parent_nbs[SOMR_OCTANT_RIGHT] != NULL) {
        orientation_weights[orientation_weights_count] = parent_nbs[SOMR_OCTANT_RIGHT]->weights;
        orientation_weights_count++;
    }
    //assert(orientation_weights_count > 1);
    somr_vectors_mean(orientation_weights, orientation_weights_count, m->features_count, unit->weights);
}

static void somr_map_init_down_right_weights(somr_map_t *m, somr_unit_t *parent, somr_unit_t **parent_nbs) {
    assert(m->units_count == 4);
    somr_unit_t *unit = &m->units[SOMR_OCTANT_DOWN_RIGHT];

    double *orientation_weights[5] = {
        parent->weights,
        NULL, NULL, NULL, NULL
    };
    unsigned int orientation_weights_count = 1;

    if (parent_nbs[SOMR_OCTANT_DOWN_RIGHT] != NULL) {
        orientation_weights[orientation_weights_count] = parent_nbs[SOMR_OCTANT_DOWN_RIGHT]->weights;
        orientation_weights_count++;
    }
    if (parent_nbs[SOMR_OCTANT_DOWN] != NULL) {
        orientation_weights[orientation_weights_count] = parent_nbs[SOMR_OCTANT_DOWN]->weights;
        orientation_weights_count++;
    }
    if (parent_nbs[SOMR_OCTANT_RIGHT] != NULL) {
        orientation_weights[orientation_weights_count] = parent_nbs[SOMR_OCTANT_RIGHT]->weights;
        orientation_weights_count++;
    }
    //assert(orientation_weights_count > 1);
    somr_vectors_mean(orientation_weights, orientation_weights_count, m->features_count, unit->weights);
}
