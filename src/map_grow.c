#include "map_grow.h"
#include "vector.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef enum somr_octant_t {
    // preserve order so that first 4 octants matches nodes in new 2x2 map
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

static void somr_map_orient_child(somr_map_t *m, somr_node_id_t node_id);
static void somr_map_init_up_left_weights(somr_map_t *m, somr_node_t *parent, somr_node_t **parent_nbs);
static void somr_map_init_up_right_weights(somr_map_t *m, somr_node_t *parent, somr_node_t **parent_nbs);
static void somr_map_init_down_left_weights(somr_map_t *m, somr_node_t *parent, somr_node_t **parent_nbs);
static void somr_map_init_down_right_weights(somr_map_t *m, somr_node_t *parent, somr_node_t **parent_nbs);

void somr_map_insert_row(somr_map_t *m, unsigned int row_before) {
    assert(row_before + 1 < m->height);

    unsigned int new_nodes_count = m->nodes_count + m->width;
    somr_node_t *new_nodes = malloc(sizeof(somr_node_t) * new_nodes_count);
    unsigned int nodes_count_before = (row_before + 1) * m->width;
    unsigned int nodes_count_after = m->nodes_count - nodes_count_before;

    memcpy(&new_nodes[0], &m->nodes[0], sizeof(somr_node_t) * nodes_count_before);
    somr_node_id_t src_node_id = nodes_count_before;
    somr_node_id_t dest_node_id = nodes_count_before + m->width;

    assert(src_node_id < m->nodes_count);
    assert(dest_node_id < new_nodes_count);
    memcpy(&new_nodes[dest_node_id], &m->nodes[src_node_id], sizeof(somr_node_t) * nodes_count_after);

    free(m->nodes);
    m->nodes = new_nodes;
    m->nodes_count = new_nodes_count;
    m->height += 1;

    // init nodes in inserted row with meam weights
    for (somr_node_id_t i = src_node_id; i < dest_node_id; i++) {
        somr_node_t *node = &m->nodes[i];
        somr_node_init(node, m->features_count);

        double *weights_before = m->nodes[i - m->width].weights;
        double *weights_after = m->nodes[i + m->width].weights;
        for (unsigned int j = 0; j < m->features_count; j++) {
            node->weights[j] = (weights_before[j] + weights_after[j]) / 2.0;
        }
    }
}

void somr_map_insert_col(somr_map_t *m, unsigned int col_before) {
    assert(col_before + 1 < m->width);

    unsigned int new_nodes_count = m->nodes_count + m->height;
    somr_node_t *new_nodes = malloc(sizeof(somr_node_t) * new_nodes_count);
    unsigned int cols_count_before = col_before + 1;
    unsigned int cols_count_after = m->width - cols_count_before;

    somr_node_id_t src_node_id = 0;
    somr_node_id_t dest_node_id = 0;
    while (src_node_id < m->nodes_count) {
        memcpy(&new_nodes[dest_node_id], &m->nodes[src_node_id], sizeof(somr_node_t) * cols_count_before);
        src_node_id += cols_count_before;
        dest_node_id += cols_count_before + 1;

        memcpy(&new_nodes[dest_node_id], &m->nodes[src_node_id], sizeof(somr_node_t) * cols_count_after);
        src_node_id += cols_count_after;
        dest_node_id += cols_count_after;
    }
    assert(src_node_id == m->nodes_count);
    assert(dest_node_id == new_nodes_count);

    free(m->nodes);
    m->nodes = new_nodes;
    m->nodes_count = new_nodes_count;
    m->width += 1;

    // init nodes in inserted column with mean weights
    for (somr_node_id_t i = col_before + 1; i < m->nodes_count; i += m->width) {
        somr_node_t *node = &m->nodes[i];
        somr_node_init(node, m->features_count);

        double *weights_before = m->nodes[i - 1].weights;
        double *weights_after = m->nodes[i + 1].weights;
        for (unsigned int j = 0; j < m->features_count; j++) {
            node->weights[j] = (weights_before[j] + weights_after[j]) / 2.0;
        }
    }
}

void somr_map_add_child(somr_map_t *m, somr_node_id_t node_id, bool should_orient, unsigned int *rand_state) {
    somr_node_t *node = &m->nodes[node_id];
    somr_node_add_child(node, m->features_count);

    if (should_orient) {
        somr_map_orient_child(m, node_id);
    } else {
        somr_map_init_random_weights(node->child, rand_state);
    }
}

static void somr_map_orient_child(somr_map_t *m, somr_node_id_t node_id) {
    unsigned int node_y = node_id / m->width;
    unsigned int node_x = node_id % m->width;

    // locate available neighbors in parent map
    somr_node_t *nbs[8] = { NULL };

    if (node_y > 0) {
        nbs[SOMR_OCTANT_UP] = &m->nodes[node_id - m->width];
        if (node_x > 0) {
            nbs[SOMR_OCTANT_UP_LEFT] = &m->nodes[node_id - m->width - 1];
        }
        if (node_x < m->width - 1) {
            nbs[SOMR_OCTANT_UP_RIGHT] = &m->nodes[node_id - m->width + 1];
        }
    }

    if (node_y < m->height - 1) {
        nbs[SOMR_OCTANT_DOWN] = &m->nodes[node_id + m->width];
        if (node_x > 0) {
            nbs[SOMR_OCTANT_DOWN_LEFT] = &m->nodes[node_id + m->width - 1];
        }
        if (node_x < m->width - 1) {
            nbs[SOMR_OCTANT_DOWN_RIGHT] = &m->nodes[node_id + m->width - 1];
        }
    }

    if (node_x > 0) {
        nbs[SOMR_OCTANT_LEFT] = &m->nodes[node_id - 1];
    }
    if (node_x < m->width - 1) {
        nbs[SOMR_OCTANT_RIGHT] = &m->nodes[node_id + 1];
    }

    somr_node_t *parent = &m->nodes[node_id];
    // init child map with weights from neighbors of parent in parent map
    somr_map_init_up_left_weights(m->nodes[node_id].child, parent, nbs);
    somr_map_init_up_right_weights(m->nodes[node_id].child, parent, nbs);
    somr_map_init_down_left_weights(m->nodes[node_id].child, parent, nbs);
    somr_map_init_down_right_weights(m->nodes[node_id].child, parent, nbs);
}

static void somr_map_init_up_left_weights(somr_map_t *m, somr_node_t *parent, somr_node_t **parent_nbs) {
    assert(m->nodes_count == 4);
    somr_node_t *node = &m->nodes[SOMR_OCTANT_UP_LEFT];

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
    somr_vectors_mean(orientation_weights, orientation_weights_count, m->features_count, node->weights);
}

static void somr_map_init_up_right_weights(somr_map_t *m, somr_node_t *parent, somr_node_t **parent_nbs) {
    assert(m->nodes_count == 4);
    somr_node_t *node = &m->nodes[SOMR_OCTANT_UP_RIGHT];

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
    somr_vectors_mean(orientation_weights, orientation_weights_count, m->features_count, node->weights);
}

static void somr_map_init_down_left_weights(somr_map_t *m, somr_node_t *parent, somr_node_t **parent_nbs) {
    assert(m->nodes_count == 4);
    somr_node_t *node = &m->nodes[SOMR_OCTANT_DOWN_LEFT];

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
    somr_vectors_mean(orientation_weights, orientation_weights_count, m->features_count, node->weights);
}

static void somr_map_init_down_right_weights(somr_map_t *m, somr_node_t *parent, somr_node_t **parent_nbs) {
    assert(m->nodes_count == 4);
    somr_node_t *node = &m->nodes[SOMR_OCTANT_DOWN_RIGHT];

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
    somr_vectors_mean(orientation_weights, orientation_weights_count, m->features_count, node->weights);
}
