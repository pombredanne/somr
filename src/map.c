#include "map.h"
#include "node.h"
#include "vector.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

void somr_map_init(somr_map_t *m, unsigned int features_count) {
    assert(features_count > 0);

    m->width = 2;
    m->height = 2;
    m->nodes_count = m->width * m->height;
    m->features_count = features_count;

    m->nodes = malloc(sizeof(somr_node_t) * m->nodes_count);
    for (somr_node_id_t i = 0; i < m->nodes_count; i++) {
        somr_node_init(&m->nodes[i], m->features_count);
    }
}

void somr_map_clear(somr_map_t *m) {
    for (somr_node_id_t i = 0; i < m->nodes_count; i++) {
        somr_node_clear(&m->nodes[i]);
    }
    free(m->nodes);
}

void somr_map_init_weights(somr_map_t *m, double *weights) {
    // init nodes weights with random values
    for (somr_node_id_t i = 0; i < m->nodes_count; i++) {
        somr_node_init_weights(&m->nodes[i], weights, m->features_count);
    }
}

void somr_map_activate(somr_map_t *m, somr_data_vector_t *data_vector) {
    // TODO parallelize
    for (somr_node_id_t i = 0; i < m->nodes_count; i++) {
        somr_node_activate(&m->nodes[i], data_vector, m->features_count);
    }
}

/** computes activation value for all nodes in map, and returns in @p[out] bmu
all best matching unit with same activation value */
void somr_map_find_bmus(somr_map_t *m, somr_data_vector_t *data_vector, somr_node_id_t *bmus, unsigned int *bmu_count) {
    somr_map_activate(m, data_vector);

    double lowest_activation = DBL_MAX;
    unsigned int count = 0;
    for (somr_node_id_t i = 0; i < m->nodes_count; i++) {
        double activation = m->nodes[i].activation;
        // found lower activation, clear the results array
        if (activation < lowest_activation) {
            lowest_activation = activation;
            count = 1;
            bmus[0] = i;
        }
        // found equal activation, append to results array
        else if (activation == lowest_activation) {
            bmus[count] = i;
            count++;
        }
    }
    assert(lowest_activation < DBL_MAX);
    assert(count > 0);
    // returns by ref number of bmu founds
    *bmu_count = count;
}

/** computes activation value for all nodes in map, and returns first bmu encountered */
somr_node_id_t somr_map_find_bmu(somr_map_t *m, somr_data_vector_t *data_vector) {
    somr_map_activate(m, data_vector);

    somr_node_id_t bmu_id = 0;
    double lowest_activation = DBL_MAX;
    for (somr_node_id_t i = 0; i < m->nodes_count; i++) {
        // double activation = somr_map_activate(m, i, vector);
        double activation = m->nodes[i].activation;
        // found lower activation, select new bmu
        if (activation < lowest_activation) {
            lowest_activation = activation;
            bmu_id = i;
        }
    }
    assert(lowest_activation < DBL_MAX);
    return bmu_id;
}

void somr_map_insert_row(somr_map_t *m, unsigned int row_before) {
    assert(row_before + 1 < m->height);

    unsigned int new_nodes_count = m->nodes_count + m->width;
    somr_node_t *new_nodes = malloc(sizeof(somr_node_t) * new_nodes_count);
    if (new_nodes == NULL) {
        fprintf(stderr, "Could not allocate enough memory for new nodes");
        exit(EXIT_FAILURE);
    }

    unsigned int nodes_count_before = (row_before + 1) * m->width;
    unsigned int nodes_count_after = m->nodes_count - nodes_count_before;

    memcpy(&new_nodes[0], &m->nodes[0], sizeof(somr_node_t) * nodes_count_before);
    somr_node_id_t src_somr_node_id = nodes_count_before;
    somr_node_id_t dest_somr_node_id = nodes_count_before + m->width;

    assert(src_somr_node_id < m->nodes_count);
    assert(dest_somr_node_id < new_nodes_count);
    memcpy(&new_nodes[dest_somr_node_id], &m->nodes[src_somr_node_id], sizeof(somr_node_t) * nodes_count_after);

    free(m->nodes);
    m->nodes = new_nodes;
    m->nodes_count = new_nodes_count;
    m->height += 1;

    // init nodes in inserted row
    for (somr_node_id_t i = src_somr_node_id; i < dest_somr_node_id; i++) {
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

    somr_node_id_t src_somr_node_id = 0;
    somr_node_id_t dest_somr_node_id = 0;
    while (src_somr_node_id < m->nodes_count) {
        memcpy(&new_nodes[dest_somr_node_id], &m->nodes[src_somr_node_id], sizeof(somr_node_t) * cols_count_before);
        src_somr_node_id += cols_count_before;
        dest_somr_node_id += cols_count_before + 1;

        memcpy(&new_nodes[dest_somr_node_id], &m->nodes[src_somr_node_id], sizeof(somr_node_t) * cols_count_after);
        src_somr_node_id += cols_count_after;
        dest_somr_node_id += cols_count_after;
    }
    assert(src_somr_node_id == m->nodes_count);
    assert(dest_somr_node_id == new_nodes_count);

    free(m->nodes);
    m->nodes = new_nodes;
    m->nodes_count = new_nodes_count;
    m->width += 1;

    // init nodes in inserted col
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

somr_label_t somr_map_classify(somr_map_t *m, somr_data_vector_t *data_vector) {
    somr_node_id_t bmu_id = somr_map_find_bmu(m, data_vector);
    somr_node_t *bmu = &m->nodes[bmu_id];
    if (bmu->child == NULL) {
        return m->nodes[bmu_id].label;
    }
    return somr_map_classify(bmu->child, data_vector);
}

void somr_map_print_topology(somr_map_t *m) {
    printf("----\n");
    for (unsigned int y = 0; y < m->height; y++) {
        for (unsigned int x = 0; x < m->width; x++) {
            somr_node_id_t somr_node_id = y * m->width + x;
            somr_label_t label = m->nodes[somr_node_id].label;
            if (label == SOMR_EMPTY_LABEL) {
                printf(". ");
            } else {
                printf("%d ", label);
            }
        }
        printf("\n");
    }
    printf("----\n");

    for (somr_node_id_t i = 0; i < m->nodes_count; i++) {
        if (m->nodes[i].child != NULL) {
            somr_map_print_topology(m->nodes[i].child);
        }
    }
}

void somr_map_add_child(somr_map_t *m, somr_node_id_t somr_node_id) {
    somr_node_t *node = &m->nodes[somr_node_id];
    somr_node_add_child(node, m->features_count);
    return;

    unsigned int somr_node_y = somr_node_id / m->width;
    unsigned int somr_node_x = somr_node_id % m->width;

    double *src_up_weights = NULL;
    double *src_up_left_weights = NULL;
    double *src_up_right_weights = NULL;
    double *src_down_weights = NULL;
    double *src_down_left_weights = NULL;
    double *src_down_right_weights = NULL;
    double *src_left_weights = NULL;
    double *src_right_weights = NULL;

    if (somr_node_y > 0) {
        src_up_weights = m->nodes[somr_node_id - m->width].weights;
        if (somr_node_x > 0) {
            src_up_left_weights = m->nodes[somr_node_id - m->width - 1].weights;
        }
        if (somr_node_x < m->width - 1) {
            src_up_right_weights = m->nodes[somr_node_id - m->width + 1].weights;
        }
    }

    if (somr_node_y < m->height - 1) {
        src_down_weights = m->nodes[somr_node_id + m->width].weights;
        if (somr_node_x > 0) {
            src_down_left_weights = m->nodes[somr_node_id + m->width - 1].weights;
        }
        if (somr_node_x < m->width - 1) {
            src_down_right_weights = m->nodes[somr_node_id + m->width + 1].weights;
        }
    }

    if (somr_node_x > 0) {
        src_left_weights = m->nodes[somr_node_id - 1].weights;
    }
    if (somr_node_x < m->width - 1) {
        src_right_weights = m->nodes[somr_node_id + 1].weights;
    }

    double *dest_up_left_weights = node->child->nodes[0].weights;
    double *dest_up_right_weights = node->child->nodes[1].weights;
    double *dest_down_left_weights = node->child->nodes[2].weights;
    double *dest_down_right_weights = node->child->nodes[3].weights;

    somr_vector_mean_3(src_up_left_weights, src_up_weights, src_left_weights,
        dest_up_left_weights, m->features_count);
    somr_vector_mean_3(src_up_right_weights, src_up_weights, src_right_weights,
        dest_up_right_weights, m->features_count);
    somr_vector_mean_3(src_down_left_weights, src_down_weights, src_left_weights,
        dest_down_left_weights, m->features_count);
    somr_vector_mean_3(src_down_right_weights, src_down_weights, src_right_weights,
        dest_down_right_weights, m->features_count);
}

void somr_map_write_to_img(somr_map_t *m, unsigned char *img, unsigned int img_width, unsigned int img_height, unsigned char *colors) {
    // fill with black
    memset(img, 0, 3 * img_width * img_height);

    unsigned char white[] = { 255, 255, 255 };
    unsigned int border = 1;

    assert(img_height > 0);
    assert(img_width > 0);

    for (unsigned int somr_map_y = 0; somr_map_y < m->height; somr_map_y++) {
        for (unsigned int somr_map_x = 0; somr_map_x < m->width; somr_map_x++) {
            somr_node_id_t somr_node_id = somr_map_y * m->width + somr_map_x;
            somr_node_t *node = &m->nodes[somr_node_id];

            int somr_node_height = round(img_height / m->height);
            int somr_node_width = round(img_width / m->width);
            assert(somr_node_height > 0);
            assert(somr_node_height <= img_height);
            assert(somr_node_width > 0);
            assert(somr_node_width <= img_width);

            int img_y_begin = somr_map_y * somr_node_height;
            assert(img_y_begin >= 0);
            assert(img_y_begin < img_height);
            int img_x_begin = somr_map_x * somr_node_width;
            assert(img_x_begin >= 0);
            assert(img_x_begin < img_width);

            if (somr_map_y == m->height - 1) {
                somr_node_height = img_height - (m->height - 1) * somr_node_height;
            }

            if (somr_map_x == m->width - 1) {
                somr_node_width = img_width - (m->width - 1) * somr_node_width;
            }

            assert(somr_node_height > 0);
            assert(somr_node_height <= img_height);
            assert(somr_node_width > 0);
            assert(somr_node_width <= img_width);

            int img_y_end = img_y_begin + somr_node_height;
            assert(img_y_end > 0);
            assert(img_y_end <= img_height);
            int img_x_end = img_x_begin + somr_node_width;
            assert(img_x_end > 0);
            assert(img_x_end <= img_width);

            if (node->child != NULL) {
                unsigned char *somr_node_img = malloc(sizeof(unsigned char) * 3 * somr_node_width * somr_node_height);
                somr_map_write_to_img(node->child, somr_node_img, somr_node_width, somr_node_height, colors);

                for (int somr_node_img_y = border; somr_node_img_y < somr_node_height - border; somr_node_img_y++) {
                    assert(somr_node_img_y >= 0);
                    for (int somr_node_img_x = border; somr_node_img_x < somr_node_width - border; somr_node_img_x++) {
                        assert(somr_node_img_x >= 0);
                        int pixel_src = somr_node_img_y * somr_node_width + somr_node_img_x;
                        assert(pixel_src >= 0);
                        assert(pixel_src < somr_node_width * somr_node_height);

                        int img_y = img_y_begin + somr_node_img_y;
                        assert(img_y >= 0);
                        assert(img_y < img_height);
                        int img_x = img_x_begin + somr_node_img_x;
                        assert(img_x >= 0);
                        assert(img_x < img_width);
                        int pixel_dest = img_y * img_width + img_x;
                        assert(pixel_dest >= 0);
                        assert(pixel_dest < img_width * img_height);

                        memcpy(&img[pixel_dest * 3], &somr_node_img[pixel_src * 3], sizeof(unsigned char) * 3);
                    }
                }

                free(somr_node_img);
            } else {
                unsigned char *color;
                if (node->label == SOMR_EMPTY_LABEL) {
                    color = white;
                } else {
                    assert(node->label < 128); // TODO
                    color = &colors[node->label * 3];
                }

                for (unsigned int img_y = img_y_begin + border; img_y < img_y_end - border; img_y++) {
                    //assert(img_y < img_height);
                    if (img_y >= img_height) {
                        continue;
                    }
                    for (unsigned int img_x = img_x_begin + border; img_x < img_x_end - border; img_x++) {
                        //assert(img_x < img_width);
                        if (img_x >= img_width) {
                            continue;
                        }
                        unsigned int pixel_dest = img_y * img_width + img_x;
                        assert(pixel_dest < img_width * img_height);
                        memcpy(&img[pixel_dest * 3], color, sizeof(unsigned char) * 3);
                    }
                }
            }
        }
    }
}
