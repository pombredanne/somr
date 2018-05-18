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
    m->nodes = NULL;
}

void somr_map_init_random_weights(somr_map_t *m, unsigned int *rand_state) {
    for (somr_node_id_t i = 0; i < m->nodes_count; i++) {
        somr_node_init_random_weights(&m->nodes[i], rand_state, m->features_count);
    }
}

void somr_map_activate(somr_map_t *m, somr_data_vector_t *data_vector) {
    // TODO parallelize?
    for (somr_node_id_t i = 0; i < m->nodes_count; i++) {
        somr_node_activate(&m->nodes[i], data_vector, m->features_count);
    }
}

/** computes activation value for all nodes in map, and returns in @p[out] bmu
all best matching unit with same activation value */
// void somr_map_find_bmus(somr_map_t *m, somr_data_vector_t *data_vector, somr_node_id_t *bmus, unsigned int *bmu_count) {
//     somr_map_activate(m, data_vector);

//     double lowest_activation = DBL_MAX;
//     unsigned int count = 0;
//     for (somr_node_id_t i = 0; i < m->nodes_count; i++) {
//         double activation = m->nodes[i].activation;
//         // found lower activation, clear the results array
//         if (activation < lowest_activation) {
//             lowest_activation = activation;
//             count = 1;
//             bmus[0] = i;
//         }
//         // found equal activation, append to results array
//         else if (activation == lowest_activation) {
//             bmus[count] = i;
//             count++;
//         }
//     }
//     assert(lowest_activation < DBL_MAX);
//     assert(count > 0);
//     // returns by ref number of bmu founds
//     *bmu_count = count;
// }

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

void somr_map_teach_nbhd(somr_map_t *m, somr_node_id_t node_id, somr_data_vector_t *data_vector, double learn_rate, double radius) {
    double node_y = node_id / m->width;
    double node_x = node_id % m->width;

    // TODO avoid loop on all nodes ?
    for (unsigned int y = 0; y < m->height; y++) {
        unsigned int row_begin = y * m->width;
        double dist_y = node_y - y;
        double dist_y_square = dist_y * dist_y;
        for (unsigned int x = 0; x < m->width; x++) {
            somr_node_id_t node_id = row_begin + x;
            double dist_x = node_x - x;
            // compute euclidean distance
            double dist = sqrt(dist_x * dist_x + dist_y_square);
            // compute gaussian attenuation factor
            double nbhd_factor = learn_rate * exp(-1.0 * (dist * dist) / (2.0 * radius * radius));
            if (nbhd_factor > 0.0) {
                // teach node
                somr_node_learn(&m->nodes[node_id], data_vector, m->features_count, nbhd_factor);
            }
        }
    }
}

unsigned int somr_map_get_depth(somr_map_t *m) {
    unsigned int max_child_depth = 0;
    for (somr_node_id_t i = 0; i < m->nodes_count; i++) {
        if (m->nodes[i].child != NULL) {
            unsigned int child_depth = somr_map_get_depth(m->nodes[i].child);
            if (child_depth > max_child_depth) {
                max_child_depth = child_depth;
            }
        }
    }
    return max_child_depth + 1;
}

somr_label_t somr_map_classify(somr_map_t *m, somr_data_vector_t *data_vector) {
    somr_node_id_t bmu_id = somr_map_find_bmu(m, data_vector);
    somr_node_t *bmu = &m->nodes[bmu_id];
    if (bmu->child == NULL) {
        return m->nodes[bmu_id].label;
    }
    return somr_map_classify(bmu->child, data_vector);
}

void somr_map_find_error_range(somr_map_t *m, double *min_error, double *max_error) {
    *max_error = -1.0;
    *min_error = DBL_MAX;
    for (somr_node_id_t i = 0; i < m->nodes_count; i++) {
        somr_node_t *node = &m->nodes[i];
        double node_min_error = 0.0;
        double node_max_error = 0.0;
        /*if (node->child != NULL) {
            somr_map_find_error_range(node->child, &node_min_error, &node_max_error);
        } else {*/
        node_min_error = node->error;
        node_max_error = node->error;
        //}

        if (node_max_error > *max_error) {
            *max_error = node_max_error;
        }
        if (node_min_error < *min_error) {
            *min_error = node_min_error;
        }
    }

    assert(*max_error > -1.0);
    assert(*min_error < DBL_MAX);
}

void somr_map_write_to_img(somr_map_t *m, unsigned char *img, unsigned int img_width, unsigned int img_height, unsigned char *colors, unsigned int border) {
    assert(img_height > 0 && img_width > 0);
    assert(border < img_height && border < img_width);

    unsigned char white[] = { 255, 255, 255 };

    // compute node dimensions
    unsigned int node_height = round(img_height / m->height);
    unsigned int node_width = floor(img_width / m->width);
    assert(node_height > 0 && node_height <= img_height);
    assert(node_width > 0 && node_width <= img_width);

    // fill with black (border color)
    memset(img, 0, 3 * img_width * img_height);

    for (unsigned int map_y = 0; map_y < m->height; map_y++) {
        for (unsigned int map_x = 0; map_x < m->width; map_x++) {
            somr_node_id_t node_id = map_y * m->width + map_x;
            somr_node_t *node = &m->nodes[node_id];

            unsigned int img_y_begin = map_y * node_height;
            unsigned int img_x_begin = map_x * node_width;
            assert(img_y_begin < img_height);
            assert(img_x_begin < img_width);

            unsigned int img_y_end = img_y_begin + node_height;
            unsigned int img_x_end = img_x_begin + node_width;
            // if node is a left or bottom border, make sure it fills all pixels left
            if (map_y == m->height - 1) {
                img_y_end = img_height;
            }
            if (map_x == m->width - 1) {
                img_x_end = img_width;
            }
            assert(img_y_end > 0 && img_y_end <= img_height);
            assert(img_x_end > 0 && img_x_end <= img_width);

            // if node has child, compute sub image for child map and copy it into parent image
            if (node->child != NULL) {
                unsigned char *node_img = malloc(sizeof(unsigned char) * 3 * node_width * node_height);
                assert(border >= 2);
                somr_map_write_to_img(node->child, node_img, node_width, node_height, colors, border - 2);

                for (unsigned int node_img_y = border; node_img_y < node_height - border; node_img_y++) {
                    for (unsigned int node_img_x = border; node_img_x < node_width - border; node_img_x++) {
                        unsigned int pixel_src = node_img_y * node_width + node_img_x;
                        assert(pixel_src < node_width * node_height);

                        unsigned int img_y = img_y_begin + node_img_y;
                        unsigned int img_x = img_x_begin + node_img_x;
                        unsigned int pixel_dest = img_y * img_width + img_x;
                        assert(pixel_dest < img_width * img_height);

                        memcpy(&img[pixel_dest * 3], &node_img[pixel_src * 3], sizeof(unsigned char) * 3);
                    }
                }

                free(node_img);
            } else {
                unsigned char *color;
                // white color if node a no label
                if (node->label == SOMR_EMPTY_LABEL) {
                    color = white;
                } else {
                    color = &colors[node->label * 3];
                }

                for (unsigned int img_y = img_y_begin + border; img_y < img_y_end - border; img_y++) {
                    assert(img_y < img_height);
                    if (img_y >= img_height) {
                        continue;
                    }
                    for (unsigned int img_x = img_x_begin + border; img_x < img_x_end - border; img_x++) {
                        unsigned int pixel_dest = img_y * img_width + img_x;
                        assert(pixel_dest < img_width * img_height);
                        memcpy(&img[pixel_dest * 3], color, sizeof(unsigned char) * 3);
                    }
                }
            }
        }
    }
}
