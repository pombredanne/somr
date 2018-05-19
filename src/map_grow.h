#pragma once
#include "map.h"
#include "unit.h"
#include <stdbool.h>

void somr_map_insert_row(somr_map_t *m, unsigned int row_before);
void somr_map_insert_col(somr_map_t *m, unsigned int col_before);
void somr_map_add_child(somr_map_t *m, somr_unit_id_t unit_id, bool should_orient, unsigned int *rand_state);
