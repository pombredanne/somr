#pragma once
#include <stdbool.h>

typedef struct somr_list_link_t somr_list_link_t;

/** Linked list of strings, used to store classes */
typedef struct somr_list_t {
    /** number of items in list */
    unsigned int size;
    bool owns_items;
    /** pointer to first item, NULL if list is empty */
    somr_list_link_t *first;
    /** pointer to last item, NULL if list is empty */
    somr_list_link_t *last;
} somr_list_t;

void somr_list_init(somr_list_t *l, bool owns_items);
void somr_list_clear(somr_list_t *l);
void somr_list_copy(somr_list_t *dest, somr_list_t *src);
unsigned int somr_list_push(somr_list_t *l, char *item);
char *somr_list_get(somr_list_t *l, unsigned int index);
/**
@return true if list contains an item equal to @p item (using strcmp for comparison)
@p[out] index_found: index of item if found
*/
bool somr_list_find(somr_list_t *l, char *item, unsigned int *index_found);
/**
removes all items from list
@p should_free_items: if true, items (strings) will be deallocated as well */
void somr_list_empty(somr_list_t *l);
