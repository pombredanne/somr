#define _GNU_SOURCE // for strdup
#include "list.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef char *item_t;

struct somr_list_link_t {
    char *item;
    somr_list_link_t *next;
};

static item_t somr_list_copy_item(item_t item) {
    return strdup(item);
}

static void somr_list_free_item(item_t item) {
    free(item);
}

static int somr_list_cmp_items(item_t lhs, item_t rhs) {
    return strcmp(lhs, rhs);
}

void somr_list_init(somr_list_t *l, bool owns_items) {
    l->size = 0;
    l->owns_items = owns_items;
    l->first = NULL;
    l->last = NULL;
}

void somr_list_clear(somr_list_t *l) {
    somr_list_empty(l);
}

void somr_list_copy(somr_list_t *dest, somr_list_t *src) {
    somr_list_init(dest, src->owns_items);

    for (unsigned int i = 0; i < src->size; i++) {
        item_t item = somr_list_get(src, i);
        somr_list_push(dest, item);
    }
}

item_t somr_list_get(somr_list_t *l, unsigned int index) {
    assert(index < l->size);

    somr_list_link_t *link = l->first;
    for (unsigned int i = 0; i < index; i++) {
        assert(link->next != NULL);
        link = link->next;
    }
    return link->item;
}

bool somr_list_find(somr_list_t *l, item_t item, unsigned int *index_found) {
    somr_list_link_t *link = l->first;
    for (unsigned int i = 0; i < l->size; i++) {
        if (somr_list_cmp_items(link->item, item) == 0) {
            *index_found = i;
            return true;
        }
        link = link->next;
    }

    return false;
}

unsigned int somr_list_push(somr_list_t *l, item_t item) {
    // init new link
    somr_list_link_t *new_link = malloc(sizeof(somr_list_link_t));
    new_link->next = NULL;
    if (l->owns_items) {
        new_link->item = somr_list_copy_item(item);
    } else {
        new_link->item = item;
    }

    if (l->size == 0) {
        assert(l->first == NULL);
        assert(l->last == NULL);
        l->first = new_link;
    } else {
        // update last link
        somr_list_link_t *last_link = l->last;
        assert(last_link->next == NULL);
        last_link->next = new_link;
    }

    l->last = new_link;
    unsigned int index = l->size;
    l->size++;

    return index;
}

void somr_list_empty(somr_list_t *l) {
    if (l->size == 0) {
        return;
    }

    somr_list_link_t *link = l->first;
    for (unsigned int i = 0; i < l->size; i++) {
        somr_list_link_t *next_link = link->next;
        if (l->owns_items) {
            somr_list_free_item(link->item);
        }
        free(link);
        link = next_link;
    }

    l->size = 0;
    l->first = NULL;
    l->last = NULL;
}
