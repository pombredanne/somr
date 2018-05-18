#include "vector.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

void somr_vector_normalize(double *v, unsigned int length) {
    assert(length > 0);
    double sum = 0.0;
    for (unsigned int i = 0; i < length; i++) {
        sum += v[i] * v[i];
    }
    double norm = sqrt(sum);
    for (unsigned int i = 0; i < length; i++) {
        v[i] /= norm;
    }
}

double somr_vector_euclid_dist_squared(double *lhs, double *rhs, unsigned int length) {
    assert(length > 0);
    double result = 0.0;
    for (unsigned int i = 0; i < length; i++) {
        assert(lhs[i] >= 0);
        assert(rhs[i] >= 0);
        double delta = lhs[i] - rhs[i];
        result += delta * delta;
    }
    assert(result >= 0.0);
    return result;
}

double somr_vector_euclid_dist(double *lhs, double *rhs, unsigned int length) {
    assert(length > 0);
    return sqrt(somr_vector_euclid_dist_squared(lhs, rhs, length));
}

void somr_vectors_mean(double **vectors, unsigned int vectors_count, unsigned int length, double *result) {
    assert(length > 0);
    assert(vectors_count > 0);

    for (unsigned int i = 0; i < length; i++) {
        double sum = 0.0;
        for (unsigned int j = 0; j < vectors_count; j++) {
            sum += vectors[j][i];
        }
        result[i] = sum / (double) vectors_count;
    }
}
