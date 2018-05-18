#pragma once

void somr_vector_normalize(double *v, unsigned int length);
double somr_vector_euclid_dist_squared(double *lhs, double *rhs, unsigned int length);
double somr_vector_euclid_dist(double *lhs, double *rhs, unsigned int length);
void somr_vectors_mean(double **vectors, unsigned int vectors_count, unsigned int length, double *result);
