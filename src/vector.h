#pragma once

void somr_vector_normalize(double *v, unsigned int length);
double somr_vector_euclid_dist_squared(double *lhs, double *rhs, unsigned int length);
double somr_vector_euclid_dist(double *lhs, double *rhs, unsigned int length);
void somr_vector_mean_3(double *v1, double *v2, double *v3, double *result, unsigned int length);
