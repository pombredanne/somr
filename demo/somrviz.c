#include <getopt.h>
#include <somr/somr.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_ITERS_COUNT 500

void usage(char *exec_name) {
    fprintf(stderr, "Usage: %s -n <nb_vectors> -f <nb_features> [options] <in_data.csv>\n", exec_name);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -n <nb_vectors> Number of input vectors\n");
    fprintf(stderr, "  -f <nb_features> Number of values per input vector\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -l <learning_rate> Initial learning rate\n");
    fprintf(stderr, "  -t <tau_1> TODO\n");
    fprintf(stderr, "  -u <tau_2> TODO\n");
    fprintf(stderr, "  -v <lambda> TODO\n");
    fprintf(stderr, "  -s <seed> Seed for random number generator\n");
    fprintf(stderr, "  -p <out.ppm> Output topography as PPM image\n");
}

// feed all input vectors to network and check they are mapped to correct class
void print_errors(somr_network_t *network, somr_dataset_t *dataset) {
    printf("Testing input vectors classification\n");
    int error_count = 0;
    somr_dataset_shuffle(dataset);
    for (unsigned int i = 0; i < dataset->size; i++) {
        somr_data_vector_t *data_vector = somr_dataset_get_vector(dataset, i);
        somr_label_t label = somr_network_classify(network, data_vector);
        if (label != data_vector->label) {
            //printf("Error: %d mapped to %d\n", data_vector->label, label);
            error_count++;
        }
    }
    printf("Total number of incorrectly mapped vectors: %d\n", error_count);
}

void write_ppm(somr_network_t *network, char *filename, unsigned int width, unsigned int height) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Could not open %s\n", filename);
        exit(EXIT_FAILURE);
    }

    unsigned char *img = malloc(sizeof(unsigned char) * 3 * width * height);
    somr_network_write_to_img(network, img, width, height);
    fprintf(file, "P6\n%d %d\n255\n", width, height);
    fwrite(img, sizeof(unsigned char), 3 * height * width, file);
    free(img);
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    int data_length = -1;
    int features_count = -1;
    double learn_rate = -1.0;
    double tau_1 = -1.0;
    double tau_2 = -1.0;
    int lambda = -1.0;
    int seed = -1;
    char *ppm_filename = NULL;

    char opt;
    while ((opt = getopt(argc, argv, "n:f:l:t:u:v:s:p:")) != -1) {
        switch (opt) {
        case 'n':
            data_length = atoi(optarg);
            break;
        case 'f':
            features_count = atoi(optarg);
            break;
        case 'l':
            learn_rate = atof(optarg);
            break;
        case 't':
            tau_1 = atof(optarg);
            break;
        case 'u':
            tau_2 = atof(optarg);
            break;
        case 'v':
            lambda = atoi(optarg);
            break;
        case 's':
            seed = atoi(optarg);
            break;
        case 'p':
            ppm_filename = optarg;
            break;
        default:
            usage(argv[0]);
            exit(EXIT_FAILURE);
            break;
        }
    }

    if (data_length <= 0) {
        fprintf(stderr, "Number of input vectors missing\n");
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    if (features_count <= 0) {
        fprintf(stderr, "Number of values per input vector missig\n");
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    if (learn_rate <= 0.0) {
        learn_rate = 0.9;
    }

    if (tau_1 <= 0) {
        tau_1 = 0.9;
    }

    if (tau_2 <= 0) {
        tau_2 = 0.01;
    }

    if (lambda <= 0) {
        lambda = 100;
    }

    if (seed <= -1) {
        struct timeval time;
        gettimeofday(&time, NULL);
        seed = (time.tv_sec * 1000000) + time.tv_usec;
    }

    // input data filename
    if (argc - optind != 1) {
        fprintf(stderr, "Input file missing\n");
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    // open input data file
    char *input_filename = argv[optind];
    FILE *file = fopen(input_filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Could not open %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

    // read input data;
    somr_dataset_t dataset;
    somr_dataset_init_from_file(&dataset, file, data_length, features_count);
    somr_dataset_normalize(&dataset);
    // init and train network
    somr_network_t network;
    somr_network_init(&network, features_count);
    srand(seed); // TODO
    printf("Training network with tau_1=%f, tau_2=%f, lambda=%d, learning_rate=%f...\n", tau_1, tau_2, lambda, learn_rate);
    somr_network_train(&network, &dataset, learn_rate, tau_1, tau_2, lambda, MAX_ITERS_COUNT);

    // display results
    // printf("-----\n");
    // for (unsigned int i = 0; i < class_list.size; i++) {
    //     printf("%d: %s\n", i, list_get(&class_list, i));
    // }
    // printf("-----\n");
    // somr_network_print_topology(&network);
    printf("-----\n");
    print_errors(&network, &dataset);

    // write ppm
    if (ppm_filename != NULL) {
        write_ppm(&network, ppm_filename, 512, 512);
    }

    // clean all
    somr_network_clear(&network);
    somr_dataset_clear(&dataset);
}
