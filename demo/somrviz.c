#include <getopt.h>
#include <png.h>
#include <somr/somr.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define IMG_WIDTH 512
#define IMG_HEIGHT 512

unsigned char COLORS[] = {
    230, 25, 75,
    60, 180, 75,
    255, 225, 25,
    0, 130, 200,
    245, 130, 48,
    145, 30, 180,
    70, 240, 240,
    240, 50, 230,
    0, 128, 128,
    230, 190, 255,
};

// feed all input vectors to network and check they are mapped to correct class
int print_errors(somr_network_t *network, somr_dataset_t *dataset, unsigned int seed) {
    printf("Testing input vectors classification\n");
    int error_count = 0;
    somr_dataset_shuffle(dataset, &seed);
    for (unsigned int i = 0; i < dataset->size; i++) {
        somr_data_vector_t *data_vector = somr_dataset_get_vector(dataset, i);
        somr_label_t label = somr_network_classify(network, data_vector);
        if (label != data_vector->label) {
            //printf("Error: %u mapped to %u\n", data_vector->label, label);
            error_count++;
        }
    }
    printf("Total number of classification errors: %u\n", error_count);
    return error_count;
}

void write_img_to_png(FILE *file, unsigned char *img, unsigned int width, unsigned int height) {
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop png_info = png_create_info_struct(png);
    jmp_buf png_jmp;
    setjmp(png_jmp);
    png_init_io(png, file);

    png_set_IHDR(
        png,
        png_info,
        width,
        height,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE,
        PNG_FILTER_TYPE_BASE);

    png_write_info(png, png_info);

    // pointers to img data
    png_bytep row = malloc(sizeof(png_byte) * width * 3);
    for (unsigned int y = 0; y < height; y++) {
        unsigned int row_offset = y * width * 3;
        for (unsigned int i = 0; i < width * 3; i++) {
            row[i] = img[row_offset + i];
        }
        png_write_row(png, row);
    }
    free(row);
    row = NULL;
    png_write_end(png, NULL);

    png_destroy_write_struct(&png, &png_info);
}

void usage(char *exec_name) {
    fprintf(stderr, "Usage: %s -n <nb_vectors> -f <nb_features> [options] <in.csv> <out.png>\n", exec_name);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -n <nb_vectors>\t\tNumber of input vectors\n");
    fprintf(stderr, "  -f <nb_features>\t\tNumber of values per input vector\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -l <learning_rate>\t\tInitial learning rate [default: 0.8]\n");
    fprintf(stderr, "  -i <nb_iters>\t\t\tNumber of full training passes [default: 100]\n");
    fprintf(stderr, "  -s <spread_threshold>\t\tUnit insertion treshold [default: 0.05]\n");
    fprintf(stderr, "  -d <depth_threshold>\t\tChild map creation treshold  [default: 0.01]\n");
    fprintf(stderr, "  -o\t\t\t\tSwitch off orientation\n");
    fprintf(stderr, "  -r <random_seed>\t\t\tSeed for random number generator\n");
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    int data_length = -1;
    int features_count = -1;
    double learn_rate = -1.0;
    double spread_threshold = -1.0;
    double depth_threshold = -1.0;
    int iters_count = -1.0;
    unsigned int seed;
    bool has_seed = false;
    bool should_orient = true;

    char opt;
    while ((opt = getopt(argc, argv, "n:f:l:i:s:d:or:")) != -1) {
        switch (opt) {
        case 'n':
            data_length = atoi(optarg);
            if (data_length <= 0) {
                fprintf(stderr, "Invalid number of input vectors\n");
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            break;
        case 'f':
            features_count = atoi(optarg);
            if (features_count <= 0) {
                fprintf(stderr, "Invalid number of values per input vector\n");
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            break;
        case 'i':
            iters_count = atoi(optarg);
            if (iters_count < 0) {
                fprintf(stderr, "Invalid number of full training passes\n");
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            break;
        case 'l':
            learn_rate = atof(optarg);
            if (learn_rate <= 0.0 || learn_rate >= 1.0) {
                fprintf(stderr, "Invalid learning rate\n");
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            break;
        case 's':
            spread_threshold = atof(optarg);
            if (spread_threshold <= 0.0 || spread_threshold > 1.0) {
                fprintf(stderr, "Invalid spread threshold\n");
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            break;
        case 'd':
            depth_threshold = atof(optarg);
            depth_threshold = atof(optarg);
            if (depth_threshold <= 0.0 || depth_threshold > 1.0) {
                fprintf(stderr, "Invalid depth_threshold\n");
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
            break;
        case 'r':
            seed = atoi(optarg);
            has_seed = true;
            break;
        case 'o':
            should_orient = false;
            break;
        default:
            usage(argv[0]);
            exit(EXIT_FAILURE);
            break;
        }
    }

    if (argc - optind != 2) {
        fprintf(stderr, "Positional arguments missing\n");
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    if (data_length <= 0) {
        fprintf(stderr, "Number of input vectors missing\n");
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    if (features_count <= 0) {
        fprintf(stderr, "Number of values per input vector missing\n");
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    if (learn_rate <= 0.0) {
        learn_rate = 0.8;
    }
    if (spread_threshold <= 0) {
        spread_threshold = 0.05;
    }
    if (depth_threshold <= 0) {
        depth_threshold = 0.01;
    }
    if (iters_count <= 0) {
        iters_count = 100;
    }
    if (!has_seed) {
        struct timeval time;
        gettimeofday(&time, NULL);
        seed = (time.tv_sec * 1000000) + time.tv_usec;
    }

    char *csv_filename = argv[optind];
    char *png_filename = argv[optind + 1];

    // read input data;
    FILE *file = fopen(csv_filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Could not open %s\n", csv_filename);
        exit(EXIT_FAILURE);
    }
    somr_dataset_t dataset;
    somr_dataset_init_from_file(&dataset, file, data_length, features_count);
    fclose(file);
    somr_dataset_normalize(&dataset);

    if (dataset.class_list->size > 10) {
        fprintf(stderr, "Too many classes (> 10) found in input data\n");
        exit(EXIT_FAILURE);
    }

    // init and train network
    somr_network_t network;
    somr_network_init(&network, features_count);

    printf("Training settings:\n");
    printf("  spread_threshold=%f\n  depth_threshold=%f\n  iters_count=%u\n  learning_rate=%f\n  orient=%s\n  seed=%u\n",
        spread_threshold, depth_threshold, iters_count, learn_rate, should_orient ? "true" : "false", seed);
    printf("Training network...\n");

    somr_network_train(&network, &dataset, learn_rate, spread_threshold, depth_threshold, iters_count, should_orient, seed);

    print_errors(&network, &dataset, seed);

    // gen image
    unsigned char *img = malloc(sizeof(unsigned char) * 3 * IMG_WIDTH * IMG_HEIGHT);
    somr_network_write_to_img(&network, img, IMG_WIDTH, IMG_HEIGHT, COLORS);

    file = fopen(png_filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Could not open %s\n", png_filename);
        exit(EXIT_FAILURE);
    }
    write_img_to_png(file, img, IMG_WIDTH, IMG_HEIGHT);
    free(img);
    img = NULL;
    fclose(file);

    // clean up
    somr_network_clear(&network);
    somr_dataset_clear(&dataset);
}
