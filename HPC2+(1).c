#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Sequential Bubble Sort
void bubbleSortSequential(int arr[], int n) {
    int temp;
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Optimized Parallel Bubble Sort
void bubbleSortParallel(int arr[], int n) {
    if (n < 2000) {
        bubbleSortSequential(arr, n);
        return;
    }

    int temp;
    for (int i = 0; i < n; i++) {
        #pragma omp parallel for private(temp)
        for (int j = (i % 2 == 0) ? 0 : 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Merge function
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

// Sequential Merge Sort
void mergeSortSequential(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortSequential(arr, left, mid);
        mergeSortSequential(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Optimized Parallel Merge Sort
void mergeSortParallel(int arr[], int left, int right) {
    if (right - left < 2000) {
        mergeSortSequential(arr, left, right);
        return;
    }
    int mid = left + (right - left) / 2;
    #pragma omp parallel sections
    {
        #pragma omp section
        mergeSortParallel(arr, left, mid);

        #pragma omp section
        mergeSortParallel(arr, mid + 1, right);
    }
    merge(arr, left, mid, right);
}

// Function to generate random numbers
void generateRandomArray(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }
}

int main() {
    srand(time(0));
    omp_set_num_threads(4);

    int inputSizes[5];
    printf("Enter 5 input sizes: ");
    for (int i = 0; i < 5; i++) {
        scanf("%d", &inputSizes[i]);
    }
    
    printf("\n Pratyush Funde [BE A ] 41013");
    
    printf("\n+------------+---------------------+----------------------+-------------------+-------------------+---------------------+----------------------+-------------------+-------------------+\n");
    printf("| Input Size | Bubble Seq Time | Bubble Par Time | Bubble Speedup | Bubble Effi | Merge Seq Time | Merge Par Time | Merge Speedup | Merge Effi |");
    printf("\n+------------+---------------------+----------------------+-------------------+-------------------+---------------------+----------------------+-------------------+-------------------+\n");
    
    for (int t = 0; t < 5; t++) {
        int n = inputSizes[t];
        int *arr1 = (int *)malloc(n * sizeof(int));
        int *arr2 = (int *)malloc(n * sizeof(int));
        int *arr3 = (int *)malloc(n * sizeof(int));

        generateRandomArray(arr1, n);
        for (int i = 0; i < n; i++) {
            arr2[i] = arr1[i];
            arr3[i] = arr1[i];
        }

        double start_time, end_time;

        start_time = omp_get_wtime();
        bubbleSortSequential(arr1, n);
        end_time = omp_get_wtime();
        double bubbleSortSeqTime = end_time - start_time;

        start_time = omp_get_wtime();
        bubbleSortParallel(arr1, n);
        end_time = omp_get_wtime();
        double bubbleSortParTime = end_time - start_time;

        start_time = omp_get_wtime();
        mergeSortSequential(arr2, 0, n - 1);
        end_time = omp_get_wtime();
        double mergeSortSeqTime = end_time - start_time;

        start_time = omp_get_wtime();
        mergeSortParallel(arr3, 0, n - 1);
        end_time = omp_get_wtime();
        double mergeSortParTime = end_time - start_time;

        double bubbleSortSpeedup = bubbleSortSeqTime / bubbleSortParTime;
        double mergeSortSpeedup = mergeSortSeqTime / mergeSortParTime;
        double bubbleSortEfficiency = bubbleSortSpeedup / 4;
        double mergeSortEfficiency = mergeSortSpeedup / 4;

        printf("| %8d | %16.4f | %16.4f | %14.4f | %10.4f | %16.4f | %16.4f | %14.4f | %10.4f |\n",
       n, bubbleSortSeqTime, bubbleSortParTime, bubbleSortSpeedup, bubbleSortEfficiency,
       mergeSortSeqTime, mergeSortParTime, mergeSortSpeedup, mergeSortEfficiency);


        free(arr1);
        free(arr2);
        free(arr3);
    }
    
    printf("+------------+---------------------+----------------------+-------------------+-------------------+---------------------+----------------------+-------------------+-------------------+\n");
    return 0;
}


