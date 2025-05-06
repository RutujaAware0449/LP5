#include <iostream>
#include <cstdlib>
#include <omp.h>

using namespace std;

// Sequential Bubble Sort
void SequentialBubbleSort(int* a, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

// Parallel Odd-Even Bubble Sort
void ParallelBubbleSort(int* b, int n) {
    for (int i = 0; i < n; i++) {
        // Even phase
        #pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2) {
            if (b[j] > b[j + 1]) {
                swap(b[j], b[j + 1]);
            }
        }
        // Odd phase
        #pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2) {
            if (b[j] > b[j + 1]) {
                swap(b[j], b[j + 1]);
            }
        }
    }
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    int* a = new int[n];
    int* b = new int[n];

    // Generate random numbers and copy to both arrays
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 1000;
        b[i] = a[i];
    }

    double start_time, end_time;

    // Sequential Sort
    start_time = omp_get_wtime();
    SequentialBubbleSort(a, n);
    end_time = omp_get_wtime();
    cout << "Execution time for Sequential Bubble Sort: " << end_time - start_time << " seconds" << endl;

    // Parallel Sort
    start_time = omp_get_wtime();
    ParallelBubbleSort(b, n);
    end_time = omp_get_wtime();
    cout << "Execution time for Parallel Bubble Sort: " << end_time - start_time << " seconds" << endl;

    delete[] a;
    delete[] b;

    return 0;
}

