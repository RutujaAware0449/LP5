#include <iostream>
#include <cstdlib>
#include <omp.h>

using namespace std;

// Merge two sorted halves
void merge(int* arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int* left = new int[n1];
    int* right = new int[n2];

    for (int i = 0; i < n1; i++) left[i] = arr[l + i];
    for (int i = 0; i < n2; i++) right[i] = arr[m + 1 + i];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        arr[k++] = (left[i] <= right[j]) ? left[i++] : right[j++];
    }
    while (i < n1) arr[k++] = left[i++];
    while (j < n2) arr[k++] = right[j++];

    delete[] left;
    delete[] right;
}

// Sequential Merge Sort
void sequentialMergeSort(int* arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        sequentialMergeSort(arr, l, m);
        sequentialMergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort using OpenMP tasks
void parallelMergeSort(int* arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        #pragma omp task shared(arr)
        parallelMergeSort(arr, l, m);

        #pragma omp task shared(arr)
        parallelMergeSort(arr, m + 1, r);

        #pragma omp taskwait
        merge(arr, l, m, r);
    }
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    int* a = new int[n];
    int* b = new int[n];

    for (int i = 0; i < n; i++) {
        a[i] = rand() % 10000;
        b[i] = a[i];
    }

    double start_time, end_time;

    // Sequential Merge Sort
    start_time = omp_get_wtime();
    sequentialMergeSort(a, 0, n - 1);
    end_time = omp_get_wtime();
    cout << "Sequential Merge Sort Time: " << end_time - start_time << " seconds" << endl;

    // Parallel Merge Sort
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        parallelMergeSort(b, 0, n - 1);
    }
    end_time = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << end_time - start_time << " seconds" << endl;

    delete[] a;
    delete[] b;

    return 0;
}

