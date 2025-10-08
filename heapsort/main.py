from sort_algorithms.bubble_sort import bubble_sort
from sort_algorithms.insertion_sort import insertion_sort
from sort_algorithms.selection_sort import selection_sort

if __name__ == "__main__":
    test_arrays = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 2, 9, 1, 5, 6],
        [3, 0, -1, 8, 7, 2]
    ]

    print("Testando Bubble Sort:")
    for arr in test_arrays:
        print(f"Original: {arr}")
        print(f"Ordenado: {bubble_sort(arr.copy())}\n")

    print("Testando Insertion Sort:")
    for arr in test_arrays:
        print(f"Original: {arr}")
        print(f"Ordenado: {insertion_sort(arr.copy())}\n")

    print("Testando Selection Sort:")
    for arr in test_arrays:
        print(f"Original: {arr}")
        print(f"Ordenado: {selection_sort(arr.copy())}\n")
