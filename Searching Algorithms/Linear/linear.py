def LinearSearch(arr, target):
    n = len(arr)
    for i in range(0, n):
        if arr[i] == target:
            return i
        
    return -1

arr = [2, 3, 4, 10, 40]
x = 10
print(LinearSearch(arr, x))