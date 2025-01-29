def IterativeBinary(arr, x):
    start = 0
    end = len(arr)-1
    while(start <= end):
        mid = start + (end - start) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            start = mid + 1
        else:
            end = mid - 1
    return -1
    