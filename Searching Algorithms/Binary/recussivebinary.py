def RecussiveBinarSearch(arr, x):
    start = 0
    end = len(arr)-1
    if end >= start:
        mid = start + (end - start)//2

        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            end = mid -1
            return RecussiveBinarSearch(arr, x)
        else:
            start = mid + 1
            return RecussiveBinarSearch(arr, x)
        
    return -1