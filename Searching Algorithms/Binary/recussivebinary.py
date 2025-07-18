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

if __name__ == "__main__":
    arr = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    x = 80
    result = RecussiveBinarSearch(arr, x)
    if result == -1:
        print("Element not found in the array")
    else: 
        print(f"Element found at the index {result}")