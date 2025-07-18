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

if __name__ == "__main__":
    arr = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    x = 50
    result = IterativeBinary(arr, x)
    if result != -1:
        print(f"Element found at index {result}")
    else:
        print("Element not found in the array")