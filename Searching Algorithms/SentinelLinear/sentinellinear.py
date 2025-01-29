def SentinelLinearSearch(arr, target):
    """
    args: 
        arr: an array of elements
        target: the value to search for
    Output:
        index: the index of target in arr or -1 if not found
    
    """
    # 1. Append target as a sentinel value
    n = len(arr)
    last = arr[n-1]
    arr[n-1] = target
    # 2. index
    i = 0
    while(arr[i] != target):
        i += 1
        
    arr[n-1] = last
    if i < n-1 or arr[n-1] == target:
        print(target, "is present at index", i)
    else:
        print("Element not found")
   
arr = [10, 20, 180, 30, 60, 50, 110, 100, 70] 
x = 180

print(SentinelLinearSearch(arr, x))
