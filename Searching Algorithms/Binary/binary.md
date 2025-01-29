# Binary Search Algorithm





```
Iterative Algorithm: BinarySearch(arr, low, high, x)

1. Step: While low <= high
    - Set mid = low + (high - low) // 2

2. Step: Check if arr[mid] == x
    - If true, return mid (the index of the element)

3. Step: Check if arr[mid] < x
    - If true, set low = mid + 1 (ignore the left half)

4. Step: Else
    - Set high = mid - 1 (ignore the right half)

5. Step: If the element is not found
    - Return -1
```
Hi Bro

```
Recussive Algorithm: BinarySearch(arr, low, high, x)

1. Step: Check base case
    - If high >= low:
        - Set mid = low + (high - low) // 2

2. Step: Check if arr[mid] == x
    - If true, return mid (the index of the element)

3. Step: Check if arr[mid] > x
    - If true, return BinarySearch(arr, low, mid - 1, x) (search in the left subarray)

4. Step: Else
    - Return BinarySearch(arr, mid + 1, high, x) (search in the right subarray)

5. Step: If the element is not found
    - Return -1
```


### Reference 
1. https://www.geeksforgeeks.org/binary-search/#iterative-binary-search-algorithm