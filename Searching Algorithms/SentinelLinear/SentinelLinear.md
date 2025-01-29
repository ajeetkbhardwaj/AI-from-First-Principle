# Sentinel Linear Search
In Sentinel Linear Search is similar to Classical Linear search but here we find a way to reduce the number of comparision that are made, for this we use sentinel value to avoide any out-of bounds comparisions, but no additional comparision made for index of element being searched.

We assume last element of the array is replaced with the element to be searched and then the linear search is performed on the array without checking whether the current index is inside the index range of the array or not because the element to be searched will definitely be found inside the array even if it was not present in the original array since the last element got replaced with it. So, the index to be checked will never be out of the bounds of the array.

```
Algorithm SentinelLinearSearch(arr, target)
    Input: 
        arr: an array of elements
        target: the value to search for
    Output:
        index: the index of target in arr or -1 if not found

    // Step 1: Append the target as a sentinel
    n ← length(arr)               // Get the original length of the array
    arr[n] ← target                // Set the last element as the target (sentinel)

    // Step 2: Initialize index variable
    i ← 0                          // Start index at 0

    // Step 3: Search loop
    while arr[i] ≠ target do      // Loop until we find the target
        i ← i + 1                 // Increment index

    // Step 4: Check result
    if i < n then                 // Check if we found the target within original array bounds
        return i                  // Return the index where target was found
    else
        return -1                 // Return -1 if target is not found
```

The Core concept behind the sentinel linear search is to add an extra element(sentinel value) at the end of array that matches the search target value. Due to this we can avoid the conditional check, for end of array in loop and terminate the search easly, as soon as we find the sentinel element. Which eliminates the need for a separate check for the end of the array, resulting in a slight improvement in the average case performance of the algorithm.

The Worst case performance remain same 
1. $O(n)$
2. While in average case it provides an edge compared to classical linear search
3. Best case also same for them