# Two Pointers Search Algorithm






```
Algorithm: TwoSum(arr, target)

1. Step: Initialize n
    - Set n = length of arr

2. Step: Iterate through each element in the array
    - For i from 0 to n - 1:
  
3. Step: For each element arr[i], check every other element arr[j] that comes after it
    - For j from i + 1 to n - 1:
      
4. Step: Check if the sum of the current pair equals the target
    - If arr[i] + arr[j] == target:
        - Return True

5. Step: If no pair is found after checking all possibilities
    - Return False




```

### Reference 
1. https://www.geeksforgeeks.org/two-pointers-technique/