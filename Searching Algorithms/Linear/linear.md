# Linear Search 
Given an array of size n and an element say x, we needed to check whether element x is present in the array or not. if yes then find at which place.

Input : arr[] = [1, 2, 3, 4, 5] and x = 4
Ouptut : 3

Core Concept : Given an array and element that we have to search, we will iterate over all the element of the array and check if current element of the array is equal to the target element, if we find then return the index else move to the next index of array i.e in next loop and after end of loop if we don't find any element equal to the target element then we return -1 which indicate that element is not present in the array.

# Linear Search Algorithm

**Algorithm : LinearSearch(A, N, VAL)**

1. Step : Initialize
    - Set `pos = -1`
2. Step: Initialize
    - Set `i = 1`
3. Step : Repeat Step 4 while `i <= N`
4. Step :
    - If `A[i] = VAL`
        - Set `pos = i`
        - Print `pos`
        - Go to Step 6
    - [End of If]
    - Set `i = i + 1`
    - [End of Loop]
5. Step :
    - If `pos = -1`
         - Print "Value is not present in the array"
    - [End of If]
6. Step : Exit

**Complexity of Linear Search**
1. Time Complexity
    - Best Case : $O(1)$
    - Avg Case : $O(n)$
    - Worst Case : $O(n)$

2. Space Complexity 
   - $O(1)$ because only veriable to iterate through the list of element used.

### Interview Questions
1. What are applications of the Linear Search Algorithms
   - Find the any elemenent in an unsorted array(list)
   - We prefer to use the linear search compared to binary search when we have smaller data(array)
   - In linked list data structure implementation we commonly use the linear search algorithm to find the element within linked list.
   - It is prefere to used because it's implementation and idea are simple.
2. What are advatanges of the Linear Search Algorithms
   - 

3. What are disadvantages of the Linear Search ?

4. When we should prefere to use the Linear Search Algorithm ? 
- Small Datasets: Linear search is best suited for small datasets where the overhead of more complex algorithms would not provide significant benefits.
- Unsorted Data: It is ideal when searching through unsorted data or when there are frequent insertions and deletions that prevent sorting.
- Contiguous Memory Storage: Use linear search when dealing with datasets stored in contiguous memory locations, such as arrays, where each element can be accessed sequentially.

5. What is Linear Search Algorithm ?
6. How does Linear Search Algorithm work ?
7. What is time complexity of Linear Search Algorithm ?
8. When Linear Search Algorithm is prefered to used over other searching algorithm ?
9. How do you implement Linear Search Algorithm in programming language like python, c, c++, etc ?
10. Can Linear Search Algorithm applied to other data structures ?
11. Is Linear Search Algorithm suitable of sorted arrays ?
12. What are some real-world applications of linear search algorithmn?


### References
1. https://www.geeksforgeeks.org/linear-search/