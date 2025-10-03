def find_max_subarray(arr):
    if not arr:
        return 0
    
    if len(arr) == 1:
        return arr[0]
    
    max_ending_here = arr[0]  # Start with first element
    max_so_far = arr[0]  # Also start max_so_far with first element
    
    for x in arr[1:]:  # Start from second element
        max_ending_here = max(x, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)
    
    return max_so_far