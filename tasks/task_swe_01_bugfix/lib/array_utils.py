def find_max_subarray(arr):
    if not arr:
        return 0
    
    # Special case for all negative numbers: return the max (largest) negative number
    if max(arr) < 0:
        return max(arr)
    
    max_ending_here = 0
    max_so_far = float('-inf')
    
    for x in arr:
        max_ending_here = max(x, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)
    
    return max_so_far