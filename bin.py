def binary_search(arr, x):
    l = 0
    r = len(arr) - 1
    while(l <= r):
        mid = (l + r)//2
        if (arr[mid] == x):
            return mid
        elif(x < arr[mid]):
            r = mid - 1
        else:
            l = mid + 1
    return -1

array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(binary_search(array, 11))