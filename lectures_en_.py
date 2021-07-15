# Algorithms from IB002 course at MUNI
# written by Daniel Rozehnal, 514184

from random import randint

## 1) Time complexity of Algorithms

# LINEAR SEARCH
# All of the below algorithms have linear complexity in worst case O(n)
# and only difference is in constant factors

# Naive linear search A[0 ... n-1]
def basic_linear_search(array, key):
    index = -1
    for i in range(len(array)):
        if array[i] == key:
            index = i
    return index

# Return when our element is found
def better_linear_search(array, key):
    for i in range(len(array)):
        if array[i] == key:
            return i
    return -1

# Optimalization with the use of stop
# thus meaning in while loop there is only 1 test array[i] == key
# and above implementations use 2 tests
def even_better_linear_search(array, key):
    last = array[len(array) - 1]
    array[len(array) - 1] = key
    i = 0
    while array[i] != key:
        i += 1
    if i < (len(array)- 1) or last == key:
        array[len(array) - 1] = last  # array in original state
        return i
    else:
        array[len(array) - 1] = last
        return -1

# Insert sort
# Complexity O(n^2)
def insert_sort(array) -> None:
    for j in range(len(array)):
        key = array[j]
        i = j - 1
        # set boundary for start of array
        while i >= 0 and array[i] > key:  
            array[i + 1] = array[i]
            i -= 1
        array[i + 1] = key

## 2) Design of algorithms, divide and conquer

# Finding maximum and minimum in sequence of numbers

# Iterative solution, need for 2 comparisons in a loop
def minmax_iterative(array):
    current_max = array[0]
    current_min = array[0]
    for i in range(1, len(array) - 1):
        if array[i] > current_max:
            current_max = array[i]
        if array[i] < current_min:
            current_min = array[i]
    return current_min, current_max

# Recursive solution with divide and conquer technique
# Better performance than iterative one
def minmax_rec(array, l, r):
    # base for 1 element
    if r == l:
        return array[l], array[r]
    # base for 2 elements
    if r == l + 1:
        return min(array[l], array[r]), max(array[l], array[r])
    # Recursive call
    if r > l + 1:
        l_min, l_max = minmax_rec(array, l, (l+r) // 2)
        r_min, r_max = minmax_rec(array, (l+r) // 2 + 1, r)
    return min(l_min, r_min), max(l_max, r_max)

# Problem of maximum subsequence

# This function tries to find maximum subsequence at the cross section
# in the middle of 2 recursive calls for each side
def cross(array, low, mid, high):
    # highest subsequence on the left
    left_sum = array[mid]
    total = array[mid]
    left_index = mid
    for i in range(mid - 1, low, -1):
        total += array[i]
        if total > left_sum:
            left_sum = total
            left_index = i
    # highest subsequence on the right
    right_sum = array[mid+1]
    total = array[mid+1]
    right_index = mid+1
    for i in range(mid + 2, high):
        total += array[i]
        if total > right_sum:
            right_sum = total
            right_index = i

    return left_index, right_index, left_sum + right_sum

def max_sum(array, low, high):
    # base for one element
    if low == high:
        return low, high, array[low]
    mid = (low + high) // 2
    (left_i, left_j, left_s) = max_sum(array, low, mid)
    (right_i, right_j, right_s) = max_sum(array, mid+1, high)
    (cross_i, cross_j, cross_s) = cross(array, low, mid, high)

    # returning maximum subsequence
    if left_s > right_s and left_s> cross_s:
        return left_i, left_j, left_s
    if left_s < right_s and right_s > cross_s:
        return right_i, right_j, right_s
    return cross_i, cross_j, cross_s

## 3) Sorting algorithms

# Merge sort

# Realized with divide and conquer technique, not in-situ
def merge_sort(array, left, right):
    # for array of lenght >= 1
    if left < right:
        mid = (left + right) // 2
        merge_sort(array, left, mid)
        merge_sort(array, mid+1, right)
        merge(array, left, mid, right)
        

def merge(array, left, mid, right):
    # creating aux array as copy of original array
    aux = [0 for _ in range(len(array))]
    for k in range(left, right+1):
        aux[k] = array[k]
    # going through the array and comparison of its element
    # while sorting elements on a correct position
    i = left
    j = mid + 1
    for k in range(left, right + 1):
        # not only comparing, we also ensure that we are not
        # at the end of one subsequence
        if i <= mid and (j > right or aux[i] <= aux[j]):
            array[k] = aux[i]
            i += 1
        else:
            array[k] = aux[j]
            j += 1

# Problem of inversions (Inversion count)

def inversions(array, left, right):
    # for sequence of length >= 1
    if left < right:
        mid = (left + right) // 2
        count_left = inversions(array, left, mid)
        count_right = inversions(array, mid+1, right)
        count = merge_count(array, left, mid, right)
        return count + count_left + count_right
    return 0

def merge_count(array, left, mid, right):
    # creation of auxilary array
    aux = [0 for _ in range(len(array))]
    for k in range(left, right + 1):  
        aux[k] = array[k]
    # going through elements and comparing them
    count = 0
    i = left
    j = mid + 1
    for k in range(left, right + 1):
        if i <= mid and (j > right or aux[i] <= aux[j]):
            array[k] = aux[i]
            i += 1
        else:
            array[k] = aux[j]
            count += (mid - i) + 1
            j += 1
    # returning count of inversed numbers
    return count

# Quicksort

# Sorting with pivot, very useful in practice
# Complexity O(n^2), but on average O(n log n), is in-situ
def quicksort(array, left, right):
    # sequence of length >= 1
    if left < right:
        m = partition(array, left, right)  # index pivotu
        quicksort(array, left, m-1)
        quicksort(array, m+1, right)
        
# divides and sorts sequence with pivot value, returns pivot index
def partition(array, left, right):
    pivot = array[right]
    i = left - 1
    for j in range(left, right+1):
        if array[j] <= pivot:
            i += 1
            swap(array, i, j)
    return i

# simple element swapping
def swap(array, a_i, b_i):
    array[a_i], array[b_i] = array[b_i], array[a_i]

## 4) Heap sort

# For represesentation of heap (or binary heap), we use python array
# where first element is the root of the heap.

def parent(i):
    if i > 0:
        return (i-1) // 2
    return None

def left(i):
    return 2 * i + 1

def right(i):
    return 2 * i + 2

# This function fixes array from a given index from bottom up
# to the root, applied to all elements (excluding leafs)
# we can build entire (maximal) binary heap represented by an array.
def heapify(array, i):
    largest = i  # index největšího prvku
    if left(i) <= len(array)-1 and array[left(i)] > array[i]:
        largest = left(i)
    if right(i) <= len(array)-1 and array[right(i)] > array[largest]:
        largest = right(i)
    # nutnost opravy
    if largest != i:
        swap(array, i, largest)
        heapify(array, largest)

def build_heap(array):
    size = len(array)
    for i in range(size//2, -1, -1):
        heapify(array, i)

# Complexity of heap_sort is O(n log n)
# and should be in-situ (which is not implemented here)
def heapsort(array):
    build_heap(array)
    new_array = []
    for i in range(len(array)-1, -1, -1):
        swap(array, i, 0)
        el = array.pop()
        new_array.append(el)
        heapify(array, 0)

    # because of python, we cannot make array decrease size
    # like A.size -= 1, thus we need a way to get rid of elements
    # with using array.pop().
    # This could probably be fixed with creating new Class or smth.

    for i in range(len(new_array)-1, -1, -1):
        array.append(new_array[i])

# Thanks to the heap, we can effectively create Priority queue
# while requesting effective: Insert, Maximum, ExtractMax and IncreaseKey

def pq_maximum(array):
    return array[0]

# Swapping root with last element and then repairing the heap
# with use of heapify()
def pq_extractmax(array):
    if len(array) == 0:
        return None

    maximum = array[0]
    array[0] = array[len(array) - 1]
    array.pop()
    heapify(array, 0)
    return maximum


# Increases key of node with given index, which
# needs a path to root to be fixed
def pq_increase_key(array, i, key):
    array[i] = key
    while i > 0 and array[parent(i)] < array[i]:
        swap(array, i, parent(i))
        i = parent(i)

def pq_insert(array, key):
    array.append(0)
    pq_increase_key(array, len(array)-1, key)

# Sorting in linear time O(k + n)

# Counting sort
# Input condition: sequence of known range of integers
# ex. 0 ... k where k is fixed known number
# Not in situ, but it is stable

def countingsort(in_array, out_array, k):
    c = [0 for _ in range(k+1)]
    # on index of number, we put the count of the number in in_array
    for i in range(len(in_array)):
        c[in_array[i]] += 1
    # we aquire the count of previous numbers in sequence
    for i in range(1, k+1):
        c[i] += c[i-1]
    # from back to front we take numbers from in_array and put them in
    # their correct position, while using aux array C
    for j in range(len(in_array)-1, -1, -1):
        value = in_array[j]
        out_array[c[value]-1] = in_array[j]
        c[in_array[j]] -= 1

## 5) Binary search trees

# Total order is defined over nodes (keys) in tree
# Operations: Search, Min., Max., Predecessor, Successor, Insert, Delete

# Representation of Node in Binary search tree
class Node:
    def __init__(self, key) -> None:
        self.key = key
        self.parent = None
        self.right = None
        self.left = None

# Representation of tree by pointing to its root
class BinarySearchTree:
    def __init__(self) -> None:
        self.root = None

# Ways to go through a tree

def inorder(x: Node):
    if x is not None:
        inorder(x.left)
        print(x.key)
        inorder(x.right)

def preorder(x: Node):
    if x is not None:
        print(x.key)
        inorder(x.left)
        inorder(x.right)

def postorder(x: Node):
    if x is not None:
        inorder(x.left)
        inorder(x.right)
        print(x.key)

# Operations

def BVS_search(x, k):
    if x is None or k == x.key:
        return x
    if k < x.key:
        return BVS_search(x.left, k)
    return BVS_search(x.right, k)

def BVS_minimum(x):
    if x is None:
        return None
    while x.left is not None:
        x = x.left
    return x

def BVS_maximum(x):
    if x is None:
        return None
    while x.right is not None:
        x = x.right
    return x

# Succesor is the minimal element of right subtree
# if not defined, we go through parents and search a node
# that x is the predecessor of.
def BVS_successor(x):
    if x.right is not None:
        return BVS_minimum(x.right)

    # going up to the root
    y = x.parent
    while y is not None and x == y.right:
        x = y
        y = x.parent
    return y

# Anologically with the predecessor
def BVS_predeccessor(x):
    if x.left is not None:
        return BVS_maximum(x.left)

    # going up to the root
    y = x.parent
    while y is not None and x == y.left:
        x = y
        y = x.parent
    return y

def BVS_insert(T, new_node):
    current = T.root
    parent = T.root
    while current is not None:
        parent = current
        if new_node.key < current.key:
            current = current.left
        else:
            current = current.right
    new_node.parent = parent
    if parent is None:
        T.root = new_node
    else:
        if new_node.key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node

def BVS_delete(T, z):
    # if it does not have one son, we apply transplant
    if z.left is None:
        transplant(T, z, z.right)
    elif z.right is None:
        transplant(T, z, z.left)
    else:  # search for successor (or predecessor)
        y = minimum(z.right)
        if y.parent is not z:  # not a direct son
            transplant(T, y, y.right)  # minimum does not have left son
            y.right = z.right
            z.right.parent = y
        transplant(T, z, y)
        y.left = z.left
        z.left.parent = y 

# In tree T, switches node U with node V
# and also treats case of v == None
def transplant(T, u, v):
    if u.parent is None:
        T.root = v
    else:
        if u.parent.left == u:
            u.parent.left = v
        else:
            u.parent.right = v
            
    if v is not None:
        v.parent = u.parent

# Binary search trees can be further modified as interval trees etc.

## 6) Red-black trees

# The goal is to build balanced binary search tree
# and retain effective operations over this structure
# ex. we dont want Search to have O(n) complexity like with normal BST
# RULES:
# - nodes are either red or black
# - root is black
# - every node has 2 children, leafs are null (None)
# - leafs have black color
# - parent and child cannot have both red color (red is between 2 black nodes)
# - on every path between root -> leaf, there must be the same number of black nodes

# black height = number of black nodes from root -> leaf

# aliases
BLACK = 0
RED = 1
class RB_Node:
    def __init__(self, key) -> None:
        self.key = key
        self.color = RED  # new red node does not change black-height
        self.parent = None
        self.right = None
        self.left = None

# Representaiton of tree by pointing to its root
class RedBlackTree:
    def __init__(self) -> None:
        self.root = None

# left / right rotation of node x
def left_rotate(T, x):
    y = x.right
    if y is None:
        return
    x.right = y.left
    if y.left is not None:
        y.left.parent = x
    y.parent = x.parent
    if x.parent is None:
        T.root = y
    else:
        if x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y

    y.left = x
    x.parent = y

# Insertion can break R-B tree structure

# TODO RB_INSERT(T, n) 3 cases solution

# TODO RB_REMOVE(T, z) Different number of sons solution



# ---===== UNIT TESTING =====---
def run_tests():
    # linear searches
    a = [1, 2, 3, 4, 5, 6]
    assert(basic_linear_search(a, 3) == 2)
    assert(basic_linear_search(a, 7) == -1)
    assert(better_linear_search(a, 6) == 5)
    assert(better_linear_search(a, 0) == -1)
    assert(even_better_linear_search(a, 1) == 0)
    assert(even_better_linear_search(a, 69) == -1)

    # insert sort
    insert_sort(a)
    assert(a == sorted(a))
    
    a = generate_array(20)
    insert_sort(a)
    assert(a == sorted(a))

    a = generate_array(50)
    insert_sort(a)
    assert(a == sorted(a))

    # minmax
    a = [42, 69, 100, 7, 66, 81, 10, 10, 3, 77]
    assert(minmax_iterative(a) == (3, 100))
    assert(minmax_rec(a, 0, len(a) - 1) == (3, 100))

    # max sum
    a = [20, 10, -1, 69, 100]
    assert(max_sum(a, 0, (len(a) - 1)) == (3, 4, 169))
    a = [20, 10, -20, 69, 50, -50, 500, -42]
    assert(max_sum(a, 0, (len(a) - 1)) == (3, 6, 569))

    # merge sort
    a = [420, 42, 69, 0, 10, -20, -5]
    merge_sort(a, 0, (len(a)-1))
    assert(a == sorted(a))

    # merge count
    a = [1, 2, 3, 4, 6, 5, 4]
    assert(inversions(a, 0, (len(a)-1)) == 3)
    a = [1, 2, 3, 4, 5]
    assert(inversions(a, 0, (len(a)-1)) == 0)
    a = [9, 8, 10, 11, 12, 13]
    assert(inversions(a, 0, (len(a)-1)) == 1)

    # quicksort
    a = generate_array(20)
    quicksort(a, 0, (len(a)-1))
    assert(a == sorted(a))

    a = generate_array(50)
    quicksort(a, 0, (len(a)-1))
    assert(a == sorted(a))

    # build_heap
    a = generate_array(20)
    build_heap(a)
    is_valid_heap(a)

    a = generate_array(50)
    build_heap(a)
    is_valid_heap(a)


    # heapsort
    a = generate_array(20)
    heapsort(a)
    assert(a == sorted(a))

    a = generate_array(50)
    heapsort(a)
    assert(a == sorted(a))

    # prioritní fronta (priority queue)
    a = generate_array(20)
    build_heap(a)
    
    # pQ: maximum
    assert(pq_maximum(a) == a[0])
    start_len = len(a)
    current_max = max(a)
    
    # pQ: extract max
    assert(pq_extractmax(a) == current_max)
    assert(start_len - 1 == len(a))
    is_valid_heap(a)

    # pQ: increase key
    pq_increase_key(a, 5, 130)
    is_valid_heap(a)

    pq_increase_key(a, 6, 140)
    is_valid_heap(a)

    pq_increase_key(a, 1, 150)
    is_valid_heap(a)

    # pQ: insert
    pq_insert(a, -42)
    is_valid_heap(a)

    pq_insert(a, 27)
    is_valid_heap(a)

    pq_insert(a, 200)
    is_valid_heap(a)

    # counting sort
    a = generate_positive_array(20)
    out_a = [0 for _ in range(len(a))]
    countingsort(a, out_a, 100)
    assert(out_a == sorted(a))
    assert(len(out_a) == len(a))

    a = generate_positive_array(100)
    out_a = [0 for _ in range(len(a))]
    countingsort(a, out_a, 100)
    assert(out_a == sorted(a))
    assert(len(out_a) == len(a))
    
    # BVS insert
    a = Node(10)
    tree = BinarySearchTree()
    BVS_insert(tree, a)
    is_valid_BVS(tree.root)
    assert(tree.root is a)
    b = Node(20)
    c = Node(15)
    d = Node(40)
    e = Node(5)
    BVS_insert(tree, b)
    BVS_insert(tree, c)
    BVS_insert(tree, d)
    BVS_insert(tree, e)
    assert(tree.root.right == b)
    assert(tree.root.right.left == c)
    assert(tree.root.right.right == d)
    assert(tree.root.right.parent == tree.root)
    assert(tree.root.right.left.parent == tree.root.right)
    is_valid_BVS(tree.root)

    # BVS delete

    BVS_delete(tree, d)
    assert(tree.root.right.right is None)
    is_valid_BVS(tree.root)

    # BVS minimum maximum

    assert(BVS_minimum(tree.root).key == 5)
    assert(BVS_maximum(tree.root).key == 20)

    # BVS search

    assert(BVS_search(tree.root, 5) == e)
    assert(BVS_search(tree.root, 42) is None)
    
    print("All tests passed")

# ---===== END OF UNIT TESTS =====---
    
# testing heap properties
def is_valid_heap(a):
    for i in range(len(a)):
        l = left(i)
        r = right(i)
        p = parent(i)

        if l < len(a):
            assert(a[l] <= a[i])
        if r < len(a):
            assert(a[r] <= a[i])
        if p is not None:
            assert(a[p] >= a[i])

# test of binary search tree properties
def is_valid_BVS(x):
    if x is None:
        return None

    if x.left is not None:
        assert(x.left.key < x.key)
    if x.right is not None:
        assert(x.right.key > x.key)
    if x.parent is not None:
        if x.parent.left is x:
            assert(x.key < x.parent.key)
        else:
            assert(x.key > x.parent.key)

    # going through whole tree
    is_valid_BVS(x.left)
    is_valid_BVS(x.right)
    

# generating random array of length = size
def generate_array(size):
    array = [0 for _ in range(size)]
    for i in range(size):
        array[i] = randint(-100, 100)
    assert(size == len(array))
    return array

# generates random array in interval <0, 100>
def generate_positive_array(size):
    array = [0 for _ in range(size)]
    for i in range(size):
        array[i] = randint(0, 100)
    assert(size == len(array))
    return array
        

run_tests()
