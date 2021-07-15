# Algoritmy z předmětu IB002 (jaro 2021)
# sepsal Daniel Rozehnal, 514184

from random import randint

## 1) Složitost algoritmů

# LINEÁRNÍ VYHLEDÁVÁNÍ
# všechny algoritmy v nejhroším čase mají linární složitost O(n)
# rozdíl je v konstantních faktorech

# naivní lineární hledání v seznamu A[0 ... n-1]
def basic_linear_search(array, key):
    index = -1
    for i in range(len(array)):
        if array[i] == key:
            index = i
    return index

# return při nalezení prvku
def better_linear_search(array, key):
    for i in range(len(array)):
        if array[i] == key:
            return i
    return -1

# optimalizace s použitím zarážky na ko
# ve while cyklu probíhá pouze 1 test zda array[i] == key
# implementace výše používají v cyklu 2 testy
def even_better_linear_search(array, key):
    last = array[len(array) - 1]
    array[len(array) - 1] = key
    i = 0
    while array[i] != key:
        i += 1
    if i < (len(array)- 1) or last == key:
        array[len(array) - 1] = last  # seznam v původním stavu
        return i
    else:
        array[len(array) - 1] = last  # seznam v původním stavu
        return -1

# řazení vkládáním (insert sort)
# složitost O(n^2)
def insert_sort(array) -> None:
    for j in range(len(array)):
        key = array[j]
        i = j - 1
        # nejdeme před index 0 a předchozí klíč je větší
        while i >= 0 and array[i] > key:  
            array[i + 1] = array[i]
            i -= 1
        array[i + 1] = key

## 2) Návrh algoritmů, rozděl a panuj

# Hledání minima a maximima v posloupnosti čísel

# iterativní řešení, nutno 2 porovnání pro každý prvek
def minmax_iterative(array):
    current_max = array[0]
    current_min = array[0]
    for i in range(1, len(array) - 1):
        if array[i] > current_max:
            current_max = array[i]
        if array[i] < current_min:
            current_min = array[i]
    return current_min, current_max

# rekurzivní řešení, metoda rozděl a panuj
# lepší složitost než iterativní řešení
def minmax_rec(array, l, r):
    # báze pro 1 prvek
    if r == l:
        return array[l], array[r]
    # báze pro 2 prvky
    if r == l + 1:
        return min(array[l], array[r]), max(array[l], array[r])
    # rekurzivní volání
    if r > l + 1:
        l_min, l_max = minmax_rec(array, l, (l+r) // 2)
        r_min, r_max = minmax_rec(array, (l+r) // 2 + 1, r)
    return min(l_min, r_min), max(l_max, r_max)

# problém maximální podposloupnosti

# hledá nejvyšší podposloupnost na hranici 2 podposloupností
# na které poté bude problém rekurzivně rozdělen
def cross(array, low, mid, high):
    # největší podposloupnost vlevo
    left_sum = array[mid]
    total = array[mid]
    left_index = mid
    for i in range(mid - 1, low, -1):
        total += array[i]
        if total > left_sum:
            left_sum = total
            left_index = i
    # největší podposloupnost vpravo
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
    # báze pro jednoprvkovou posloupnost
    if low == high:
        return low, high, array[low]
    mid = (low + high) // 2
    (left_i, left_j, left_s) = max_sum(array, low, mid)
    (right_i, right_j, right_s) = max_sum(array, mid+1, high)
    (cross_i, cross_j, cross_s) = cross(array, low, mid, high)

    # vracíme největší posloupnost
    if left_s > right_s and left_s> cross_s:
        return left_i, left_j, left_s
    if left_s < right_s and right_s > cross_s:
        return right_i, right_j, right_s
    return cross_i, cross_j, cross_s

## 3) Řadící algoritmy (sorting algorithms)

# Řazení sléváním (merge sort)

# realizováno technikou rozděl a panuj, není in situ
def merge_sort(array, left, right):
    # pro posloupnost velikosti > 1
    if left < right:
        mid = (left + right) // 2
        merge_sort(array, left, mid)
        merge_sort(array, mid+1, right)
        merge(array, left, mid, right)
        

def merge(array, left, mid, right):
    # tvorba pomocného seznamu jako kopie array
    aux = [0 for _ in range(len(array))]
    for k in range(left, right+1):
        aux[k] = array[k]
    # průchod a porovnání prvků v pomocném poli
    # přičemž řadíme prvky na správnou pozici v původním seznamu
    i = left
    j = mid + 1
    for k in range(left, right + 1):
        # kromě porovnání ošetřujeme, že jsme došli
        # na konec jedné z podposloupností (levé či právé)
        if i <= mid and (j > right or aux[i] <= aux[j]):
            array[k] = aux[i]
            i += 1
        else:
            array[k] = aux[j]
            j += 1

# problém inverzí (inversion count)

def inversions(array, left, right):
    # pro posloupnost velikosti > 1
    if left < right:
        mid = (left + right) // 2
        count_left = inversions(array, left, mid)
        count_right = inversions(array, mid+1, right)
        count = merge_count(array, left, mid, right)
        return count + count_left + count_right
    return 0

def merge_count(array, left, mid, right):
    # tvorba pomocného seznamu
    aux = [0 for _ in range(len(array))]
    for k in range(left, right + 1):  
        aux[k] = array[k]
    # průchod a porovnání prvků v pomocném poli
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
    return count

# Quicksort

# řazení s použitím pivota, velmi používáné v praxi
# O(n^2), ale průměrná složitost O(n log n), je in situ
def quicksort(array, left, right):
    # posloupnost velikosti > 1
    if left < right:
        m = partition(array, left, right)  # index pivotu
        quicksort(array, left, m-1)
        quicksort(array, m+1, right)
        
# rozdělení na 2 podle pivotu, vrací index pivota
def partition(array, left, right):
    pivot = array[right]
    i = left - 1
    for j in range(left, right+1):
        if array[j] <= pivot:
            i += 1
            swap(array, i, j)
    return i

# jednoduchá záměna prvků
def swap(array, a_i, b_i):
    array[a_i], array[b_i] = array[b_i], array[a_i]
    # šlo by implementovat více způsoby

## 4) Řazení haldou (heapsort)

# pro reprezentaci haldy (popř. binární hlady) používáme pole
# kde je kořenem 1. prvek (index 0)

def parent(i):
    if i > 0:
        return (i-1) // 2
    return None

def left(i):
    return 2 * i + 1

def right(i):
    return 2 * i + 2

# vstupní podmínka: binární stromy s kořeny left(i) a right(i)
# již splňují vlastnosti binární haldy

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

# složitost heap_sort je O(n log n), v jiné implementaci by byla in situ
def heapsort(array):
    build_heap(array)
    new_array = []
    for i in range(len(array)-1, -1, -1):
        swap(array, i, 0)
        el = array.pop()
        new_array.append(el)
        heapify(array, 0)

    # kvůli pythonu nelze provést na poli příkaz A.size -= 1
    # místo vyhazování prvku pomocí array.pop()

    # šlo by implementovat pomocí předávání parametru array_size do fcí výše
    for i in range(len(new_array)-1, -1, -1):
        array.append(new_array[i])

# pomocí haldy lze realizovat Prioritní frontu (priority queue)
# požadujeme efektivní Insert, Maximum, ExtractMax a IncreaseKey

def pq_maximum(array):
    return array[0]

# záměna kořene a posledního uzlu
# na konci je potřeba strom opravit pomocí heapify
def pq_extractmax(array):
    if len(array) == 0:
        return None

    maximum = array[0]
    array[0] = array[len(array) - 1]
    array.pop()
    heapify(array, 0)
    return maximum

# zvýší klíč uzlu s daným indexem
# nutno opravit cestu po rodičích ke kořeni
def pq_increase_key(array, i, key):
    array[i] = key
    while i > 0 and array[parent(i)] < array[i]:
        swap(array, i, parent(i))
        i = parent(i)

def pq_insert(array, key):
    array.append(0)
    pq_increase_key(array, len(array)-1, key)

# Řazení v lineárním čase O(k + n)

# Řazení počítáním (Counting sort)
# vstupní podmínka: posloupnost obsahuje čísla ze známého intervalu
# např. 0 ... k kde k je fixně dané přirozené číslo
# kvůli využití pomocných polí není in situ, ale je stabilní

def countingsort(in_array, out_array, k):
    c = [0 for _ in range(k+1)]
    # na index čísla vložíme počet výskytu tohoto čísla v in_array
    for i in range(len(in_array)):
        c[in_array[i]] += 1
    # pro každý index zjistíme, kolik menších čísel se vyskytuje v in_array
    for i in range(1, k+1):
        c[i] += c[i-1]
    # odzadu bereme čísla z in_array a řadíme je na korektní pozice
    # pomocí pomocného pole C a počtu čísel, které před ně náleží
    for j in range(len(in_array)-1, -1, -1):
        value = in_array[j]
        out_array[c[value]-1] = in_array[j]
        c[in_array[j]] -= 1

## 5) Vyhledávací stromy (binární vyhledávací stromy a.k.a. BVS)

# nad prvky množiny stromu je definované úplné uspořádání
# operace: Search, Min., Max., Predecessor, Successor, Insert, Delete

# reprezentace uzlu v BVS stromě
class Node:
    def __init__(self, key) -> None:
        self.key = key
        self.parent = None
        self.right = None
        self.left = None

# reprezentace stromu ukazatalem na jeho kořen
class BinarySearchTree:
    def __init__(self) -> None:
        self.root = None

# průchody stromem (a)

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

# operace BVS

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

# minimální prvek pravého podstromu
# nebo pokud není x.right, tak hledáme node, pro který je x předchůdce
def BVS_successor(x):
    if x.right is not None:
        return BVS_minimum(x.right)

    # musíme jít nahoru ke kořeni
    y = x.parent
    while y is not None and x == y.right:
        x = y
        y = x.parent
    return y

# analogicky k successor
def BVS_predeccessor(x):
    if x.left is not None:
        return BVS_maximum(x.left)

    # musíme hledat směrem ke kořeni
    y = x.parent
    while y is not None and x == y.left:
        x = y
        y = x.parent
    return y

# vkládání klíče key do BVS T
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
    # pokud nemá jednoho ze synů, aplikujeme transplantaci
    if z.left is None:
        transplant(T, z, z.right)
    elif z.right is None:
        transplant(T, z, z.left)
    else:  # hledání náhradníka (následovníka)
        y = minimum(z.right)
        if y.parent is not z:  # není přímý syn
            transplant(T, y, y.right)  # minimum nemá levého syna
            y.right = z.right
            z.right.parent = y
        transplant(T, z, y)
        y.left = z.left
        z.left.parent = y 
    
# zamění ve stromě T uzel u za uzel v (zavěsí jeho podstrom místo u)
# ošetřuje i případ pro (v == None)
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

# BVS lze modifikovat např. jako intervalový strom
# kde bude struktura seřazená podle začátku intervalu


## 6) červeno černé stromy (red-black trees)

# cíl je budovat více balancovaný binární vyhledávací strom
# a udržet efektivní operace nad BVS (zamezit O(n) pro vyhledávání)
# PRAVIDLA:
# uzly jsou červené nebo černé
# kořen je černý
# každý uzel má 2 syny, listy jsou nil (None)
# listy (nil) mají černou barvu
# otec a syn nemůžou mít červenou barvu (červený uzel je mezi černými)
# na každé cestě z kořene do listu musí být stejný počet černých uzlů

# černá výška bh(x)= počet černých uzlů na cestě z x do listu (bez x)

# aliases
BLACK = 0
RED = 1
class RB_Node:
    def __init__(self, key) -> None:
        self.key = key
        self.color = RED  # červená nám neporuší černou výšku
        self.parent = None
        self.right = None
        self.left = None

# reprezentace stromu ukazatalem na jeho kořen
class RedBlackTree:
    def __init__(self) -> None:
        self.root = None

# levá (resp. pravá) rotace, pomocí které změníme výšku x
# tím, že zaměníme x za jeho pravého syna y, x.right nahradíme x.right.left

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

# vkládání může narušit vlastnosti červeno černých stromů

# TODO RB_INSERT(T, n) s ošetřením 3 případů

# TODO RB_REMOVE(T, z) s ošetřením případů pro různý počet synů



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

# ---===== KONEC UNIT TESTŮ =====---
    
# test vlatností haldy
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

# test vlastností binárního vyhledávacícho stromu
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

    # průchod celým stromem
    is_valid_BVS(x.left)
    is_valid_BVS(x.right)
    

# vygeneruje náhodné pole velikosti size
def generate_array(size):
    array = [0 for _ in range(size)]
    for i in range(size):
        array[i] = randint(-100, 100)
    assert(size == len(array))
    return array

# vygeneruje náhodné pole s prvky v intervalu <0, 100>
def generate_positive_array(size):
    array = [0 for _ in range(size)]
    for i in range(size):
        array[i] = randint(0, 100)
    assert(size == len(array))
    return array
        

run_tests()
