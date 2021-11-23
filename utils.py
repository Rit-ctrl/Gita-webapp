def sort_list(list1, list2):
 
    zipped_pairs = zip(list2, list1)
 
    z = [x for _, x in sorted(zipped_pairs,reverse=True)]
     
    return z

def get_unique_N(iterable, N):
    """Yields (in order) the first N unique elements of iterable. 
    Might yield less if data too short."""
    seen = set()
    for e in iterable:
        if e in seen:
            continue
        seen.add(e)
        yield e
        if len(seen) == N:
            return