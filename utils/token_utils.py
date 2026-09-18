def find_subsequence(haystack, needle, occurrence=1):
    if not needle or occurrence < 1:
        return None
    n = len(needle)
    count = 0
    for i in range(len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            count += 1
            if count == occurrence:
                return i
    return None