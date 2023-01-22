def fib(n):
    if n <= 1:
        return 1
    return fib(n-1) + fib(n-2)

def smart_fib(n, cache):
    if n <= 1:
        result = 1
    elif n in cache:
        return cache[n]
    else:
        result = smart_fib(n-1, cache) + smart_fib(n-2, cache)
        cache[n] = result
    return result

if __name__ == "__main__":
    import sys
    print(sys.argv)
    print(fib(13))
    print(smart_fib(13, {}))