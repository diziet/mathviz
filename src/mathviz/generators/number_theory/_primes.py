"""Prime number utilities shared across number theory generators."""

import numpy as np


def sieve_of_eratosthenes(limit: int) -> np.ndarray:
    """Return sorted array of all primes up to limit (inclusive)."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = False
    return np.nonzero(is_prime)[0].astype(np.int64)


def is_prime_array(n: int) -> np.ndarray:
    """Return boolean array of length n where True marks prime indices."""
    if n < 2:
        return np.zeros(n, dtype=bool)
    is_prime = np.ones(n, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(n - 1)) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = False
    return is_prime


def first_n_primes(count: int) -> np.ndarray:
    """Return the first count prime numbers."""
    if count <= 0:
        return np.array([], dtype=np.int64)
    # Upper bound for the nth prime (generous for small n)
    if count < 6:
        upper = 15
    else:
        upper = int(count * (np.log(count) + np.log(np.log(count))) + 10)
    primes = sieve_of_eratosthenes(upper)
    while len(primes) < count:
        upper = int(upper * 1.5)
        primes = sieve_of_eratosthenes(upper)
    return primes[:count]
