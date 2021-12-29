#!/usr/bin/env pypy3

import numpy as np
import secrets
import sys
import logging
logging.basicConfig(level=logging.INFO)
import time

IDENTITY = -2**63

def add(a, b):
    return [[a if a > b else b for a, b in zip(xr, yr)] for xr, yr in zip(a, b)]

def multiply(a, b):
    zb = list(zip(*b))
    return [[max(x + y if x != IDENTITY and y != IDENTITY else IDENTITY for x, y in zip(row, col)) for col in zb] for row in a]

def scalar_multiply(a, b):
    if type(a) == list:
        a, b = b, a

    return [[e+a if e != IDENTITY else IDENTITY for e in c] for c in b]

def add_adj(a, b):
    return add(add(a, b), multiply(a, b))

def scalar_sum(l):
    total = 0
    for t in l:
        if t == IDENTITY:
            return IDENTITY
        total += t
    return total

def eigen(m):
    """
    Determines the actual eigenvalue of the matrix through computing a critical cycle
    """
    return critical(m)[1]

def has_cycle(m):
    """
    Returns true if the matrix contains a critical cycle
    """
    return len(critical(m)[0]) != 0

def critical(A):
    """
    Computes a pair (critical cycles, eigenvalue) of a matrix

    From:
    An efficient algorithm for critical circuits and finite eigenvectors in the max-plus algebra
    https://www.sciencedirect.com/science/article/pii/S0024379599001202
    """

    # We first estimate mu
    if all(A[i][j] == IDENTITY for i in range(len(A)) for j in range(len(A))):
        logging.debug("Hey, this matrix is kinda sus")
        return [], 0
    mu = max(A[i][i] for i in range(len(A)))
    if mu == IDENTITY:
        logging.debug("Fallback guess on computing circuit")
        mu = min(A[j][i] for i in range(len(A)) for j in range(len(A)) if A[j][i] != IDENTITY)
    logging.debug("Guess for mu is " + str(mu))
    paths_ = []
    for i in range(len(A)):
        row = []
        paths_.append(row)
        for j in range(len(A)):
            if A[i][j] != IDENTITY:
                row.append(i)
            else:
                row.append(None)
    while True:
        # Check if circuit exists with positive weight to mu
        Aprime = [[e - mu if e != IDENTITY else IDENTITY for e in r] for r in A]
        paths = [[e for e in r] for r in paths_]
        Aprime_k = Aprime
        zero_nodes = []
        for k in range(len(A)):
            Aprime_k_ = []
            for i in range(len(A)):
                row = []
                Aprime_k_.append(row)
                for j in range(len(A)):
                    if Aprime_k[i][j] < scalar_sum((Aprime_k[i][k],Aprime_k[k][j])):
                        paths[i][j] = paths[k][j]
                        row.append(scalar_sum((Aprime_k[i][k],Aprime_k[k][j])))
                    else:
                        row.append(Aprime_k[i][j])
            Aprime_k = Aprime_k_
            # Check for positive circuits on diagonal entries
            done = False
            for i in range(len(A)):
                if Aprime_k[i][i] > 0:
                    # print('b', i)
                    done = True
                    # Backtrack
                    prev = paths[i][i]
                    path = [i, prev]
                    while prev != i:
                        prev = paths[i][prev]
                        path.append(prev)
                elif i not in zero_nodes and Aprime_k[i][i] == 0:
                    zero_nodes.append(i)
            if done:
                break
        else:
            # We are done
            if len(zero_nodes) == 0:
                logging.debug("No cycle here")
                return [], mu
            logging.debug("Found cycle using " + str(zero_nodes))
            target = zero_nodes[-1]
            # Backtrack to obtain paths
            prev = paths[target][target]
            path = [target, prev]
            while prev != target:
                prev = paths[target][prev]
                path.append(prev)
            logging.debug('Mu (correct eigenvalue) is %d', mu)
            return path, mu

        # Update guess and continue
        mu = 0
        for i in reversed(range(len(path)-1)):
            mu += A[path[i+1]][path[i]]
        mu /= len(path) - 1
        logging.debug("mu guess updated to " + str(mu))

def generate_strong_matrix(DIMENSION):
    """
    Generate a "strong" square matrix of given dimension using the method
    in section 5.2
    """

    ev = 0
    m_ = [[IDENTITY]]
    while ev <= 0 or not all(i != IDENTITY for row in m_ for i in row):
        # (a)
        k1 = np.random.randint((3*DIMENSION)//12, (5*DIMENSION)//12)
        k2 = np.random.randint(k1, DIMENSION-k1)
        # Check that they contain a cycle, need to restart if not
        m1 = np.random.choice([IDENTITY, 0], size=(k1, k1), replace=True, p=[2/3, 1/3]).astype(int)
        if not has_cycle(m1):
            continue
        m2 = np.random.choice([IDENTITY, 0], size=(k2-k1, k2-k1), replace=True, p=[2/3, 1/3]).astype(int)
        if not has_cycle(m2):
            continue
        m3 = np.random.choice([IDENTITY, 0], size=(DIMENSION-k2, DIMENSION-k2), replace=True, p=[2/3, 1/3]).astype(int)
        if not has_cycle(m3):
            continue

        # (b)
        m = np.full((DIMENSION, DIMENSION), IDENTITY, dtype=int)
        m[:k1, :k1] = m1
        m[k1:k2, k1:k2] = m2
        m[k2:, k2:] = m3

        # (c)
        mask = m == IDENTITY
        np.putmask(m, mask, np.random.randint(-100, 0, size=(DIMENSION, DIMENSION)))
        m += secrets.randbelow(100)

        # (d)
        d = np.diag(np.random.randint(-2**50, 2**50, size=DIMENSION))
        mask = d == 0
        di = np.negative(d)
        np.putmask(d, mask, np.full((DIMENSION, DIMENSION), IDENTITY))
        np.putmask(di, mask, np.full((DIMENSION, DIMENSION), IDENTITY))
        m_ = multiply(multiply(di, m), d)

        # compute eigenvalue of the matrix
        ev = eigen(m_)
        logging.debug("lambda %d" % ev)
    
    m_ = [[int(e) for e in r] for r in m_]
    return m_

def tropical_pow(x, y, op=multiply):
    """
    Simple binary expansion
    """
    if 1 == y:
        return x
    exp = bin(y)
    value = x
 
    for i in range(3, len(exp)):
        value = op(value, value)
        if(exp[i:i+1]=='1'):
            value = op(value, x)
    return value

def semidirect_product(XG, YH):
    x, g = XG
    y, h = YH
    return add(add_adj(x, h), y), add_adj(g, h)

def semidirect_pair_product(M, H, k):
    return tropical_pow((M, H), k, op=semidirect_product)

def star(H, I, DIMENSION):
    """
    Star operator
    """
    assert eigen(H) <= 0

    Hstar = add(I, H)
    for i in range(2, DIMENSION):
        Hstar = add(Hstar, tropical_pow(H, i))
    return Hstar

def mat_repr(m):
    """
    Display matrix replacing IDENTITY with -inf
    """
    
    res = []
    for row in m:
        r = []
        for item in row:
            if item == IDENTITY:
                r.append("−∞")
            else:
                r.append(str(item))
        res.append("[" + ", ".join(r) + "]")
    return "[" + ", ".join(res) + "]"

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        DIMENSION = int(sys.argv[1])
    else:
        DIMENSION = 100
    logging.info("Generating random values of dimension %d" % DIMENSION)

    # For secrets like in 5.2
    # m = np.random.randint((DIMENSION - 1) ** 2 + 1, DIMENSION**2 + 1)
    # n = np.random.randint((DIMENSION - 1) ** 2 + 1, DIMENSION**2 + 1)

    # For bigger secrets
    m = secrets.randbits(64)
    n = secrets.randbits(64)

    M = []
    for i in range(DIMENSION):
        row = []
        M.append(row)
        for j in range(DIMENSION):
            if secrets.randbits(1):
                row.append(IDENTITY)
            else:
                row.append(secrets.randbits(128) - 2**64)
    H = generate_strong_matrix(DIMENSION)
    
    # Example 1
    """
    m = 5
    n = 8
    M = [[8, 7, 2], [10, 3, 6], [-10, -1, 3]]
    H = [[0, -3, -5], [-1, -2, 2], [1, -3, -4]]
    """

    # Example 2
    """
    m = 15
    n = 16
    M = [[-75, -45, -69, 60], [83, 52, 9, -72], [27, 92, 92, -16], [87, 93, -3, 74]]
    H = [[1, 7, 2, 5], [-1, -2, 2, 4], [3, 4, 2, 2], [-5, -10, 10, 0]]
    """

    logging.info("Computing keys (pretty slow)")
    start_time = time.time()

    A, Hm = semidirect_pair_product(M, H, m)
    B, Hn = semidirect_pair_product(M, H, n)

    Ka = add(A, add(B, add(Hm, multiply(B, Hm))))
    Kb = add(A, add(B, add(Hn, multiply(A, Hn))))

    end_time = time.time()
    logging.info("Computing keys took %d seconds" % (end_time - start_time))

    logging.info("- Private keys:")
    logging.info("    m %d" % m)
    logging.info("    n %d" % n)
    logging.debug("    Ka " + mat_repr(Ka))
    logging.debug("    Hm " + mat_repr(Hm))
    logging.debug("    Hn " + mat_repr(Hm))
    logging.debug("- Public keys:")
    logging.debug("    M " + mat_repr(M))
    logging.debug("    H " + mat_repr(H))
    logging.debug("    A " + mat_repr(A))
    logging.debug("    B " + mat_repr(B))

    assert Ka == Kb

    logging.info("Testing attack")
    start_time = time.time()
    # A = M?
    if A == M:
        print("A=M, m=1")
        exit()
    elif B == M:
        print("B=M, n=1")
        exit()

    logging.info("Computing critical cycle...")
    cH, e = critical(H)
    logging.debug("critical(H) " + str(cH))
    logging.debug('eigen(H) %d' % e)

    # Skip lambda <= 0
    DIMENSION = len(H)
    I = np.identity(DIMENSION, dtype=int)
    mask = I == 0
    np.putmask(I, mask, np.full((DIMENSION, DIMENSION), IDENTITY))
    mask = I == 1
    np.putmask(I, mask, np.full((DIMENSION, DIMENSION), 0))

    if e <= 0: 
        print("Trivial case")
        assert add(A, B) == Ka
        exit()
    
    F = add(I.astype(int), H)
    logging.debug("F " + mat_repr(F))
    V = add(multiply(M, add(I, H)), H)
    logging.debug("V " + mat_repr(V))

    #eF = eigen(F)
    #logging.debug('eigen(F) %d' % eF)
    eF = e

    # 2.7 1
    """
    for t in tqdm(range((DIMENSION-1)**2), desc='Checking t'):
        if multiply(V, tropical_pow(F, t)) == A:
            print("recovered", t)
            exit()
    """

    U = scalar_multiply(-eF, F)
    logging.debug('U ' + mat_repr(U))
    U = tropical_pow(U, len(cH))
    U = star(U, I, DIMENSION=DIMENSION)

    # Cz = [[U[i][j] if j in cH else IDENTITY for j in range(DIMENSION)] for i in range(DIMENSION)]
    # Rz = [[U[i][j] if i in cH else IDENTITY for j in range(DIMENSION)] for i in range(DIMENSION)]
    # Note that Cz should equal Rz for us
    Cz = U
    Rz = U
    logging.debug("Cz=Rz " + mat_repr(Cz))

    cH_pairs = []
    for ii in range(len(cH)-1):
        cH_pairs.append((cH[ii],cH[ii+1]))

    Sz = [[(F[i][j] - eF) if (i, j) in cH_pairs else IDENTITY for j in range(DIMENSION)] for i in range(DIMENSION)]
    logging.debug('Sz ' + mat_repr(Sz))

    # 2.7 step 2
    intermediate = add(Cz, Rz)
    intermediate = multiply(V, intermediate)

    target = [[A[i][j] - intermediate[i][j] for j in range(DIMENSION)] for i in range(DIMENSION)]

    logging.debug('target matrix ' + mat_repr(target))
    delta = 0
    res = target[0][0]
    while res % eF != 0:
        if delta <= 0:
            delta = -delta + 1
        else:
            delta = -delta
        res = target[0][0]+delta

    res = res//eF + 2
    end_time = time.time()
    logging.info("Attack took %d seconds " % (end_time - start_time))
    logging.info("Recovered key     %d" % res)
    logging.info("Private key was   %d" % m)
    assert res == m
