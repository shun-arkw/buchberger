import bisect
import numpy as np
import time
import IPython
from line_profiler import LineProfiler

def monic(f):
    """
    >>> R.<x,y,z> = PolynomialRing(QQ, 'x, y, z', order='degrevlex') 
    >>> monic(f)    
    x + 2/3*y

    >>> R.<x,y,z> = PolynomialRing(GF(32003), 'x, y, z', order='degrevlex')
    >>> f = 3*y^2*z^3 + 4*z^4 - y^2 + z^2
    >>> monic(f)
    y^2*z^3 + 10669*z^4 - 10668*y^2 + 10668*z^2
    """

    # if f.is_zero():
    if f == 0:
        raise ZeroDivisionError('Cannot make a zero polynomial monic')
    return f / f.lc()



def select(poly_ring, G, P, strategy='normal'):
    """Select and return a pair from P."""
    assert len(G) > 0, "polynomial list must be nonempty"
    assert len(P) > 0, "pair set must be nonempty"
    R = poly_ring

    if isinstance(strategy, str):
        strategy = [strategy]

    def strategy_key(p, s):
        """Return a sort key for pair p in the strategy s."""
        if s == 'first':
            return p[1], p[0]
        elif s == 'normal':
            lcm = R.monomial_lcm(G[p[0]].lm(), G[p[1]].lm())
            return lcm
        elif s == 'degree':
            lcm = R.monomial_lcm(G[p[0]].lm(), G[p[1]].lm())
            return lcm.total_degree()
        elif s == 'random':
            return np.random.rand()
        else:
            raise ValueError('unknown selection strategy')

    return min(P, key=lambda p: tuple(strategy_key(p, s) for s in strategy))




# f gの最高次の係数は1でなくてはいけない
def spoly(poly_ring, f, g, lmf=None, lmg=None):
    """Return the s-polynomial of monic polynomials f and g.
    
    >>> R.<x,y,z> = PolynomialRing(ZZ, 'x, y, z', order='lex')
    >>> F = [x^3*y^2 - x^2*y^3 + x, x^4*y + y^2]
    >>> spoly(R, F[0], F[1])
    -x^3*y^3 + x^2 - y^3
    
    """
    lmf = f.lm() if lmf is None else lmf
    lmg = g.lm() if lmg is None else lmg

    R = poly_ring
    lcm = R.monomial_lcm(lmf, lmg)
    s1 = f * R.monomial_quotient(lcm, lmf, coeff=True)
    s2 = g * R.monomial_quotient(lcm, lmg, coeff=True)
    
    return s1 - s2



# Fは、最高次の係数が1の多項式のリスト
def reduce(poly_ring, g, F, lmF=None):
    """Return a remainder and stats when g is divided by monic polynomials F.

    Parameters
    ----------
    poly_ring : polynomial ring
     
    g : polynomial
        Dividend polynomial.
    F : list
        List of monic divisor polynomials.
    lmF : list, optional
        Precomputed list of lead monomials of F for efficiency.

    Examples
    --------

    >>> R.<a,b,c> = PolynomialRing(ZZ, 'a, b, c', order='deglex')
    >>> F = [x*y - 1, y^2 - 1]
    >>> f = x^2*y + x*y^2 + y^2
    >>> reduce(R, f, F)
    (a + b + 1, {'steps': 3})

    >>> F2 = [a^2 + b, a*b*c + c, a*c^2 + b^2]
    >>> f2 = a^3*b*c^2 + a^2*c
    >>> reduce(R, f2, F2)
    (b*c^2 - b*c, {'steps': 3})

    """
    ring = poly_ring
    monomial_divides = ring.monomial_divides
    lmF = [f.lm() for f in F] if lmF is None else lmF

    stats = {'steps': 0}
    r = 0 #あまりを0で初期化
    h = g # hは中間被除数
    
    while h != 0:
        lmh, lch = h.lm(), h.lc()
        found_divisor = False
        # print(f'中間被除数: {h}')
        for f, lmf in zip(F, lmF):
            
            if monomial_divides(lmf, lmh):
                m = ring.monomial_quotient(lmh, lmf)
                h = h - f * m * lch
                
                found_divisor = True
                stats['steps'] += 1
          
                break

        if not found_divisor:
            r = r + h.lt()
            h = h - h.lt()
        
    
    return r, stats

def reduce_log(poly_ring, g, F, lmF=None):
   
    ring = poly_ring
    monomial_divides = ring.monomial_divides
    lmF = [f.lm() for f in F] if lmF is None else lmF

    stats = {'steps': 0}
    r = 0 #あまりを0で初期化
    h = g # hは中間被除数
    
    print(f'spoly: {g}')
    print(f'F: {F}')
    print('h', h)
    while h != 0:
        lmh, lch = h.lm(), h.lc()
        found_divisor = False
        # print(f'中間被除数: {h}')
        for f, lmf in zip(F, lmF):
            
            if monomial_divides(lmf, lmh):
                m = ring.monomial_quotient(lmh, lmf)
                h = h - f * m * lch
                
                found_divisor = True
                stats['steps'] += 1
                print(f'step: {stats["steps"]}')
                break

        if not found_divisor:
            r = r + h.lt()
            h = h - h.lt()

        print('h', h)

    return r, stats

def update(poly_ring, G, P, f, strategy='gebauermoeller', lmG=None):
    """Return the updated lists of polynomials and pairs when f is added to the basis G.

    The inputs G and P are modified by this function.

    Parameters
    ----------
    poly_ring : polynomial ring 

    G : list
        Current list of polynomial generators.
    P : list
        Current list of s-pairs.
    f : polynomial
        New polynomial to add to the basis.
    strategy : {'gebauermoeller', 'lcm', 'none'}, optional
        Strategy for pair elimination.

        Strategy can be 'none' (eliminate no pairs), 'lcm' (only eliminate pairs that
        fail the LCM criterion), or 'gebauermoeller' (use full Gebauer-Moeller elimination).
    lmG : list, optional
        Precomputed list of the lead monomials of G for efficiency.

    Examples
    --------
    >>> R.<x,y,z> = PolynomialRing(GF(32003), 'x, y, z', order='degrevlex')
    >>> G = [x*y^2 + 2*z, x*z^2 - y^2 - z, x + 3]
    >>> P = [(0, 2)]
    >>> f = y^2*z^3 + 4*z^4 - y^2 + z^2
    >>> update(R, G, P, f)
    ([x*y^2 + 2*z, x*z^2 - y^2 - z, x + 3, y^2*z^3 + 4*z^4 - y^2 + z^2], [(0, 2)])

    """
    lmf = f.lm()
    lmG = [g.lm() for g in G] if lmG is None else lmG
    R = poly_ring

    lcm = R.monomial_lcm
    mul = operator.mul
    div = R.monomial_divides  #　割り切るかどうか div(A, B) AがBを割り切るときTrue
    
    m = len(G)

    if strategy == 'none':
        P_ = [(i, m) for i in range(m)]

    elif strategy == 'lcm':
        P_ = [(i, m) for i in range(m) if lcm(lmG[i], lmf) != mul(lmG[i], lmf)]

    elif strategy == 'gebauermoeller':
        def can_drop(p):
            i, j = p
            gam = lcm(lmG[i], lmG[j])
            return div(lmf, gam) and gam != lcm(lmG[i], lmf) and gam != lcm(lmG[j], lmf)
        P[:] = [p for p in P if not can_drop(p)]

        lcms = {}
        for i in range(m):
            lcms.setdefault(lcm(lmG[i], lmf), []).append(i)
        min_lcms = []
        P_ = []
        for gam in sorted(lcms.keys()): 
            if all(not div(m, gam) for m in min_lcms):
                min_lcms.append(gam)
                if not any(lcm(lmG[i], lmf) == mul(lmG[i], lmf) for i in lcms[gam]):
                    P_.append((lcms[gam][0], m))
        P_.sort(key=lambda p: p[0])

    else:
        raise ValueError('unknown elimination strategy')

    G.append(f)
    P.extend(P_)

    return G, P

def order(poly_ring, monomial):
    """
    """
    if not monomial.is_monomial():
        return ValueError('not monomial')

    # sagemath(singular)の使用上，weight matrixの成分が実数であるとき，整数に丸め込まれている可能性がある
    matrix = poly_ring.term_order().matrix() # 実数のまま処理しているので，intにする必要があるかも
    exponential_vector = vector(monomial.exponents().pop())

    if matrix:
        return matrix * exponential_vector
    else:
        raise ValueError('monomial order is not defined by the matrix')

def minimalize(poly_ring, G):
    """Return a minimal Groebner basis from arbitrary Groebner basis G."""
    R = poly_ring if len(G) > 0 else None
    Gmin = []
    for f in sorted(G, key=lambda h: h.lm()):
        if all(not R.monomial_divides(g.lm(), f.lm()) for g in Gmin):
            Gmin.append(f)
    return Gmin


def interreduce(poly_ring, G):
    """Return the reduced Groebner basis from minimal Groebner basis G.
    
    Examples
    --------
    >>> R.<x,y,z> = PolynomialRing(GF(32003), 'x, y, z', order='degrevlex')
    >>> G = [x^3-2*x*y, x^2*y - 2*y^2 + x, -x^2, -2*x*y, -2*y^2 + x]
    >>> I = Ideal(G)
    >>> print(I.basis_is_groebner()) # Gがグレブナー基底であること確認
    True

    interreduce(R, minimalize(R, G))
    >>> [y^2 + 16001*x, x*y, x^2]

    I.groebner_basis() # 確かめ
    >>> [x^2, x*y, y^2 + 16001*x]
    
    """
    Gred = []
    for i in range(len(G)):
        g, _ = reduce(poly_ring, G[i], G[:i] + G[i+1:])
        Gred.append(monic(g))
    return Gred


def buchberger(poly_ring, F, S=None, select_strategy='degree', stop_algorithm=True, threshold=100000, elimination='gebauermoeller', rewards='additions', sort_reducers=True, gamma=0.99):
    """Return the Groebner basis for the ideal generated by F using Buchberger's algorithm.

    Parameters
    ----------
    poly_ring : polynomial ring
        Monomial order must be defined by the matrix.
    F : list
        List of polynomial generators.
    S : list or None, optional
        List of current remaining s-pairs (None indicates no s-pair has been done yet).
    select_strategy : {'firs', 'degree', 'normal', 'random'}, optional
        Strategy for selecting the s-pair.
    stop_algorithm : bool, optional
        Whether to stop the algorithm when the number of polynomial additions exceeds the threshold.
    threshold : int, optional
        The threshold for the number of polynomial additions.
    elimination : {'gebauermoeller', 'lcm', 'none'}, optional
        Strategy for pair elimination.
    rewards : {'additions', 'reductions'}, optional
        Reward value for each step.
    sort_reducers : bool, optional
        Whether to choose reducers in sorted order by lead monomial.
    gamma : float, optional
        Discount rate for rewards.

    Examples
    --------
    >>> m_lex = matrix([[1,0,0], [0,1,0], [0,0,1]])
    >>> m_grlex = matrix([[1,1,1], [1,0,0], [0,1,0]])
    >>> m_grevlex = matrix([[1,1,1], [0,0,-1], [0,-1,0]])
    
    >>> T1 = TermOrder(m_lex)
    >>> T2 = TermOrder(m_grlex)
    >>> T3 = TermOrder(m_grevlex)

    >>> P = PolynomialRing(GF(32003), 'x, y, z', order=T3)
    >>> x, y, z = P.gens()

    >>> F = [x^3 - 2*x*y, x^2*y - 2*y^2 + x]
    >>> F2 = [x^3 - 2*x*y, x^2*y - 2*y^2 + x, x^2*z^2 + x*y^2*z]
    >>> buchberger(P, F, select_strategy='normal')

    ([y^2 + 16001*x, x*y, x^2],
    {'zero_reductions': 2,
    'nonzero_reductions': 3,
    'polynomial_additions': 7,
    'total_reward': -7.00000000000000,
    'discounted_return': -6.83189002000000})

    """

    Ring = poly_ring
    # 多項式環Rで定義されている単項式順序に基づいて、多項式の項の順序を並び替える。
    
    F_new = list(map(Ring, F))

    if S is None:
        G, lmG = [], []
        P = []
        for f in F_new:
            G, P = update(Ring, G, P, monic(f), strategy=elimination)
            lmG.append(f.lm())
    else:
        G, lmG = F_new, [f.lm() for f in F_new]
        P = S


    stats = {'zero_reductions': 0,
             'nonzero_reductions': 0,
             'polynomial_additions': 0,
             'total_reward': 0.0,
             'discounted_return': 0.0}
    discount = 1.0

    if sort_reducers and len(G) > 0:
        G_ = G.copy()
        G_.sort(key=lambda g: g.lm())
        lmG_, keysG_ = [g.lm() for g in G_], [order(Ring, g.lm()) for g in G_]
    else:
        G_, lmG_ = G, lmG


    while P:
        i, j = select(Ring, G, P, strategy=select_strategy)
        P.remove((i, j))
        spoly_ = spoly(Ring, G[i], G[j], lmf=lmG[i], lmg=lmG[j])

        r, s = reduce(Ring, spoly_, G_)
        
        reward = (-1.0 - s['steps']) if rewards == 'additions' else -1.0
        stats['polynomial_additions'] += s['steps'] + 1
        stats['total_reward'] += reward
        stats['discounted_return'] += discount * reward
        discount *= gamma

        if stop_algorithm and stats['polynomial_additions'] > threshold:
            # raise RuntimeError('Failed to find a Groebner basis with')
            # print('Failed to find a Groebner basis, then the number of polynomial additions is ', stats['polynomial_additions'])
            return [], stats

        if r != 0:
            G, P = update(Ring, G, P, monic(r), lmG=lmG, strategy=elimination)
            lmG.append(r.lm())
            if sort_reducers:
                key = order(Ring, r.lm())
                index = bisect.bisect(keysG_, key)
                G_.insert(index, monic(r))
                lmG_.insert(index, r.lm())
                keysG_.insert(index, key)
            else:
                G_ = G
                lmG_ = lmG
            stats['nonzero_reductions'] += 1
        else:
            stats['zero_reductions'] += 1

    return interreduce(Ring, minimalize(Ring, G)), stats


def buchberger_log(poly_ring, F, S=None, select_strategy='degree', stop_algorithm=False, threshold=1000, elimination='gebauermoeller', rewards='additions', sort_reducers=True, gamma=0.99):
    
    print('start buchberger')
    Ring = poly_ring
    # 多項式環Rで定義されている単項式順序に基づいて、多項式の項の順序を並び替える。
    
    F_new = list(map(Ring, F))
    print(f'polynomial_list: {F_new}')

    if S is None:
        G, lmG = [], []
        P = []
        for f in F_new:
            print(f'monic(f): {monic(f)}')
            G, P = update(Ring, G, P, monic(f), strategy=elimination)

            print(f'G: {G}')
            print(f'P: {P}')

            lmG.append(f.lm())
    else:
        G, lmG = F_new, [f.lm() for f in F_new]
        P = S

    if not P:
        print('Pのupdate失敗')

    stats = {'zero_reductions': 0,
             'nonzero_reductions': 0,
             'polynomial_additions': 0,
             'total_reward': 0.0,
             'discounted_return': 0.0}
    discount = 1.0

    if sort_reducers and len(G) > 0:
        G_ = G.copy()
        G_.sort(key=lambda g: g.lm())
        lmG_, keysG_ = [g.lm() for g in G_], [order(Ring, g.lm()) for g in G_]
    else:
        G_, lmG_ = G, lmG

    print()
    print('while内部')
    while P:
        i, j = select(Ring, G, P, strategy=select_strategy)
        P.remove((i, j))
        print(G[i], 'と', G[j], 'のspolyを計算')
        spoly_ = spoly(Ring, G[i], G[j], lmf=lmG[i], lmg=lmG[j])

        
        print('割り算開始')
        r, s = reduce_log(Ring, spoly_, G_)
        print('割り算終了')
        print()
        
        reward = (-1.0 - s['steps']) if rewards == 'additions' else -1.0
        stats['polynomial_additions'] += s['steps'] + 1
        stats['total_reward'] += reward
        stats['discounted_return'] += discount * reward
        discount *= gamma

        if stop_algorithm and stats['polynomial_additions'] > threshold:
            # raise RuntimeError('Failed to find a Groebner basis with')
            print('Failed to find a Groebner basis with')
            return [], stats

        if r != 0:
            G, P = update(Ring, G, P, monic(r), lmG=lmG, strategy=elimination)
            lmG.append(r.lm())
            if sort_reducers:
                key = order(Ring, r.lm())
                index = bisect.bisect(keysG_, key)
                G_.insert(index, monic(r))
                lmG_.insert(index, r.lm())
                keysG_.insert(index, key)
            else:
                G_ = G
                lmG_ = lmG
            stats['nonzero_reductions'] += 1
        else:
            stats['zero_reductions'] += 1

        print()
        print('polynomial_additions:', stats['polynomial_additions'])
        print()

    print('while終了')
    print()
    print('final additons:', stats['polynomial_additions'])
    print('簡約前のグレブナー基底:', G)
    print()
    return interreduce(Ring, minimalize(Ring, G)), stats

def validation(weight_matrix):
    """Return True if the weight matrix is valid, False otherwise.
    
    Parameters
    ----------
    weight_matrix : np.ndarray
        正方行列かつ成分が整数必要がある
    
    
    """

    num_rows, num_columns = weight_matrix.shape
    assert num_rows == num_columns, "The matrix must be square"

    # The weight matrix must contain integer elements.
    if not np.issubdtype(weight_matrix.dtype, np.integer):
        raise ValueError("The weight matrix must contain integer elements")
    
    # The first non-zero element in each column of the weight matrix must be positive.
    transposed_matrix = weight_matrix.T 
    for row in transposed_matrix:
        for elem in row:
            if elem > 0:
                break
            elif elem < 0:
                return False
            
    # The weight matrix must be full rank.
    rank = np.linalg.matrix_rank(weight_matrix)
    if rank == num_rows:
        return True
    else:
        return False

# def count(params, polynomial_list, coeff_field, num_variables=3, select_strategy='degree', stop_algorithm=False, threshold=10000):
#     '''Return the number of polynomial additions to compute the Groebner basis.

#     Parameters
#     ----------
#     params : np.ndarray
#         Parameters for the term order matrix.
#     F : list
#         List of polynomial generators.
#     coeff_field : sage.rings.ring.Ring


#     num_variables : int
#         The number of variables.
#     select_strategy : str, optional
#         The strategy for selecting the S-pair. The default is 'degree'.
#     stop_algorithm : bool, optional

#     threshold : int, optional
#         The threshold for the number of polynomial additions. The default is 10000.

#     '''

#     assert len(params) == num_variables**2, "The number of parameters must be equal to the number of variables in the weight matrix."

#     # Check if the weight matrix is valid or not.
#     int_params = params.astype(int)
#     int_matrix = int_params.reshape(num_variables, num_variables)
#     if not validation(int_matrix):
#         return -1

    
#     matrix_ = matrix(num_variables, list(params)) # paramsがnp.ndarrayの場合うまくいかない
    
#     P = PolynomialRing(coeff_field, num_variables, 'x', order=TermOrder(matrix_))
    
    
#     polynomial_list = list(map(P, polynomial_list))
#     _, stat = buchberger(P, polynomial_list, select_strategy=select_strategy, stop_algorithm=stop_algorithm, threshold=threshold)

#     additons = stat['polynomial_additions']

#     if additons > threshold:
#         return threshold
#     else:
#         return additons

def count_add(weight_matrix, polynomial_list, coeff_field, num_variables=3, select_strategy='normal', stop_algorithm=False, threshold=10000):
    '''Return the number of polynomial additions to compute the Groebner basis.

    Parameters
    ----------
    weight_matrix : np.ndarray
        正方行列である必要がある
    polynomial_list : list
        List of polynomial generators.
    coeff_field : sage.rings.ring.Ring


    num_variables : int
        The number of variables.
    select_strategy : str, optional
        The strategy for selecting the S-pair. The default is 'normal'.
    stop_algorithm : bool, optional

    threshold : int, optional
        The threshold for the number of polynomial additions. The default is 10000.

    '''

    num_rows, num_columns = weight_matrix.shape
    assert num_rows == num_columns, "The weight matrix must be square"


    # Check if the weight matrix is valid or not.
    int_matrix = weight_matrix.astype(int)

    if not validation(int_matrix):
        return -1

    
    matrix_ = matrix(int_matrix.tolist()) # paramsがnp.ndarrayの場合うまくいかない
    
    P = PolynomialRing(coeff_field, num_variables, 'x', order=TermOrder(matrix_))
    
    
    polynomial_list = list(map(P, polynomial_list))
    _, stat = buchberger(P, polynomial_list, select_strategy=select_strategy, stop_algorithm=stop_algorithm, threshold=threshold)

    additons = stat['polynomial_additions']

    if additons > threshold:
        return threshold
    else:
        return additons





    

if __name__ == '__main__':

    m_lex = matrix([[1,0,0], [0,1,0], [0,0,1]])
    m_grlex = matrix([[1,1,1], [1,0,0], [0,1,0]])
    m_grevlex = matrix([[1,1,1], [0,0,-1], [0,-1,0]])
    m_original = matrix([[696, 226,  13, 623], [-796, -818,  748,  559], [-822, -593,  394,  101], [546, 533, 852, 830]])

    
    T1 = TermOrder(m_lex)
    T2 = TermOrder(m_grlex)
    T3 = TermOrder(m_grevlex)
    T4 = TermOrder(m_original)

    P = PolynomialRing(GF(32003), 4, 'x', order=T4)
    x0, x1, x2, x3 = P.gens()

    # F = [x^3 - 2*x*y, x^2*y - 2*y^2 + x]
    # F2 = [x^7 - 2*x*y, x^7*y - 2*y^7 + x, x^2*z^2 + x*y^5*z]
    F = [-15261*x0^2*x1 - 13733*x0*x2^2*x3 - 982*x3^2 - 6313*x0*x1^2*x2 - 4354*x0 - 8201*x2, -9860*x1^2*x3^3, 12217*x0^3*x2 - 880*x1^3*x3 + 6416*x1*x2 + 8232*x1 + 8221]
    
    start = time.time()
    result = buchberger(P, F, select_strategy='normal', threshold=10000)
    end = time.time()
    print('time:', end-start)
    print(result)

    prof = LineProfiler()
    prof.add_function(buchberger)
    prof.add_function(reduce)
    prof.runcall(buchberger, P, F, select_strategy='normal', threshold=10000)
    prof.print_stats()
    