from numpy.random import randint
import itertools as it


class Polynomial_Sampler():
    def __init__(self, 
                 ring, 
                 degree_sampling='uniform', 
                 term_sampling='uniform', 
                 strictly_conditioned=True,
                 conditions={}):
        self.ring = ring
        self.field = ring.base_ring()
        self.degree_sampling = degree_sampling
        self.term_sampling = term_sampling
        self.strictly_conditioned = strictly_conditioned
        self.conditions= conditions

    def sample(self, num_samples=1, size=None, density=1.0, matrix_type=None):
        max_degree      = self.conditions['max_degree']
        max_num_terms   = self.conditions['max_num_terms']
        min_degree      = self.conditions['min_degree'] if 'min_degree' in self.conditions else 0
        max_coeff       = self.conditions['max_coeff'] if 'max_coeff' in self.conditions else None
        num_bound       = self.conditions['num_bound'] if 'num_bound' in self.conditions else None
        nonzero_instance= self.conditions['nonzero_instance'] if 'nonzero_instance' in self.conditions else False

        if isinstance(size, tuple):
            assert(len(size) == 2)
            return [self._sample_matrix(self.ring,
                                        size,
                                        max_degree, 
                                        max_num_terms, 
                                        min_degree=min_degree,
                                        max_coeff=max_coeff, 
                                        num_bound=num_bound, 
                                        strictly_conditioned=self.strictly_conditioned,
                                        degree_sampling=self.degree_sampling,
                                        term_sampling=self.term_sampling,
                                        matrix_type=matrix_type,
                                        density=density,
                                        nonzero_instance=nonzero_instance) for _ in range(num_samples)]
        else:
            return [self._sample(self.ring,
                                 max_degree, 
                                 max_num_terms, 
                                 min_degree=min_degree,
                                 max_coeff=max_coeff, 
                                 num_bound=num_bound, 
                                 strictly_conditioned=self.strictly_conditioned,
                                 degree_sampling=self.degree_sampling,
                                 term_sampling=self.term_sampling,
                                 nonzero_instance=nonzero_instance) for _ in range(num_samples)]

    def _sample_matrix(self, 
                       ring,
                       size,
                       max_degree,
                       max_num_terms,
                       min_degree=0,
                       max_coeff=None,
                       num_bound=None,
                       strictly_conditioned=True,
                       degree_sampling=None,
                       term_sampling=None,
                       matrix_type=None,
                       density=1.0,
                       nonzero_instance=False,
                       max_iters=100):
        
        num_polys = prod(size)
        for i in range(max_iters):
            P = [self._sample(ring,
                            max_degree, 
                            max_num_terms, 
                            min_degree=min_degree,
                            max_coeff=max_coeff, 
                            num_bound=num_bound, 
                            strictly_conditioned=strictly_conditioned,
                            degree_sampling=degree_sampling,
                            term_sampling=term_sampling,
                            max_iters=max_iters) for _ in range(num_polys)]
            P = [p if random() < density else p*0 for p in P]
            if not nonzero_instance or not all([p==0 for p in P]): break

            if i == max_iters-1:
                raise RuntimeError(f'Failed to find a nonzero polynomial with {max_iters} iterations')
        
        A = matrix(ring, *size, P)
        if matrix_type == 'unimoduler_upper_triangular':
            for i, j in it.product(range(size[0]), range(size[1])):
                if i == j: A[i, j] = 1
                if i < j: A[i, j] = 0

        return A

    def _sample(self, 
                ring,
                max_degree,
                max_num_terms,
                min_degree=0,
                max_coeff=None,
                num_bound=None,
                strictly_conditioned=True,
                degree_sampling='uniform',
                term_sampling='uniform',
                nonzero_instance=False,
                max_iters=100):
        
        degree          = int(ZZ.random_element(min_degree, max_degree+1)) if degree_sampling == 'uniform' else max_degree
        max_num_terms   = min(max_num_terms, binomial(degree+ring.ngens(), degree))
        num_terms       = int(ZZ.random_element(1, max_num_terms+1)) if term_sampling == 'uniform' else max_num_terms
        ngens           = ring.ngens()

        for i in range(max_iters):
            coeff_ring = ring.base_ring()
            # degree_ = degree + 1 if ngens == 1 else degree  # NOTE: When univariate, the degree options seems work as exclusive. It is a bug in SageMath?
            if coeff_ring == QQ: 
                p = ring.random_element(degree=degree, terms=num_terms, num_bound=num_bound, choose_degree=True)
            elif coeff_ring == RR:
                p = ring.random_element(degree=degree, terms=num_terms, min=-max_coeff, max=max_coeff, choose_degree=True)
            elif coeff_ring == ZZ:
                p = ring.random_element(degree=degree, terms=num_terms, x=-max_coeff, y=max_coeff+1, choose_degree=True)
            else:
                assert(coeff_ring.order() < Infinity)
                p = ring.random_element(degree=degree, terms=num_terms, choose_degree=True)

            if p == 0 and nonzero_instance: continue
            if not strictly_conditioned: break
            if p.total_degree() == degree and len(p.monomials()) == num_terms: break

            # print(p.total_degree(), len(p.monomials()))

            if i == max_iters - 1:
                print(f'conditions: degree={degree}, num_terms={num_terms}')
                raise RuntimeError(f'Failed to find a polynomial satisfying the conditions with {max_iters} iterations')
            
        assert(p != 0)
        return p
    
    