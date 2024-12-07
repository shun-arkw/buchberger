import itertools
import numpy as np
from joblib import Parallel, delayed

load("/app/src/lib/count_num_additions.sage")

nvars = 8
field = GF(32003)
Ring = PolynomialRing(field, nvars, 'x', order='lex')


I_katsura = sage.rings.ideal.Katsura(Ring, nvars) 
F_katsura = list(I_katsura.gens())

for f in F_katsura:
    print(f)


num_additions_counter = NumAdditionsCounter(coeff_field = field,
                                                num_variables = nvars,
                                                select_strategy = 'normal',
                                                threshold = 1000000,
                                                timeout = 10,
                                                check_gb=False)


permutations = list(itertools.permutations(range(nvars)))
permutations = [list(perm) for perm in permutations]

print(len(permutations))

results = Parallel(n_jobs=-1, backend="multiprocessing", verbose=True)(delayed(num_additions_counter.run)(weight_matrix='grevlex', polynomial_list=F_katsura, variable_order=perm) for perm in permutations)


default_result = results[0]
max_result = results[np.argmax([res.polynomial_additions for res in results])]
min_result = results[np.argmin([res.polynomial_additions for res in results])]
    
print(default_result.variable_order, default_result.polynomial_additions, default_result.total_time)
print(max_result.variable_order, max_result.polynomial_additions, max_result.total_time)
print(min_result.variable_order, min_result.polynomial_additions, min_result.total_time)



