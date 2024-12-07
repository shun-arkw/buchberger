load("/app/src/lib/buchberger.sage")
load("/app/src/lib/count_num_additions.sage")
import numpy as np


def validation(weight_matrix):
    """Return True if the weight matrix is valid, False otherwise."""

    num_rows, num_columns = weight_matrix.shape
    assert num_rows == num_columns, "The matrix must be square"
    
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

def output_additions(params, data, coeff_field, operation='sum', num_variables=3, select_strategy='degree', stop_algorithm=False, threshold=10000):
    '''Return the number of polynomial additions to compute the Groebner basis.

    Parameters
    ----------
    params : np.ndarray
        Parameters for the term order matrix.
    data : list
        List of generators.
    coeff_field : sage.rings.ring.Ring

    operation : {'sum', 'median'}, optional

    num_variables : int
        The number of variables.
    select_strategy : str, optional
        The strategy for selecting the S-pair. The default is 'degree'.
    stop_algorithm : bool, optional

    threshold : int, optional
        The threshold for the number of polynomial additions. The default is 10000.


    
    example:
    --------
    params = np.array([1,1,1,0,0,-1,0,-1,0])
    polynomial_list = [x^3 - 2*x*y, x^2*y - 2*y^2 + x]
    
    '''
    INF = 1e10

    assert len(params) == num_variables**2, "The number of parameters must be equal to the number of variables in the weight matrix."
    assert operation in ['sum', 'median'], "operation must be either 'sum' or 'median'"

    # 
    int_params = params.astype(int)
    int_matrix = int_params.reshape(num_variables, num_variables)
    if not validation(int_matrix):
        return INF

    
    matrix_ = matrix(num_variables, list(params)) # paramsがnp.ndarrayの場合うまくいかない
    
    P = PolynomialRing(coeff_field, num_variables, 'x', order=TermOrder(matrix_))
    
    additions = 0
    additions_list = np.array([])

    for polynomial_list in data:
        polynomial_list = list(map(P, polynomial_list))
        _, stat = buchberger(P, polynomial_list, select_strategy=select_strategy, stop_algorithm=stop_algorithm, threshold=threshold)
        additions = stat['polynomial_additions']

        if additions > threshold:
            return INF
        else:
            additions_list = np.append(additions_list, additions)

    if operation == 'sum':
        return np.mean(additions_list)
    elif operation == 'median':
        return  np.median(additions_list)
    else:
        raise ValueError('operation must be either "sum" or "median"')
    





def output_additions_log(params, data, coeff_field, num_variables=3, select_strategy='degree', stop_algorithm=False, threshold=10000):
    '''Return the number of polynomial additions to compute the Groebner basis.

    Parameters
    ----------
    params : np.ndarray
        Parameters for the term order matrix.
    data : list
        List of generators.
    coeff_field : sage.rings.ring.Ring
        
    num_variables : int
        The number of variables.
    select_strategy : str, optional
        The strategy for selecting the S-pair. The default is 'degree'.
    stop_algorithm : bool, optional

    threshold : int, optional
        The threshold for the number of polynomial additions. The default is 10000.


    
    example:
    --------
    params = np.array([1,1,1,0,0,-1,0,-1,0])
    polynomial_list = [x^3 - 2*x*y, x^2*y - 2*y^2 + x]
    
    '''
    INF = 1e10

    assert len(params) == num_variables**2, "The number of parameters must be equal to the number of variables in the weight matrix."

    int_params = params.astype(int)
    int_matrix = int_params.reshape(num_variables, num_variables)
    

    if not validation(int_matrix):
        print('invalid matrix')
        return INF


    print(int_matrix)
    
    matrix_ = matrix(num_variables, list(params)) # paramsがnp.ndarrayの場合うまくいかない
    
    P = PolynomialRing(coeff_field, num_variables, 'x', order=TermOrder(matrix_))
    print(TermOrder(matrix_)._singular_str)
    
    additions = 0
    sum_additions = 0

    for i, polynomial_list in enumerate(data):
        polynomial_list = list(map(P, polynomial_list))

        _, stat = buchberger(P, polynomial_list, select_strategy=select_strategy, stop_algorithm=stop_algorithm, threshold=threshold)
        # GB, stat = buchberger_log(P, polynomial_list, select_strategy=select_strategy)

        if stat['polynomial_additions'] > threshold:
            return INF

        additions = stat['polynomial_additions']
        print(f'{i}th additions: {additions}')
        sum_additions += additions
        
    ave_additions = sum_additions / len(data)

    return float(ave_additions)
    
    

# if __name__ == "__main__" and '__file__' in globals():

#     load("/app/src/lib/sampler.sage")
#     params = np.array([1,1,1,0,0,-1,0,-1,0])
#     field = GF(32003)
#     num_variables = 3
    
#     Ring = PolynomialRing(field, 3, 'x', order='lex')

#     conditions = {  'max_degree'      : 5, 
#                     'min_degree'      : 2,
#                     'max_num_terms'   : 10, 
#                     'max_coeff'       : 20,
#                     'num_bound'       : 20,
#                     'nonzero_instance': True,}

#     psampler = Polynomial_Sampler(Ring, 
#                                 degree_sampling='uniform', 
#                                 term_sampling='uniform', 
#                                 strictly_conditioned=True, 
#                                 conditions=conditions)

#     data = []
#     for i in range(10):
#         set_random_seed(i) # seed値の設定
#         F = psampler.sample(3)
#         data.append(F)
    
#     additions = output_additions(params, data, coeff_field=field, operation='sum', num_variables=num_variables, select_strategy='degree', stop_algorithm=False, threshold=10000)
#     # additions = output_additions_log(params, data, coeff_field=field, num_variables=num_variables, select_strategy='degree', stop_algorithm=False, threshold=10000)

#     print(additions)