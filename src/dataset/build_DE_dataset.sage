import os
import time
import argparse
import numpy as np
import yaml
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it 


 
load('src/dataset/symbolic_utils.sage')
load('src/lib/count_num_additions.sage')


def load_data(input_dir, filename, dataset_size):
    input_path = os.path.join(input_dir, filename)
    with open(input_path, 'r') as file:
        contents = file.read()

    contents = contents.strip()
    F_G_str_list = contents.split('\n') # 多項式系を要素として持つリスト

    assert len(F_G_str_list) >= dataset_size, f'dataset_size is too large. The number of samples in {filename} is {len(F_G_str_list)}'

    return F_G_str_list[:dataset_size]

class ChooseData():
    def __init__(self, tag, F_G_str_list, nvars, field, threshold, timeout, lower_lim_for_num_tokens=None, upper_lim_for_num_tokens=None, n_jobs=-1, ):
        
        self.tag = tag
        self.F_G_str_list = F_G_str_list
        self.nvars = nvars
        self.field = field
        self.threshold = threshold
        self.timeout = timeout 
        self.lower_lim_for_num_tokens = lower_lim_for_num_tokens
        self.upper_lim_for_num_tokens = upper_lim_for_num_tokens
        self.n_jobs = n_jobs

        
    def __call__(self, weight_matrix):
        ret = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=True)(delayed(self.count)(F_G_str, weight_matrix) for F_G_str in self.F_G_str_list)

        results, F_stats, G_stats = zip(*ret)

        return results, F_stats, G_stats

    def count(self, F_G_str, weight_matrix):

        F, num_tokens_F, G, _ = self.preprocess(F_G_str)
        

        F_stat = self.get_system_stat(F)
        G_stat = self.get_system_stat(G)

        upper = self.upper_lim_for_num_tokens
        lower = self.lower_lim_for_num_tokens

        # Fのトークン数が指定範囲外の場合は無視
        if upper is not None and lower is not None:
            if num_tokens_F > upper or num_tokens_F < lower:
                return None, None, None

        num_additions_counter = NumAdditionsCounter(coeff_field = self.field,
                                                    num_variables = self.nvars,
                                                    select_strategy = 'normal',
                                                    threshold = self.threshold,
                                                    timeout= self.timeout,
                                                    check_gb=False)
        
        result = num_additions_counter.run(weight_matrix=weight_matrix, polynomial_list=F)
        result.F_G_str = F_G_str
        result.polynomial_list = F
        result.weight_matrix = weight_matrix

       
        return result, F_stat, G_stat
        

    def preprocess(self, F_G_str):
        F_str = F_G_str.split(':')[0]
        F_str = F_str.strip()
        G_str = F_G_str.split(':')[1]
        G_str = G_str.strip()

        num_tokens_F = int(len(F_str.split()))
        F_list = F_str.split('[SEP]')
        num_tokens_G = int(len(G_str.split()))
        G_list = G_str.split('[SEP]')

        Ring = PolynomialRing(self.field, 'x', self.nvars, order='lex')
        F = [sequence_to_poly(f_str.strip(), Ring) for f_str in F_list] # infix以外はうまくいかない
        G = [sequence_to_poly(g_str.strip(), Ring) for g_str in G_list]

        
        return F, num_tokens_F, G, num_tokens_G
    
    

    def get_system_stat(self, P):
        '''
        P: set of polynomials (column vector with polynomial entries)
        '''
        if not isinstance(P, list): P = P.list()
        size = len(P)
        degrees = [int(p.total_degree()) for p in P]
        num_monoms = [int(len(p.monomials())) for p in P]

        # max_coeff = max([max([abs(c) for c in p.coefficients()]) for p in P])
        field = P[0].base_ring()
        coeffs = it.chain(*[p.coefficients() for p in P])
        if field == QQ: coeffs = [c.numer() for c in coeffs] + [c.denom() for c in coeffs]
        if field.characteristic() == 0: coeffs = [abs(c) for c in coeffs]
        max_coeff = max(coeffs)
        
        stat = {'size': int(size), 
                'degrees': degrees,
                'num_monoms': num_monoms,
                'min_degree': float(np.min(degrees)),
                'min_num_monoms': float(np.min(num_monoms)),
                'max_degree': float(np.max(degrees)),
                'max_num_monoms': float(np.max(num_monoms)),
                'total_num_monoms': float(np.sum(degrees)),
                'max_coeff': int(max_coeff),
                # 'is_GB': int(self.check_gb(P)) if field is not RR else int(-1)
                }
        
        return stat 
    
def summarize_stats(stats, metric=['mean', 'std', 'max', 'min', 'median']):
    summary = {}
    for k in stats[0]:
        if isinstance(stats[0][k], list): continue
        if 'mean' in metric:
            summary[f'{k}_mean'] = float(np.mean([item[k] for item in stats]))
        if 'median' in metric:    
            summary[f'{k}_median'] = float(np.median([item[k] for item in stats]))
        if 'std' in metric:    
            summary[f'{k}_std'] = float(np.std([item[k] for item in stats]))
        if 'max' in metric:    
            summary[f'{k}_max'] = float(np.max([item[k] for item in stats]))
        if 'min' in metric:    
            summary[f'{k}_min'] = float(np.min([item[k] for item in stats]))
    return summary
    
class Writer():
    def __init__(self, results, save_dir, tag):

        self.results = results
        self.save_dir = save_dir
        self.tag = tag
        
    def save_csv(self, term_order):
        data = self.preprocess_for_results()
        df = pd.DataFrame(data)

        os.makedirs(os.path.join(self.save_dir, "csv"), exist_ok=True)
        df.to_csv(os.path.join(self.save_dir, "csv", f'{self.tag}.{term_order}.csv'))


    def save_dataset(self, filename):
        F_G_str_list = [result.F_G_str for result in self.results if result is not None and result.success]
        dataset_str = '\n'.join(F_G_str_list)

        os.makedirs(os.path.join(self.save_dir, "dataset"), exist_ok=True)
        with open(os.path.join(self.save_dir, "dataset", filename), 'w') as f:
            f.writelines(dataset_str)


    
    def preprocess_for_results(self):
        results = self.results

        success_list = [r.success if r is not None else np.nan for r in results]
        
        print(f'-------------{self.tag} dataset----------------')
        print('success rate : ', np.mean(success_list))
        print('the number of success samples : ', np.sum(success_list))
       

        data = {
            'success': [r.success if r is not None else np.nan for r in results],
            'polynomial_additions': [r.polynomial_additions if r is not None else np.nan for r in results],
            'mono_div_steps': [r.mono_div_steps if r is not None else np.nan for r in results],
            'total_time': [r.total_time if r is not None else np.nan for r in results],
            'reduction_time': [r.reduction_time if r is not None else np.nan for r in results],
            'selection_time': [r.selection_time if r is not None else np.nan for r in results],
            'variale_order' : [r.variable_order if r is not None else np.nan for r in results],
            'weight_matrix': [r.weight_matrix if r is not None else np.nan for r in results],
            'polynomial_list': [r.polynomial_list if r is not None else np.nan for r in results],
            'F_G_str': [r.F_G_str if r is not None else np.nan for r in results]
        }

        # except for the failed samples 
        polynomial_additions_list = [r.polynomial_additions for r in results if r is not None and r.success]
        total_time_list = [r.total_time for r in results if r is not None and r.success]
        print('average of polynomial additions (except for the failed samples) : ', np.mean(polynomial_additions_list))
        print('max of polynomial additions (except for the failed samples) : ', np.max(polynomial_additions_list))
        print('average runtime (except for the failed samples) : ', np.mean(total_time_list))
        print('max runtime (except for the failed samples) : ', np.max(total_time_list))
        print('sum runtime (1st ~ 1000th) : ', np.sum(total_time_list[:1000]))
        print('\n\n')

        return data
    
    
        
   

    
def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="analyze the results of Buchberger algorithm")

    # main parameters
    parser.add_argument("--save_path", type=str, help="save path")
    parser.add_argument("--input_path", type=str, help="input path")
    parser.add_argument("--config_path", type=str, help="config path")
    parser.add_argument("--nvars", type=int, default=3, help="変数の数")
    parser.add_argument("--field_name", type=str, default='GF7', help="体の名前")
    parser.add_argument("--test_dataset_size", type=int, default=1000, help="the size of test dataset")
    parser.add_argument("--train_dataset_size", type=int, default=1000, help="the size of train dataset")
    parser.add_argument("--buchberger_select_strategy", type=str, default='normal', help="Sペアの選択戦略")
    parser.add_argument("--buchberger_train_timeout", type=float, default=10, help="buchbergerアルゴリズムのタイムアウト")
    parser.add_argument("--buchberger_test_timeout", type=float, default=10, help="buchbergerアルゴリズムのタイムアウト")
    parser.add_argument("--buchberger_threshold", type=int, default=100000, help="buchbergerアルゴリズムの打ち切り閾値(多項式加算数)")
    
    return parser


def main():
    parser = get_parser()
    params = parser.parse_args()

    field_name = params.field_name
    nvars = params.nvars
    threshold = params.buchberger_threshold
    dict_timeout = {'test': params.buchberger_test_timeout, 'train': params.buchberger_train_timeout}
    dict_dataset_size = {'test': params.test_dataset_size, 'train': params.train_dataset_size}
    input_dir = params.input_path
    save_dir = params.save_path

    with open(params.config_path, 'r') as f:
        config = yaml.safe_load(f)

    print('-------------config----------------')
    for key, value in config.items():
        print(f"{key}: {value}")
    print('\n\n')

    print('-------------buchberger----------------')
    print('threshold : ', threshold)
    print('train timeout (by grevlex) : ', dict_timeout['train'])
    print('test timeout (by grevlex): ', dict_timeout['test'])
    print('\n\n')

    
    if field_name == 'QQ':
        field = QQ
    elif field_name == 'RR':
        field = RR
    elif field_name == 'ZZ':
        field = ZZ
    elif field_name[:2] == 'GF':
        order = int(field_name[2:])
        field = GF(order)

    base_name =f'data_{field_name}_n={nvars}'

    for tag in ['train', 'test']:
        filename = base_name + f'.{tag}.lex.infix'
        dataset_size = dict_dataset_size[tag]
        timeout=dict_timeout[tag]

        for term_order in ['grevlex']:
            
            F_G_str_list = load_data(input_dir, filename=filename, dataset_size=dataset_size)
            
            choose_data = ChooseData(
                tag = tag, 
                F_G_str_list = F_G_str_list, 
                nvars = nvars, 
                field = field, 
                threshold=threshold, 
                timeout=timeout, 
                n_jobs=-1
                )
            
            

            results, F_stats, G_stats = choose_data(term_order)

            writer = Writer(results=results, save_dir=save_dir, tag=tag)
            writer.save_csv(term_order)
            writer.save_dataset(filename)


if __name__ == '__main__':
    main()


