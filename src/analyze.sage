import os
import time
import argparse
import numpy as np
import yaml
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd

 
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


# grevlex, grlex, lex でのデータを .npで保存したい
class GetResults():
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

        return ret

    def count(self, F_G_str, weight_matrix):

        F, _, num_tokens_F = self.preprocess_for_F(F_G_str)

        upper = self.upper_lim_for_num_tokens
        lower = self.lower_lim_for_num_tokens

        # Fのトークン数が指定範囲外の場合は無視
        if upper is not None and lower is not None:
            if num_tokens_F > upper or num_tokens_F < lower:
                return None

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

        return result

    def preprocess_for_F(self, F_G_str):
        F_str = F_G_str.split(':')[0]
        F_str = F_str.strip()
        num_tokens_F = int(len(F_str.split()))
        F_list = F_str.split('[SEP]')

        Ring = PolynomialRing(self.field, 'x', self.nvars, order='lex')
        F = [sequence_to_poly(f_str.strip(), Ring) for f_str in F_list] # infix以外はうまくいかない
        
        return F, F_str, num_tokens_F
    
class Writer():
    def __init__(self, results, save_dir, tag):

        self.results = results
        self.save_dir = save_dir
        self.tag = tag
        
    def save(self, term_order):
        data = self.preprocess()
        df = pd.DataFrame(data)
        os.makedirs(os.path.join(self.save_dir, "csv"), exist_ok=True)
        df.to_csv(os.path.join(self.save_dir, "csv", f'{self.tag}.{term_order}.csv'))


    def preprocess(self):
        results = self.results

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

        return data
    
    # def analyze(self, df):
    #     data = self.preprocess()

    
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
    parser.add_argument("--buchberger_timeout", type=int, default=10, help="buchbergerアルゴリズムのタイムアウト")
    parser.add_argument("--buchberger_threshold", type=int, default=100000, help="buchbergerアルゴリズムの打ち切り閾値(多項式加算数)")
    
    return parser


def main():
    parser = get_parser()
    params = parser.parse_args()

    field_name = params.field_name
    nvars = params.nvars
    threshold = params.buchberger_threshold
    timeout = params.buchberger_timeout
    test_dataset_size = params.test_dataset_size
    train_dataset_size = params.train_dataset_size
    input_dir = params.input_path
    save_dir = params.save_path

    with open(params.config_path, 'r') as f:
        config = yaml.safe_load(f)

    
    for key, value in config.items():
        print(f"{key}: {value}")
    
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

    for tag in ['test']:
        input_filename = base_name + f'.{tag}.lex.infix'
        
        if tag == 'test':
            dataset_size = test_dataset_size
        else:
            dataset_size = train_dataset_size

        for term_order in ['grevlex', 'grlex']:
            
            F_G_str_list = load_data(input_dir, filename=input_filename, dataset_size=dataset_size)
            
            get_results = GetResults(tag = tag, 
                                    F_G_str_list = F_G_str_list, 
                                    nvars = nvars, 
                                    field = field, 
                                    threshold=threshold, 
                                    timeout=timeout, 
                                    n_jobs=-1)

            results = get_results(term_order)
            writer = Writer(results=results, save_dir=save_dir, tag=tag)
            writer.save(term_order)


if __name__ == '__main__':
    main()


