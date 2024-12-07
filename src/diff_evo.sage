import numpy as np
import pprint
import argparse
import os
import datetime
from joblib import Parallel, delayed
from time import time 
from scipy.optimize import differential_evolution
from zoneinfo import ZoneInfo


load("/app/src/lib/sampler.sage")
load("/app/src/lib/count_num_additions.sage")
load('src/dataset/symbolic_utils.sage')

class LoadData():
    def __init__(self, nvars, field, input_dir, filename, dataset_size):
        self.nvars = nvars
        self.field = field
        self.input_dir = input_dir
        self.filename = filename
        self.dataset_size = dataset_size
    
    def __call__(self):
        input_path = os.path.join(self.input_dir, self.filename)
        with open(input_path, 'r') as file:
            contents = file.read()

        contents = contents.strip()
        F_G_str_list = contents.split('\n')

        assert len(F_G_str_list) >= self.dataset_size, f'dataset_size is too large. The number of samples in {self.filename} is {len(F_G_str_list)}'
        F_G_str_list = F_G_str_list[:self.dataset_size]

        data = list(map(self.preprocess_for_F, F_G_str_list))

        return data


    def preprocess_for_F(self, F_G_str):
        F_str = F_G_str.split(':')[0]
        F_str = F_str.strip()
        # num_tokens_F = int(len(F_str.split()))
        F_list = F_str.split('[SEP]')

        Ring = PolynomialRing(self.field, 'x', self.nvars)
        F = [sequence_to_poly(f_str.strip(), Ring) for f_str in F_list] # infix以外はうまくいかない

        return F


def func(params, data, coeff_field, operation='mean', num_variables=3, select_strategy='normal', threshold=10000, timeout=5):
    INF = 1e10

    assert len(params) == num_variables**2, "The number of parameters must be equal to the number of entries in the weight matrix."
    assert operation in ['mean', 'median', 'max'], "operation must be either 'mean', 'max' or 'median'"

    int_params = params.astype(int)
    int_matrix = int_params.reshape(num_variables, num_variables)
    if not validation(int_matrix):
        return INF

    weight_matrix = params.reshape(num_variables, num_variables)
    
    num_additions_counter = NumAdditionsCounter(coeff_field = coeff_field,
                                            num_variables = num_variables,
                                            select_strategy = select_strategy,
                                            threshold = threshold,
                                            timeout = timeout)

    ret = Parallel(n_jobs=-1, backend="multiprocessing", verbose=True)(delayed(num_additions_counter.run)(weight_matrix=weight_matrix, polynomial_list=F) for F in data)
    additions_list = [r.polynomial_additions if r.success else INF for r in ret]
    bad_list = [r.polynomial_additions for r in ret if not r.success]

    
    # success_list = [r for r in ret if r.success]
    # if success_list:
    #     content = max(success_list, key=lambda x: x.elapsed_time) # 最大時間のものを取得
    #     print('success : ', content.elapsed_time, content.polynomial_additions, content.mono_div_steps)

    # fail_list = [r for r in ret if not r.success] 
    # if fail_list:
    #     content_max = max(fail_list, key=lambda x: x.elapsed_time) # 最大時間のものを取得
    #     content_min = min(fail_list, key=lambda x: x.elapsed_time) # 最小加算数のものを取得
    #     print()
    #     # print(content_max.elapsed_time, content_max.polynomial_list, content_max.mono_div_steps, list(content_max.weight_matrix))
    #     print(content_max.elapsed_time, content_max.mono_div_steps)

    #     print()
    #     # print(content_min.elapsed_time,  content_min.polynomial_list, content_min.mono_div_steps, list(content_min.weight_matrix))
    #     print(content_min.elapsed_time, content_min.mono_div_steps)


    if operation == 'mean':
        mean_additions = np.mean(additions_list)
        return mean_additions
    elif operation == 'median':
        median_additions = np.median(additions_list)
        return  median_additions
    elif operation == 'max':
        max_additons = np.max(additions_list)
        return max_additons
    else:
        raise ValueError('operation must be either "mean", "max" or "median"')

    


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="differential evolution")

    # main parameters
    parser.add_argument("--save_path", type=str, help="save path")
    parser.add_argument("--input_path", type=str, help="input path")
    parser.add_argument("--nvars", type=int, default=3, help="変数の数")
    parser.add_argument("--field_name", type=str, default='GF7', help="体の名前")
    parser.add_argument("--dataset_size", type=int, default=1000, help="多項式系の数")
    parser.add_argument("--upper_bound", type=int, default=1000, help="重み行列の成分の上限値")
    parser.add_argument("--lower_bound", type=int, default=-1000, help="重み行列の成分の下限値")
    parser.add_argument("--diff_evo_popsize", type=int, default=15, help="進化計算でのpopsize")
    parser.add_argument("--diff_evo_maxiter", type=int, default=1000, help="maxiter")
    parser.add_argument("--diff_evo_workers", type=int, default=1, help="進化計算での並列数")
    parser.add_argument("--diff_evo_updating", type=str, default='immediate', help="進化計算での解ベクトルの更新")
    parser.add_argument("--buchberger_select_strategy", type=str, default='normal', help="Sペアの選択戦略")
    parser.add_argument("--buchberger_timeout", type=int, default=10, help="buchbergerアルゴリズムのタイムアウト")
    parser.add_argument("--buchberger_threshold", type=int, default=100000, help="buchbergerアルゴリズムの打ち切り閾値(多項式加算数)")
    parser.add_argument("--operation", type=str, default='mean', help="DEでの目的関数での処理")
    return parser

if __name__ == '__main__':

    now = datetime.datetime.now(ZoneInfo("Asia/Tokyo"))
    datetime_str = now.strftime("%Y%m%d_%H%M%S")
    print(datetime_str)

    parser = get_parser()
    params = parser.parse_args()
    nvars = params.nvars
    field_name = params.field_name
    dataset_size = params.dataset_size
    lower = params.lower_bound
    upper = params.upper_bound
    select_strategy = params.buchberger_select_strategy
    threshold = params.buchberger_threshold
    timeout = params.buchberger_timeout

    
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
    input_filename = base_name + '.train.lex.infix'
    input_dir = params.input_path
    load_data = LoadData(nvars = nvars,
                     field = field,
                     input_dir = input_dir, 
                     filename = input_filename, 
                     dataset_size = dataset_size)
    
    data = load_data()

    
    bounds = [(0, upper)] * nvars + [(lower, upper)] * (nvars ** 2 - nvars)

    # differential_evolution の実行
    start = time()
    try:
        result = differential_evolution(func=func, 
                                        bounds=bounds, 
                                        popsize=params.diff_evo_popsize, 
                                        disp=True, 
                                        workers=params.diff_evo_workers, 
                                        updating=params.diff_evo_updating, 
                                        maxiter=params.diff_evo_maxiter, 
                                        args=(data, field, params.operation, nvars, select_strategy, threshold, timeout))
        pprint.pprint(result)
    except Exception as e:
        print(f"An error occurred: {e}")
    
    elapsed_time = time() - start

    time_h = int(elapsed_time / (60 * 60))
    time_m = int((elapsed_time % (60 * 60)) / 60)
    time_s = int(elapsed_time % 60)

    save_dir = params.save_path
    f = open(os.path.join(save_dir, f'result_{datetime_str}.txt'), "w")

    f.write('Date: ' + datetime_str + '\n\n')

    f.write('updating: ' + params.diff_evo_updating + '\n')
    f.write('operation: ' + params.operation + '\n')
    f.write('nvars: ' + str(nvars) + '\n')
    f.write('dataset_size: ' + str(dataset_size) + '\n')
    f.write('upper_bound: ' + str(upper) + '\n')
    f.write('lower_bound: ' + str(lower) + '\n')
    f.write('buchberger_threshold: ' + str(threshold) + '\n\n\n')

    f.write('success: ' + str(result.success) + '\n')
    f.write('message: ' + str(result.message) + '\n')
    f.write('fun: ' + str(result.fun) + '\n')
    f.write('nit: ' + str(result.nit) + '\n')
    f.write('nfev: ' + str(result.nfev) + '\n')
    # f.write(str(result) + '\n\n')
    f.write('population size: ' + str(result.population.shape[0]) + '\n')
    f.write('elapsed time : {0:02}:{1:02}:{2:02} [HH:MM:SS]'.format(time_h, time_m, time_s) + '\n\n')

    weight_matrix = result.x
    weight_matrix_int = weight_matrix.astype(int)
    weight_matrix = weight_matrix.tolist()
    weight_matrix_int = weight_matrix_int.tolist()
    f.write('x: ' + str(weight_matrix) + '\n\n')
    f.write('x: ' + str(weight_matrix_int) + '\n\n')

    
    f.close()

