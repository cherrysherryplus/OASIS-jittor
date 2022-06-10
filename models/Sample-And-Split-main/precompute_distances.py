import time
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from mmd_numpy_sklearn import mmd_rbf
# from tqdm import tqdm


#定义一个initializer函数
def init_pool(array,data):
    global glob_array #定义全局变量
    global data_scaled
    glob_array = array
    data_scaled = data
    
    
def my_func(i,array_SHAPE):
    for j in range(array_SHAPE[1]):
        si,sj = 29*i, 29*j
        ei,ej = si+29,sj+29
        # print(si,ei)
        glob_array[array_SHAPE[1]*i+j]  = mmd_rbf(data_scaled[np.newaxis,si:ei], data_scaled[np.newaxis,sj:ej])


def check_matrix_legal(npy_file):
    s = time.time()
    matrix = np.load(npy_file)
    e = time.time()
    print("load takes %s second" % eval(f"{e}-{s}"))
    return matrix.shape == (10000,10000)


def precompute_ed_distances(data_scaled, npy_file):
    s = time.time()
    ed_dists = pdist(data_scaled, 'sqeuclidean')
    ed_matrix = squareform(ed_dists)
    e = time.time()
    print("ed takes: %.4f minutes" % eval(f"({e}-{s})/60"))
    np.save(npy_file, ed_matrix)
    assert check_matrix_legal(npy_file), "check ed distances matrix failed"
    print("ed done")


def precompute_mmd_distances(data_scaled, npy_file):
    array = np.zeros((10000,10000))

    func_partial = partial(my_func, array_SHAPE = array.shape)
    array_shared = multiprocessing.RawArray('d', array.ravel()) 
    data_shared = multiprocessing.RawArray('d', data_scaled.ravel())
    data_shared = np.frombuffer(data_shared, dtype=np.float64)
    
    # 由于对矩阵元素的写入是有位置坐标的（各进程可以各自改动对应矩阵位置(即内存地址)处的值，故无需加进程锁
    p = multiprocessing.Pool(processes=8, initializer=init_pool, 
                             initargs=(array_shared,data_shared))
    
    # 定义一个进程池，指定进程数量(processes)，初始化函数(initializer)以及初始化函数中的输入(initargs)
    # 此处使用进程池map()函数对numpy矩阵维度0(行)进行迭代
    p.map(func_partial, range(array.shape[0]))
    
    # 此时可视为map函数向子进程中分配不同的行，各个子进程在分配的不同行中各自处理整行的数据。
    p.close()
    p.join()
    
    # 需要注意，此处要使用array_shared，而不是glob_array.
    # glob_array为子进程中的全局变量，在主进程中并未被定义，但主进程中的array_shared与子进程中的glob_radioScore_array指向同一内存地址.
    new_array = np.frombuffer(array_shared, np.double).reshape(array.shape)
    np.save(npy_file,new_array)
    assert check_matrix_legal(npy_file), "check mmd distances matrix failed"
    print("mmd done")


distances_func = {
    "mmd": precompute_mmd_distances,
    "ed": precompute_ed_distances,
}


def precompute_distances(distances_option, data_scaled, matrix_path):
    assert distances_option in distances_func.keys(), "distances func key not exists"
    return distances_func[distances_option](data_scaled, matrix_path)


if __name__ == '__main__':
    csv_file = "train.csv"
    data = pd.read_csv(csv_file, index_col=0)
    data_scaled = normalize(data)
    # ed
    ed_npy = "ed.npy"
    precompute_ed_distances(data_scaled, ed_npy)
    # 分割线
    print("-"*30)
    # mmd
    mmd_npy = "mmd_test.npy"
    precompute_mmd_distances(data_scaled, mmd_npy)
    
