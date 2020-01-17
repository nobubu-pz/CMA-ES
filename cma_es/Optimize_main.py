# coding: utf-8

import numpy as np
import copy

def Eval_Func(X, func_type = "Rosen"):
    val = 0.0

    if (func_type == "Rosen"):
        for k in range(len(X) - 1):
            val += 100* (X[k+1] - X[k]**2)**2 + (X[k] - 1)**2
    
    return val

# def _siguma(X):


if __name__ == "__main__":
    #########################################
    ###
    ### n    : 変数数
    ### lmda : サンプル数
    ### 
    ### C    : 共分散行列
    ### 
    ### 
    #########################################
    opt_type = "Minimize"

    n = 2 # 設計変数の次元数
    lmda = 10
    sigma = 0.3  # Step_size
    mu = int(np.floor(lmda/2))
    x_mean = np.random.normal(0, 1, (n))
    Z = np.random.normal(0, 1, (lmda, n))
    
    B = np.eye(n) 
    D = np.ones(n) # 固有値 
    C = np.dot(np.dot(B, np.diag(D**2)), B.T)
    C_inverse = np.dot(np.dot(B, np.diag(np.sqrt(D))), B.T)

    weights = np.log(mu+1/2) - np.log(np.linspace(1, mu, mu))
    weights = copy.deepcopy(weights/ np.sum(weights)) # 重みの正規化
    MUeff = np.sum(weights)**2/np.sum(weights**2)

    p_c = np.zeros(n)
    p_s = np.zeros(n)
    Cc = (4 + MUeff/n)/(n+4 + 2*MUeff/n)
    Cs = (MUeff+2)/(n+MUeff+5)
    C1 = 2/((n+1.3)**2 + MUeff)  # Rank-one update の学習率
    Cmu = np.min(np.array([1-C1, 2* (MUeff - 2 + 1/MUeff)/((n+2)**2 + MUeff)]))

    # Basic equattion sampling
    X = np.array([x_mean + sigma * (D* np.random.normal(0, 1, (n))) for i in range(lmda)])
    Fitness = [Eval_Func(X[i][:]) for i in range(lmda)]

    # Sort -> Fitness, Fitness_index
    Fitness_sort = np.sort(Fitness)
    Fitness_sort_inx = np.argsort(Fitness)
    X_sort = np.array(X[Fitness_sort_inx])[:mu]

    # Selection and Recombination: Moving the Mean
    x_old = copy.deepcopy(x_mean)
    x_mean = x_old + np.sum(weights.reshape(-1, 1)*(X_sort - x_old), axis = 0)

    print (x_mean)
    print (X)
    print (Fitness)
    print (x_mean)

    # Update evolution paths (Ps, Pc)
    p_c = (1 - Cc)*p_c + np.sqrt(Cc*(2 - Cc)*MUeff) * ((x_mean - x_old)/sigma)
    # hsig = 
    p_s = (1 - Cs)*p_s + np.sqrt(Cs*(2 - Cs)*MUeff) * np.dot(C_inverse, ((x_mean - x_old)/sigma))
    C = (1 - C1 - Cmu)

    print (C)
    # Adapting the Covariance Matrix 



