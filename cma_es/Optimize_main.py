# coding: utf-8

import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn-darkgrid')

def Eval_Func(X, func_type = "Rastr"):
    val = 0.0
    if (func_type == "Rosen"):
        for k in range(len(X) - 1):
            val += 100* (X[k+1] - X[k]**2)**2 + (X[k] - 1)**2
    
    elif (func_type == "Rastr"):
        for k in range(len(X)):
            val += 10 + ((X[k] - 15)**2 - 10*np.cos(2*np.pi*X[k]))
    return val

def CmaesPlot(X, Y, Z, k):
    fig = plt.figure(figsize=(5, 5))
    # ax = Axes3D(fig)

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")

    x0 = np.arange(-20, 20, 0.1)
    x1 = np.arange(-20, 20, 0.1)
    X0, X1 = np.meshgrid(x0, x1)
    
    YY = np.array([[Eval_Func([X0[i][j], X1[i][j]]) for j in range(len(X0[0]) ) ] for i in range(len(X0))])

    # ax.plot(X, Y, Z, color = "black", marker="o",linestyle='None')
    # ax.plot([0], [0], [0], color = "red", marker="o",linestyle='None')

    # ax.plot_wireframe(X0, X1, YY)
    # ax.plot_surface(X0, X1, YY, cmap = "plasma_r", linewidth=0.3)
    plt.plot(X, Y, color = "black", marker="o",linestyle='None')
    plt.plot([15], [15], color = "red", marker="o",linestyle='None')
    plt.contourf(X0, X1, YY, cmap = "plasma_r")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)

    # plt.show()
    plt.savefig('figure_' + str(k) + '.png')


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
    x_min = -20
    x_max = 20
    lmda = 50
    sigma = 0.5  # Step_size
    mu = int(np.floor(lmda/2))
    x_mean = (x_max - x_min) * np.random.rand(n) + x_min # 一様分布
    counteval = 0
    stopeval = 500*n**2
    eigeneval = 0
    stop_fitness = 1e-10
    Flag_loop = True

    B = np.eye(n) # 固有ベクトル n*n
    D = np.ones(n) # 固有値  n
    C = B @ np.diag(D**2) @ B.T
    C_invsqrt = B @ np.diag(1/D) @ B.T
    print (D)
    print ("C_invsqrt >> ")
    print (C_invsqrt)

    weights = np.log(mu+1/2) - np.log(np.linspace(1, mu, mu))
    weights = copy.deepcopy(weights/ np.sum(weights)) # 重みの正規化
    MUeff = np.sum(weights)**2/np.sum(weights**2)

    p_c = np.zeros(n) # Covriance matrix の進化パス
    p_s = np.zeros(n) # sigma の進化パス
    Cc = (4 + MUeff/n)/(n+4 + 2*MUeff/n) # p_c(k)とp_c(k+1)の結合比率を表す重みパラメータ 
    Cs = (MUeff+2)/(n+MUeff+5) # p_s(k)とp_s(k+1)の結合比率を表す重みパラメータ
    C1 = 2/((n+1.3)**2 + MUeff)  # Covriance matrix の　Rank-one update の学習率
    Cmu = np.min(np.array([1-C1, 2* (MUeff - 2 + 1/MUeff)/((n+2)**2 + MUeff)])) # Covriance matrix の　rank-mu update の学習率
    Ds = 1 + np.max(np.array([0, np.sqrt((MUeff - 1)/(n+1))-1])) + Cs # sigmaの減衰パラメータ
    E_Normrand = np.sqrt(n)*(1 - 1/(4*n) + 1/(21*n**2)) # 確率分布から生成された値の，ユークリッドノルムの値の期待値 変数が増えると少しづつ増加

    k = 0
    while ((counteval < stopeval) & (Flag_loop == True)):
        print ("c_eval >> ", counteval)
        k += 1
        # Basic equattion sampling
        X = np.array([x_mean + sigma * B @ (D* np.random.normal(0, 1, (n))) for i in range(lmda)])
        Fitness = [Eval_Func(X[i][:]) for i in range(lmda)]
        counteval += lmda
        print ("X >>")
        print (X)

        X_T = X.T
        CmaesPlot(X_T[0][:], X_T[1][:], Fitness, k)

        # Sort -> Fitness, Fitness_index
        Fitness_sort = np.sort(Fitness)
        Fitness_sort_inx = np.argsort(Fitness)
        X_sort = np.array(X[Fitness_sort_inx])[:mu]

        # Selection and Recombination: Moving the Mean
        x_old = copy.deepcopy(x_mean)
        x_mean = x_old + np.sum(weights.reshape(-1, 1)*(X_sort - x_old), axis = 0)

        # Update evolution paths (Ps, Pc)
        p_s = (1 - Cs)*p_s + np.sqrt(Cs*(2 - Cs)*MUeff) * C_invsqrt @ ((x_mean - x_old)/sigma)
        h_sigma = 1 if ( (np.linalg.norm(p_s, ord = 2) * E_Normrand)/ \
            np.sqrt(1 - (1 - Cs)**(2 * counteval/lmda) ) )  < 1.4 + 2/(n + 1) else 0
        p_c = (1 - Cc)*p_c + h_sigma * np.sqrt(Cc*(2 - Cc)*MUeff) * ((x_mean - x_old)/sigma)

        # Adapting the Covariance Matrix 
        y_tmp = (X_sort - x_old)/sigma
        # print ("y_tmp, ", y_tmp)
        # print (weights)

        C = (1 - C1 - Cmu)*C + \
            C1 * (p_c @ p_c.T + \
                (1 - h_sigma) * Cc * (2 - Cc) * C ) + \
            Cmu * (y_tmp.T @ np.diag(weights) @ y_tmp) # np.sum(weights.reshape(-1,1)*y_tmp, axis = 0)
        
        # print ("cul_C >>")
        # print (C)

        # Adapting the step-size sigma
        sigma = sigma * np.exp( (Cs/Ds) *(np.linalg.norm(p_s, ord = 2)/ E_Normrand - 1))
        # print (sigma)

        # Decomposition of Covriance matrix => B @ diag(D**2) @ B.T 
        if (counteval - eigeneval > lmda/(C1 + Cmu)/n/10):
            eigeneval = counteval
            C = np.triu(C) + np.triu(C).T - np.diag(C.diagonal())
            D, B = np.linalg.eig(C)
            # print ("tri_C >")
            # print (C)
            # print ("D >")
            # print (D)
            D = np.sqrt(D)
            C_invsqrt =  B @ (1/D) @ B.T
        # print ("D >")
        # print (D)

        # print ("C_invsqrt >> ")
        # print (C_invsqrt)

        if Fitness_sort[0] < stop_fitness :
            Flag_loop = False
        print ("best_x >> ", X_sort[0][:])





