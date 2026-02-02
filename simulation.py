from ast import If
from scipy.optimize import root, least_squares
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from matplotlib import font_manager as fm


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.unicode_minus'] = False
font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc'
fp = fm.FontProperties(fname=font_path, size=18)



def generate_parameters(I, A, sigma0 = 5, population = 1000, random: bool = False, seed = 4, verbose = False):
    if not random:
        mu    = np.ones(I) * (1/I)
        alpha = np.ones((I,I,A)) / (I+1)
        sigma = np.ones(I) * sigma0
        beta = np.ones(I)
        tau = np.ones((I,A,A))
        lam = np.zeros(A)
        zeta = np.ones((I,A))
        theta = np.ones(A)
        N = np.ones(A) * population
        NN = np.ones((I, A)) * population / I
    
    else:
        np.random.seed(seed)

        mu    = np.random.uniform(1.0, 3.0, I)
        mu    = mu / np.sum(mu)

        alpha0 = np.random.uniform(1.0, 3.0, (I+1, I, A))
        alpha = np.zeros((I,I,A))
        for i in range(I):
            for a in range(A):
                for j in range(I):
                    alpha[j,i,a] = alpha0[j,i,a] / np.sum(alpha0[:,i,a])

        sigma = np.random.uniform(5.0, 10.0, I)
        beta  = np.random.uniform(0.5, 2.0, I)

        tau = np.zeros((I, A, A))
        for i in range(I):
            for a in range(A):
                tau[i, a, a] = 1.0

                for b in range(a+1, A):
                    tau[i, a, b] = np.random.uniform(1.0,2.0)
                    tau[i, b, a] = tau[i, a, b]

        lam   = np.random.uniform(0.1, 0.9, A)
        lam = np.zeros(A)

        zeta  = np.random.uniform(0.5, 2.0, (I, A))
        for a in range(A):
            zeta[I-1, a] = 1.0
        
        theta = np.random.uniform(0.1, 3.0, A)
        NN     = np.random.uniform(400, 800, (I, A))

        N = np.empty(A)
        for a in range(A):
            N[a] = np.sum(NN[:,a])
        

    if verbose:
        print("------------ Generated parameters ------------")
        print("mu =", mu)
        print("alpha =", alpha)
        print("sigma =", sigma)
        print("beta =", beta)
        print("tau =", tau)
        print("lam =", lam)
        print("zeta =", zeta)
        print("theta =", theta)
        print("NN =", NN)


    return {
        'I': I,
        'A': A,
        'mu': mu,
        'alpha': alpha,
        'sigma': sigma,
        'beta': beta,
        'tau': tau,
        'lam': lam,
        'zeta': zeta,
        'theta': theta,
        'N': N,
        'NN': NN,
        'W_all': np.sum(NN),
    }



def compute_changes(data, verbose: bool = False, TOL_F = 1e-10):

    def compute_aux_parameters(phi, w, params):
        A, I = params['A'], params['I']

        alpha = params['alpha']
        sigma = params['sigma']
        beta = params['beta']
        tau = params['tau']
        lam = params['lam']
        NN = params['NN']

        
        psi = np.empty(I)
        for i in range(I):
            psi[i] = sigma[i] * beta[i] / (sigma[i]-1)
        
        alpha_labor = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                alpha_labor[i,a] = 1 - np.sum(alpha[:,i,a])

        w_hat = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                w_hat[i,a] = (1 - lam[a]) * w[i,a]

        if 'gamma' in params:
            gamma = params['gamma']
        else:
            gamma = np.empty((I,A))
            for i in range(I):
                for a in range(A):
                    gamma[i,a] = np.prod(psi ** alpha[:,i,a])

        n = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                n[i,a] = NN[i,a] * w[i,a] / (alpha_labor[i,a] * sigma[i] * phi[i,a])

        rho_term = np.empty((I,A,A))
        for i in range(I):
            for a in range(A):
                for b in range(A):
                    rho_term[i,a,b] = n[i,b] * (tau[i,b,a] * phi[i,b]) ** (1 - sigma[i])
            
        rho_hat = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                rho_hat[i,a] = np.sum(rho_term[i,a,:]) ** (1 / (1 - sigma[i]))
        
        return {
            'psi': psi,
            'alpha_labor': alpha_labor,
            'w_hat': w_hat,
            'gamma': gamma,
            'n': n,
            'rho_term': rho_term,
            'rho_hat': rho_hat,
        }


    def compute_equilibrium(
            parameters,
            phi_initial = None,
            w_initial = None,
            verbose: bool = False,
            initial_computed: bool = False,
            TOL_F = 1e-8,
            delta = 1.0,
            ):
        def compute_phi_and_w(
                parameters,
                phi_initial = None,
                w_initial = None,
                initial_computed: bool = False,
            ):

            def function(x, params):
                A, I = params['A'], params['I']

                X = np.append(x, 1.0)
                X = X.reshape((2, I, A))

                phi = X[0]
                w = X[1]


                mu, alpha = params['mu'], params['alpha']
                sigma, beta = params['sigma'], params['beta']
                tau, lam = params['tau'], params['lam']
                zeta, theta = params['zeta'], params['theta']
                N = params['N']
                NN = params['NN']

                
                aux_params = compute_aux_parameters(phi, w, params)

                psi = aux_params['psi']
                alpha_labor = aux_params['alpha_labor']
                w_hat = aux_params['w_hat']
                gamma = aux_params['gamma']
                n = aux_params['n']
                rho_term = aux_params['rho_term']
                rho_hat = aux_params['rho_hat']


                F1 = np.empty((I, A))
                F2 = np.empty((I-1, A))
                

                for i in range(I):
                    for a in range(A):

                        prod_term = np.empty(I)
                        for j in range(I):
                            if alpha[j,i,a] > 0:
                                prod_term[j] = (rho_hat[j,a] / alpha[j,i,a]) ** alpha[j,i,a]
                            else:
                                prod_term[j] = 1

                        f1_right = gamma[i,a] * (w[i,a] / alpha_labor[i,a]) ** alpha_labor[i,a] * np.prod(prod_term)

                        F1[i,a] = phi[i,a] - f1_right

                        if not i==I-1:  # その他の産業 i=I-1 の需給均衡条件 F2 は考えない
                            f2_left = sigma[i] * phi[i,a] ** sigma[i]

                            f2_right_term = np.empty((I, A))
                            for j in range(I):
                                for b in range(A):
                                    f2_right_term[j,b] = (tau[i,a,b] / rho_hat[i,b]) ** (1 - sigma[i]) * (mu[i] * (1 - lam[b]) + alpha[i,j,b] / alpha_labor[j,b]) * NN[j,b] * w[j,b]

                            f2_right = np.sum(f2_right_term)

                            F2[i,a] = f2_left - f2_right

                return np.concatenate([F1.ravel(), F2.ravel()])

            def jacobian(x, params):
                A, I = params['A'], params['I']

                X = np.append(x, 1.0)
                X = X.reshape((2, I, A))

                phi = X[0]
                w = X[1]


                mu, alpha = params['mu'], params['alpha']
                sigma, beta = params['sigma'], params['beta']
                tau, lam = params['tau'], params['lam']
                zeta, theta = params['zeta'], params['theta']
                N = params['N']
                NN = params['NN']

                
                aux_params = compute_aux_parameters(phi, w, params)

                psi = aux_params['psi']
                alpha_labor = aux_params['alpha_labor']
                w_hat = aux_params['w_hat']
                gamma = aux_params['gamma']
                n = aux_params['n']
                rho_term = aux_params['rho_term']
                rho_hat = aux_params['rho_hat']


                diff_rho_hat_phi = np.empty((I, A, A))    
                for i in range(I):
                    for a in range(A):
                        for b in range(A):
                            diff_rho_hat_phi[i,a,b] = sigma[i] * rho_hat[i,a] ** sigma[i] / ((sigma[i] - 1) * phi[i,b]) * rho_term[i,a,b]

                diff_rho_hat_w = np.empty((I, A, A))
                for i in range(I):
                    for a in range(A):
                        for b in range(A):
                            diff_rho_hat_w[i,a,b] = rho_hat[i,a] ** sigma[i] / ((1 - sigma[i]) * w[i,b]) * rho_term[i,a,b]


                J = np.zeros((2*I*A - A, 2*I*A - 1))

                def idx(var_block, i, a):
                    return var_block*I*A + i*A + a

                for i in range(I):
                    for a in range(A):

                        prod_term = np.empty(I)
                        for j in range(I):
                            if alpha[j,i,a] > 0:
                                prod_term[j] = (rho_hat[j,a] / alpha[j,i,a]) ** alpha[j,i,a]
                            else:
                                prod_term[j] = 1

                        f1_right = gamma[i,a] * (w[i,a] / alpha_labor[i,a]) ** alpha_labor[i,a] * np.prod(prod_term)

                        f2_right_term = np.empty((I, A))
                        for j in range(I):
                            for b in range(A):
                                f2_right_term[j,b] = (tau[i,a,b] / rho_hat[i,b]) ** (1 - sigma[i]) * (mu[i] * (1 - lam[b]) + alpha[i,j,b] / alpha_labor[j,b]) * NN[j,b] * w[j,b]


                        u1 = idx(0, i, a)
                        u2 = idx(1, i, a)


                        for j in range(I):
                            for b in range(A):
                                J[u1, idx(0,j,b)] = float(i==j and a==b) - alpha[j,i,a] / rho_hat[j,a] * diff_rho_hat_phi[j,a,b] * f1_right

                                if not (j==I-1 and b==A-1):
                                    J[u1, idx(1,j,b)] = - (float(i==j and a==b) * alpha_labor[i,a] / w[i,a] + alpha[j,i,a] / rho_hat[j,a] * diff_rho_hat_w[j,a,b]) * f1_right
                        
                        if not i==I-1:
                            for b in range(A):
                                Ju2_phi_term = np.empty((I,A))
                                for c in range(A):
                                    for k in range(I):
                                        Ju2_phi_term[k,c] = (sigma[i] - 1) / rho_hat[i,c] * diff_rho_hat_phi[i,c,b] * f2_right_term[k,c]

                                J[u2, idx(0,i,b)] = float(a==b) * sigma[i] ** 2 * phi[i,a] ** (sigma[i] - 1) - np.sum(Ju2_phi_term)

                                Ju2_w_term = np.empty((I,A))
                                for k in range(I):
                                    for c in range(A):
                                        Ju2_w_term[k,c] = ((sigma[i] - 1) / rho_hat[i,c] * diff_rho_hat_w[i,c,b] + float(i==k and b==c) / w[i,b]) * f2_right_term[k,c]
                                
                                if not (i==I-1 and b==A-1):
                                    J[u2, idx(1,i,b)] = - np.sum(Ju2_w_term)

                return J


            I = parameters['I']
            A = parameters['A']

            x0 = np.ones((2, I, A))
            phi0 = x0[0]
            w0 = x0[1]


            # シミュレーション初期値
            if initial_computed:
                mu = parameters['mu']
                alpha = parameters['alpha']
                sigma = parameters['sigma']
                beta = parameters['beta']
                tau = parameters['tau']
                lam = parameters['lam']
                zeta = parameters['zeta']
                theta = parameters['theta']
                N = parameters['N']
                NN = parameters['NN']

                alpha_labor = np.zeros((I, A))
                B = np.zeros(A)
                psi = np.zeros(I)

                for i in range(I):
                    psi[i] = sigma[i] * beta[i] / (sigma[i]-1)
                    for a in range(A):
                        alpha_labor[i,a] = max(1 - np.sum(alpha[:,i,a]), 1e-2)

                if 'gamma' in parameters:
                    gamma = parameters['gamma']
                else:
                    gamma = np.zeros((I,A))
                    for i in range(I):
                        for a in range(A):
                            gamma[i,a] = np.prod(psi[:] ** alpha[:,i,a])

                NN_bar = np.zeros((I,A))
                for i in range(I):
                    for a in range(A):
                        NN_bar[i,a] = zeta[i,a] ** theta[a] / np.sum(zeta[:,a] ** theta[a]) * N[a]

                rho_bar = np.zeros((I,A))
                for i in range(I):
                    for a in range(A):
                        rho_bar[i,a] = np.sum(NN_bar[i,:] * tau[i,:,a] ** (1 - sigma[i]) / (sigma[i] * alpha_labor[i,:])) ** (1 / (1 - sigma[i]))

                phi_rel = np.zeros((I,A))
                for i in range(I):
                    for a in range(A):
                        phi_rel_sum1 = np.zeros(A)
                        for b in range(A):
                            phi_rel_sum1[b] = np.sum((mu[i] * (1 - B[b] * lam[b]) + alpha[i,:,b] / alpha_labor[:,b]) * NN_bar[:,b])
                        
                        phi_rel_sum2 = np.zeros(A)
                        for b in range(A):
                            phi_rel_sum2[b] = np.sum(NN_bar[i,:] * tau[i,:,b] ** (1 - sigma[i]) / alpha_labor[i,:])

                        phi_rel[i,a] = np.sum(tau[i,a,:] * phi_rel_sum1[:] / phi_rel_sum2[:]) ** (1 / sigma[i])

                phi_w = np.zeros((I,A))
                for i in range(I):
                    for a in range(A):
                        phi_w[i,a] = (gamma[i,a] / phi_rel[i,a] * alpha_labor[i,a] ** (-alpha_labor[i,a]) * np.prod((1 / alpha[:,i,a] * rho_bar[:,a]) ** alpha[:,i,a])) ** (1 / (1 - np.sum(sigma[:] * alpha[:,i,a] / (sigma[:] - 1))))

                for i in range(I):
                    for a in range(A):
                        phi0[i,a] = phi_rel[i,a] * phi_w[i,a]

            if phi_initial is not None:
                phi0 = phi_initial
            if w_initial is not None:
                w0 = w_initial


            x0[0] = phi0 / w0[I-1, A-1]
            x0[1] = w0 / w0[I-1, A-1]
            x0 = np.ravel(x0)[:-1]

            try:
                sol = least_squares(function, x0, args=(parameters,), jac=jacobian, method='trf', bounds=(0, np.inf))
            
            except ValueError as e:
                print("ValueError 発生")
                return {
                    'phi': phi0,
                    'w': w0,
                    'norm': 1.0,
                    'valid': False
                }

            res_norm = np.linalg.norm(sol.fun, ord=np.inf)

            if not sol.success:
                print(f"      解法失敗: {sol.message}, 誤差 {res_norm:.1e}")

            x1 = np.append(sol.x, 1.0).reshape(2, I, A)
            phi, w = x1[0], x1[1]

            return {
                'phi': phi,
                'w': w,
                'norm': res_norm,
                'valid': True
            }

        def compute_NN(phi_0, w_0, parameters, verbose: bool = False, initial_computed: bool = False, delta = 0.1, TOL_F = 1e-8):
            I = parameters['I']
            A = parameters['A']

            N = parameters['N']
            zeta = parameters['zeta']
            theta = parameters['theta']
            lam = parameters['lam']

            NN0 = parameters['NN']
            phi0 = phi_0
            w0 = w_0
            parameters0 = parameters
            initial_comp = initial_computed

            print(" ")
            print("-------------------------- 変数計算中 --------------------------")

            for _ in range(10000):
                if verbose:
                    print(" ")
                    print(f"    {_+1} 回目の logit 計算中")
                computation = compute_phi_and_w(parameters=parameters0, phi_initial=phi0, w_initial=w0, initial_computed=initial_comp)

                valid = computation['valid']

                phi = computation['phi']
                w = computation['w']

                print(" ")
                print(f"    phi = {phi}")
                print(f"    w = {w}")
                print(" ")
                print(f"    NN0 = {NN0}")

                norm = computation['norm']

                if verbose:
                    print(f"      phiの誤差 {norm:.1e}")
                

                w_hat = np.empty((I,A))
                for i in range(I):
                    for a in range(A):
                        w_hat[i,a] = (1 - lam[a]) * w[i,a]

                NN = np.empty((I,A))
                for i in range(I):
                    for a in range(A):
                        NN[i,a] = NN0[i,a] + delta * ((w_hat[i,a] * zeta[i,a]) ** theta[a] / np.sum((w_hat[:,a] * zeta[:,a]) ** theta[a]) * N[a] - NN0[i,a])
                        print(f"    NN = {NN}")
                if verbose:
                    print(f"      NNの誤差 {np.linalg.norm(NN-NN0):.1e}")
                
                if np.linalg.norm(NN - NN0) < 1e-10:
                    return {
                        'NN': NN,
                        'phi': phi,
                        'w': w,
                        'norm': norm,
                        'number_of_logit': _+1,
                        'valid': valid
                    }
                
                else:
                    NN0 = NN
                    parameters0['NN'] = NN0
                    phi0 = phi
                    w0 = w
            
                initial_comp = False

            return {
                'NN': NN,
                'phi': phi,
                'w': w,
                'norm': norm,
                'number_of_logit': _+1,
                'valid': False
            }

        phi0 = phi_initial
        w0 = w_initial

        computation = compute_NN(phi_0=phi0, w_0=w0, parameters=parameters, verbose=verbose, initial_computed = initial_computed, delta=delta, TOL_F=TOL_F)

        I = parameters['I']
        A = parameters['A']

        valid = computation['valid']
        phi = computation['phi']
        w = computation['w']
        NN = computation['NN']
        norm = computation['norm']

        if norm > TOL_F:
            valid = False
            print(f"    logit 失敗：誤差 {norm:.1e} > 基準 {TOL_F}")

        number = computation['number_of_logit']

        W_all = parameters['W_all']
        W_all2 = np.sum(NN * w)
        phi = phi * W_all / W_all2
        w = w * W_all / W_all2


        mu = parameters['mu']
        alpha = parameters['alpha']
        sigma = parameters['sigma']
        beta = parameters['beta']
        tau = parameters['tau']
        lam = parameters['lam']
        zeta = parameters['zeta']
        theta = parameters['theta']
        N = parameters['N']


        aux_params = compute_aux_parameters(phi, w, parameters)

        psi = aux_params['psi']
        alpha_labor = aux_params['alpha_labor']
        w_hat = aux_params['w_hat']
        gamma = aux_params['gamma']
        n = aux_params['n']
        rho_hat = aux_params['rho_hat']


        W_hat = NN * w_hat

        S = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                S[i,a] = sigma[i] * n[i,a] * phi[i,a]
        
        M = np.empty((I,I,A))
        for i in range(I):
            for a in range(A):
                for j in range(I):
                    M[i,j,a] = alpha[i,j,a] * S[j,a]
        
        d = np.empty((I,I,A,A))
        for i in range(I):
            for a in range(A):
                for j in range(I):
                    for b in range(A):
                        d[j,i,b,a] = (tau[j,b,a] * phi[j,b] / rho_hat[j,a]) ** (1 - sigma[j]) * n[j,b] * mu[j] * W_hat[i,a]
        
        m = np.empty((I,I,A,A))
        for i in range(I):
            for a in range(A):
                for j in range(I):
                    for b in range(A):
                        m[j,i,b,a] = (tau[j,b,a] * phi[j,b] / rho_hat[j,a]) ** (1 - sigma[j]) * n[j,b] * alpha[j,i,a] * S[i,a]

        T = np.empty((I,A,A))
        for i in range(I):
            for a in range(A):
                for b in range(A):
                    T[i,a,b] = np.sum(m[i,:,a,b] + d[i,:,a,b])

        u = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                u[i,a] = w_hat[i,a] * np.prod((mu / (rho_hat[:,a] * psi)) ** mu)
        
        ESS = np.empty(A)
        for a in range(A):
            ESS[a] = N[a] / theta[a] * np.log(np.sum((zeta[:,a] * u[:,a]) ** theta[a]))


        if verbose:
            print(" ")
            print("---------------- Solutions ----------------")
            print("    phi =", phi)
            print("    w =", w)
            print("    NN =", NN)
            print(" ")
            print(f"    誤差：{norm:.1e}")
            print(f"    logit 計算回数：{number}")
            print(" ")
            print("    n =", n)
            print("    W_hat =", W_hat)
            print("    S =", S)
            print("    M =", M)
            print("    T =", T)



        return {
            'phi': phi,
            'w': w,
            'NN': NN,
            'n': n,
            'W_hat': W_hat,
            'w_hat': w_hat,
            'S': S,
            'M': M,
            'T': T,
            'sigma': sigma,
            'beta': beta,
            'ESS': ESS,
            'valid': valid
        }


    def compute2_aux_parameters(phi, w, params):
        A, I = params['A'], params['I']

        alpha = params['alpha']
        sigma = params['sigma']
        beta = params['beta']
        tau = params['tau']
        lam = params['lam']
        n = params['n']
        N = params['N']
        zeta = params['zeta']
        theta = params['theta']
        beta = params['beta']

        
        psi = np.empty(I)
        for i in range(I):
            psi[i] = sigma[i] * beta[i] / (sigma[i]-1)
        
        alpha_labor = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                alpha_labor[i,a] = 1 - np.sum(alpha[:,i,a])

        w_hat = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                w_hat[i,a] = (1 - lam[a]) * w[i,a]

        if 'gamma' in params:
            gamma = params['gamma']
        else:
            gamma = np.empty((I,A))
            for i in range(I):
                for a in range(A):
                    gamma[i,a] = np.prod(psi ** alpha[:,i,a])
        
        # NN = np.empty((I,A))
        # for i in range(I):
        #     for a in range(A):
        #         NN[i,a] = N[a] * (zeta[i,a] * w[i,a]) ** theta[a] / np.sum((zeta[:,a] * w[:,a]) ** theta[a])

        rho_term = np.empty((I,A,A))
        for i in range(I):
            for a in range(A):
                for b in range(A):
                    rho_term[i,a,b] = n[i,b] * (tau[i,b,a] * phi[i,b]) ** (1 - sigma[i])
            
        rho_hat = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                rho_hat[i,a] = np.sum(rho_term[i,a,:]) ** (1 / (1 - sigma[i]))
        
        s = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                s[i,a] = 1 / beta[i] * (NN[i,a] * w[i,a] / (alpha_labor[i,a] * n[i,a] * phi[i,a]) - 1)
        
        return {
            'psi': psi,
            'alpha_labor': alpha_labor,
            'w_hat': w_hat,
            'gamma': gamma,
            'rho_term': rho_term,
            'rho_hat': rho_hat,
            # 'NN': NN,
            's': s,
        }


    def compute2_phi_and_w(parameters, phi1, w1, n1, NN1, n_changed: Optional[int],
                        verbose: bool = False,
                        TOL_F = 1e-8,
                        i_changed: int = 0,
                        a_changed: int = 0):

        def function2(x, params):
            A, I = params['A'], params['I']

            X = np.append(x, 1.0)
            X = X.reshape(2, I, A)
            phi, w = X[0], X[1]


            mu, alpha = params['mu'], params['alpha']
            sigma, beta = params['sigma'], params['beta']
            tau, lam = params['tau'], params['lam']
            zeta, theta = params['zeta'], params['theta']
            N = params['N']
            NN = params['NN']
            n = params['n']

            aux_params = compute2_aux_parameters(phi, w, params)

            psi = aux_params['psi']
            alpha_labor = aux_params['alpha_labor']
            w_hat = aux_params['w_hat']
            gamma = aux_params['gamma']
            rho_term = aux_params['rho_term']
            rho_hat = aux_params['rho_hat']
            s = aux_params['s']


            F = np.empty(2*I*A - 1)
            

            for i in range(I):
                for a in range(A):

                    prod_term = np.empty(I)
                    for j in range(I):
                        if alpha[j,i,a] > 0:
                            prod_term[j] = (rho_hat[j,a] / alpha[j,i,a]) ** alpha[j,i,a]
                        else:
                            prod_term[j] = 1

                    f1_right = gamma[i,a] * (w[i,a] / alpha_labor[i,a]) ** alpha_labor[i,a] * np.prod(prod_term)
                    
                    F[i*A + a] = phi[i,a] - f1_right
            

                    if not (i==0 and a==0):
                        f2_left = psi[i] * s[i,a] * phi[i,a] ** sigma[i]

                        f2_right_term = np.empty((I, A))
                        for j in range(I):
                            for b in range(A):
                                f2_right_term[j,b] = (tau[i,a,b] / rho_hat[i,b]) ** (1 - sigma[i]) * (mu[i] * (1 - lam[b]) + alpha[i,j,b] / alpha_labor[j,b]) * NN[j,b] * w[j,b]

                        f2_right = np.sum(f2_right_term)

                        F[I*A + i*A + a - 1] = f2_left - f2_right
            
            # print("------------------- 解法の様子 ---------------------")
            # print("phi =", phi)
            # print(" ")
            # print("w =", w)
            # print(" ")
            # print("F =", F)

            return F

        def jacobian2(x, params):
            A, I = params['A'], params['I']

            X = np.append(x, 1.0)
            X = X.reshape(2, I, A)
            phi, w = X[0], X[1]


            
            mu, alpha = params['mu'], params['alpha']
            sigma, beta = params['sigma'], params['beta']
            tau, lam = params['tau'], params['lam']
            zeta, theta = params['zeta'], params['theta']
            N = params['N']
            NN = params['NN']
            n = params['n']

            aux_params = compute2_aux_parameters(phi, w, params)

            psi = aux_params['psi']
            alpha_labor = aux_params['alpha_labor']
            w_hat = aux_params['w_hat']
            gamma = aux_params['gamma']
            rho_term = aux_params['rho_term']
            rho_hat = aux_params['rho_hat']
            s = aux_params['s']


            diff_rho_hat_phi = np.zeros((I, A, A))
            diff_s_phi = np.zeros((I,A))
            diff_s_w = np.zeros((I,A))


            for i in range(I):
                for a in range(A):
                    for b in range(A):
                        diff_rho_hat_phi[i,a,b] = rho_hat[i,a] ** sigma[i] / phi[i,b] * rho_term[i,a,b]

                    diff_s_phi[i,a] = - NN[i,a] * w[i,a] / (beta[i] * alpha_labor[i,a] * n[i,a] * phi[i,a] ** 2)
            
            for i in range(I):
                for a in range(A):
                    diff_s_w[i,a] = NN[i,a] / (beta[i] * alpha_labor[i,a] * n[i,a] * phi[i,a])



            J = np.zeros((2*I*A - 1, 2*I*A - 1))


            for i in range(I):
                for a in range(A):
                    u1 = i*A + a
                    u2 = I*A + i*A + a - 1

                    prod_term = np.empty(I)
                    for j in range(I):
                        if alpha[j,i,a] > 0:
                            prod_term[j] = (rho_hat[j,a] / alpha[j,i,a]) ** alpha[j,i,a]
                        else:
                            prod_term[j] = 1

                    f1_right = gamma[i,a] * (w[i,a] / alpha_labor[i,a]) ** alpha_labor[i,a] * np.prod(prod_term)

                    for j in range(I):
                        for b in range(A):
                            J[u1, j*A + b] = float(a==b and i==j) - alpha[j,i,a] / rho_hat[j,a] * diff_rho_hat_phi[j,a,b] * f1_right

                    if not (i==I-1 and a==A-1):
                        J[u1, I*A + i*A + a] = - alpha_labor[i,a] / w[i,a] * f1_right
                    

                    if not (i==0 and a==0):
                        f2_right_term = np.zeros((I, A))
                        for j in range(I):
                            for b in range(A):
                                f2_right_term[j,b] = (tau[i,a,b] / rho_hat[i,b]) ** (1 - sigma[i]) * (mu[i] * (1 - lam[b]) + alpha[i,j,b] / alpha_labor[j,b]) * NN[j,b] * w[j,b]
                        
                        for b in range(A):
                            Ju2_phi_term = np.zeros((I,A))
                            for k in range(I):
                                for c in range(A):
                                    Ju2_phi_term[k,c] = (1 - sigma[i]) / rho_hat[i,c] * diff_rho_hat_phi[i,c,b] * f2_right_term[k,c]

                            J[u2, i*A + b] = float(a==b) * psi[i] * (diff_s_phi[i,a] * phi[i,a] ** sigma[i] + sigma[i] * s[i,a] * phi[i,a] ** (sigma[i] - 1)) + np.sum(Ju2_phi_term)

                        
                            for j in range(I):
                                if not (j==I-1 and b==A-1):
                                    J[u2, I*A + j*A + b] = float(a==b and i==j) * psi[i] * diff_s_w[i,a] * phi[i,a] ** sigma[i] - f2_right_term[j,b]

            return J

        n = n1
        n_original = n[i_changed,a_changed]
        if n_changed is None:
            n[i_changed,a_changed] = n[i_changed,a_changed] + 1
        else:
            n[i_changed,a_changed] = n[i_changed,a_changed] * (n_changed + 1) / n_changed

        NN = NN1


        # if verbose:
        print(" ")
        print(f"    企業数の変化：n[{i_changed},{a_changed}]: {n_original}  >  {n[i_changed,a_changed]}")

        mu = parameters['mu']
        alpha = parameters['alpha']
        sigma = parameters['sigma']
        beta = parameters['beta']
        tau = parameters['tau']
        lam = parameters['lam']
        zeta = parameters['zeta']
        theta = parameters['theta']
        N = parameters['N']
        W_all = parameters['W_all']
        I, A = zeta.shape


        x1 = [phi1, w1]
        x1 = x1 / w1[I-1, A-1]
        x1 = np.ravel(x1)[:-1]

        params = parameters.copy()
        params['n'] = n
        params['NN'] = NN

        sol2 = least_squares(function2, x1, args=(params,), jac=jacobian2, method='trf', bounds=(0, np.inf))
        
        res_norm2 = np.linalg.norm(sol2.fun, ord=np.inf)

        if not sol2.success:
            print(f"    解法失敗: {sol2.message}, 誤差 {res_norm2:.1e}")


        valid = True
        if res_norm2 > TOL_F:
            valid = False

        x2 = np.append(sol2.x, 1.0).reshape(2, I, A)
        phi, w = x2[0], x2[1]

        w_hat = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                w_hat[i,a] = (1 - lam[a]) * w[i,a]
        
        W_all2 = np.sum(NN * w)
        phi = phi * W_all / W_all2
        w = w * W_all / W_all2

        alpha_labor = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                alpha_labor[i,a] = max(1 - np.sum(alpha[:,i,a]), 1e-2)

        s = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                s[i,a] = 1 / beta[i] * (NN[i,a] * w[i,a] / (alpha_labor[i,a] * n[i,a] * phi[i,a]) - 1)
        
        psi = beta * sigma / (sigma - 1)
        S = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                S[i,a] = psi[i] * n[i,a] * s[i,a] * phi[i,a]
        
        Pi = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                Pi[i,a] = (beta[i] * s[i,a] / (sigma[i] - 1) - 1) * phi[i,a] * n[i,a]
        
        rho_hat = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                rho_hat[i,a] = np.sum(n[i] * (tau[i,:,a] * phi[i]) ** (1 - sigma[i])) ** (1 / (1 - sigma[i]))
        
        u = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                u[i,a] = w_hat[i,a] * np.prod((mu / (rho_hat[:,a] * psi)) ** mu)
        
        ESS = np.empty(A)
        for a in range(A):
            ESS[a] = N[a] / theta[a] * np.log(np.sum((zeta[:,a] * u[:,a]) ** theta[a]))

        if verbose:
            print(" ")
            print("------------ New solutions ------------")
            print("phi2 =", phi)
            print("w2 =", w)
            print(" ")
            print(f"方程式の値の誤差：|F|_inf = {res_norm2:.1e}")
            print(" ")
            print("------------ Changes ------------")
            print("phi2 / phi1 =", phi/phi1)
            print("w2 / w1 =", w/w1)
            print("------------ supply and profit ------------")
            print("S =", S)
            print("Pi =", Pi)

        return {
            'phi': phi,
            'w': w,
            'Pi': Pi,
            'ESS': ESS,
            'valid': valid,
        }


    def calibrate_parameters(data,
                    beta_initial: Optional[np.ndarray] = None,
                    verbose: bool = False, TOL_F = 1e-8) -> Dict[str, np.ndarray]:

        def compute_N(data) -> np.ndarray:
            A = data['A']
            NN = data['NN']

            N = np.empty(A)
            for a in range(A):
                N[a] = np.sum(NN[:,a])

            return N

        
        def compute_alpha(data) -> np.ndarray:
            I = data['I']
            A = data['A']
            IO_table = data['IO_table']

            alpha = np.empty((I, I, A), dtype=float)
            for a in range(A):
                for i in range(I):
                    for j in range(I):
                        alpha[i, j, a] = IO_table[i, j] / np.sum(IO_table[:, j])

            return alpha

        
        def compute_W(data) -> np.ndarray:
            I = data['I']
            A = data['A']
            IO_table = data['IO_table']
            S = data['S']

            W = np.empty((I, A), dtype=float)
            for a in range(A):
                for i in range(I):
                    W[i,a] = IO_table[I, i] * S[i,a] / np.sum(IO_table[:,i])

            return W


        def compute_lambda(data) -> np.ndarray:
            I = data['I']
            A = data['A']
            IO_table = data['IO_table']

            W =np.sum(IO_table[I])
            W_hat = np.sum(IO_table[:, I])

            lam = np.empty(A, dtype=float)
            for a in range(A):
                lam[a] = 1 - W_hat / W

            return lam


        def compute_W_hat(data):
            I = data['I']
            A = data['A']

            lam = compute_lambda(data)
            W = compute_W(data)

            W_hat = np.empty((I,A))
            for i in range(I):
                for a in range(A):
                    W_hat[i,a] = (1 - lam[a]) * W[i,a]

            return W_hat


        def compute_tau_hat(data) -> np.ndarray:
            I = data['I']
            A = data['A']

            tau_hat = np.empty((I, A, A), dtype=float)

            data_for_tau = data['data_for_tau']

            if data_for_tau is not None:
                Tau = data_for_tau['Tau']
                eps = data_for_tau['epsilon']
                t = data_for_tau['t']

                for i in range(I):
                    for a in range(A):
                        for b in range(A):
                            if a==b:
                                tau_hat[i, a, a] = 1.0
                            else:
                                tau_hat[i, a, b] = np.exp((Tau[i] * t[a,b] + eps[i]))
            
            else:
                T = data['T']

                tau_hat = np.empty((I,A,A))
                for i in range(I):
                    for a in range(A):
                        for b in range(A):
                            tau_hat[i, a, b] = (T[i,a,b] * T[i,b,a] / (T[i,a,a] * T[i,b,b])) ** (1/2)

            return tau_hat


        def compute_tau(data) -> np.ndarray:
            I = data['I']
            A = data['A']
            tau_hat = compute_tau_hat(data)

            tau = np.empty((I,A,A))

            for i in range(I):
                for a in range(A):
                    for b in range(A):
                        tau[i, a, b] = tau_hat[i, a, b] ** (1 / (1 - sigma[i]))

            return tau


        def compute_mu(data) -> np.ndarray:
            I = data['I']
            A = data['A']
            IO_table = data['IO_table']

            W_hat = np.sum(IO_table[:, I])

            mu = np.empty(I)
            for i in range(I):
                mu[i] = IO_table[i, I] / W_hat

            return mu


        def compute_phi(data) -> np.ndarray:
            I = data['I']
            A = data['A']
            S = data['S']
            sigma = data['sigma']
            n = data['n']

            phi = np.empty((I,A), dtype=float)

            for i in range(I):
                for a in range(A):
                    phi[i,a] = S[i,a] / (sigma[i] * n[i,a])

            return phi


        def compute_rho_hat(data) -> np.ndarray:
            I = data['I']
            A = data['A']
            n = data['n']
            sigma = data['sigma']

            rho_hat = np.empty((I,A), dtype=float)

            tau_hat = compute_tau_hat(data)
            phi = compute_phi(data)

            for i in range(I):
                for a in range(A):
                    rho_hat[i,a] = np.sum(tau_hat[i,:,a] * n[i] * phi[i] ** (1 - sigma[i])) ** (1 / (1 - sigma[i]))

            return rho_hat


        def compute_gamma(data) -> np.ndarray:
            I = data['I']
            A = data['A']
            S = data['S']
            NN = data['NN']

            alpha = compute_alpha(data)
            phi = compute_phi(data)
            rho_hat = compute_rho_hat(data)

            gamma = np.empty((I, A), dtype=float)
            for i in range(I):
                for a in range(A):

                    prod_term = np.empty(I)
                    for j in range(I):
                        if alpha[j,i,a] == 0:
                            prod_term[j] = 1.0
                        else:
                            prod_term[j] = (rho_hat[j,a] / alpha[j,i,a]) ** alpha[j,i,a]
                    
                    gamma[i, a] = phi[i,a] * (S[i,a] / NN[i,a]) ** (-1 + np.sum(alpha[:,i,a])) * np.prod(prod_term)

            return gamma


        def solve_beta(initial_beta: Optional[np.ndarray], context: Dict[str, Any]):

            def beta_residual(beta: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
                I = context['I']
                A = context['A']
                sigma = context['sigma']
                alpha = context['alpha']
                gamma = context['gamma']

                psi = sigma * beta / (sigma - 1)

                F = np.empty((I,A), dtype=float)
                for i in range(I):
                    for a in range(A):
                        F[i,a] = gamma[i,a] - np.prod(psi ** alpha[:,i,a])

                return F.ravel()

            def beta_jacobian(beta: np.ndarray, context: Dict[str, Any]):
                I = context['I']
                A = context['A']
                sigma = context['sigma']
                alpha = context['alpha']

                psi = sigma * beta / (sigma - 1)

                F_right = np.empty((I,A), dtype=float)
                for i in range(I):
                    for a in range(A):
                        F_right[i,a] = np.prod(psi ** alpha[:,i,a])
                
                J = np.empty((I*A, I))
                for i in range(I):
                    for a in range(A):
                        for j in range(I):
                            J[A*i + a, j] = - alpha[j,i,a] / beta[j] * F_right[i,a]

                return J

            I = context['I']
            if initial_beta is None:
                initial_beta = np.ones(I, dtype=float)

            def wrapped_residual(x):
                return beta_residual(x, context)

            def wrapped_jac(x):
                return beta_jacobian(x, context)

            sol = least_squares(wrapped_residual, initial_beta, jac=wrapped_jac, method='trf', bounds=(0, np.inf))
            
            res_norm = np.linalg.norm(sol.fun, ord=np.inf)

            if not sol.success:
                print(f"Beta の解放失敗 {sol.message} 誤差 {res_norm:.1e}")

            return sol.x, res_norm


        def solve_theta(context: Dict[str, Any], TOL_F = 1e-8):
            
            def theta_residual(theta, context: Dict[str, Any]) -> np.ndarray:
                NN = context['NN']  # shape I
                w = context['w']  # shape I

                P = w ** theta / np.sum(w ** theta)
                u = np.log(w)

                return np.sum(NN * (u - np.sum(P * u)))

            def theta_jacobian(theta, context: Dict[str, Any]):
                I = context['I']
                NN = context['NN']  # shape I
                w = context['w']  # shape I

                P = w ** theta / np.sum(w ** theta)
                u = np.log(w)

                sum1 = np.empty(I)
                for i in range(I):
                    sum1[i] = np.sum(P * u)

                return np.array([np.sum(NN * (u - np.sum(u * P * (u - np.sum(P * u)))))])

            I = context['I']
            NN = context['NN']
            W = context['W']
            w = W / NN
            w = I * w / np.sum(w)

            initial_theta = 1.0

            theta = np.empty(A)
            res_norms = np.empty(A)
            for a in range(A):
                context_a = {
                    'I': I,
                    'NN': NN[:,a],
                    'w': w[:,a],
                }

                def wrapped_residual(x):
                    return theta_residual(x, context_a)

                def wrapped_jac(x):
                    return theta_jacobian(x, context_a)
                
                sol_a = root(wrapped_residual, initial_theta, jac=wrapped_jac, method='hybr')
                res_norms[a] = float(np.abs(sol_a.fun).item())

                if not sol_a.success:
                    if res_norms[a] <= TOL_F:
                        print(f"警告: theta[{a}] の解法は失敗判定だが残差が十分小さいため受理します: 誤差 {res_norms[a]:.1e} < 閾値 {TOL_F:.1e}).")
                    else:
                        raise RuntimeError(f"theta[{a}] 解放失敗 {sol_a.message}, 誤差 {res_norms[a]:.1e} > 閾値 {TOL_F:.1e}")

                theta[a] = float(sol_a.x.item())

            res_norm = np.max(res_norms)

            return theta, res_norm


        def compute_zeta(context: Dict[str, Any]):
            I = context['I']
            NN = context['NN']
            W = context['W']
            theta = context['theta']
            w = W / NN

            zeta = np.ones((I, A))
            for i in range(I):
                for a in range(A):
                    zeta[i, a] = w[I-1, a] / w[i, a] * (NN[i, a] / NN[I-1, a]) ** (1 / theta[a])

            return zeta



        I = data['I']
        A = data['A']
        NN = data['NN']
        n = data['n']
        sigma = data['sigma']

        N = compute_N(data)
        alpha = compute_alpha(data)
        W = compute_W(data)
        lam = compute_lambda(data)
        W_hat = compute_W_hat(data)
        tau = compute_tau(data)
        mu = compute_mu(data)
        phi = compute_phi(data)
        rho_hat = compute_rho_hat(data)
        gamma = compute_gamma(data)
        w = W / NN
        W_all = np.sum(W)
        w_hat = W_hat / NN

        context = {
            'I': I,
            'A': A,
            'sigma': sigma,
            'alpha': alpha,
            'W': W,
            'gamma': gamma,
            'NN': NN,
        }

        print(" ")
        print("------------------ Computed parameters ------------------")

        alpha_labor = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                alpha_labor[i,a] = 1 - np.sum(alpha[:,i,a])

        if verbose:
            print("alpha =", alpha)
            print("alpha_labor =", alpha_labor)
            print("lambda =", lam)
            print("tau =", tau)
            print("mu =", mu)
            print("gamma =", gamma)
            print("N =", N)
            print("w =", w)
            print("w_hat =", w_hat)

        beta, beta_res_norm = solve_beta(beta_initial, context)
        if verbose:
            print(" ")
            print("beta =", beta)
            print(f"beta の誤差：{beta_res_norm:.1e}")

        theta, theta_res_norm = solve_theta(context, TOL_F)
        if verbose:
            print(" ")
            print("theta =", theta)
            print(f"theta の誤差：{theta_res_norm:.1e}")

        context['theta'] = theta

        zeta = compute_zeta(context)

        if verbose:
            print(" ")
            print("zeta =", zeta)

            print(" ")
            print("phi =", phi)
            print("w =", w)


        return {
            'I': I,
            'A': A,
            'N': N,
            'NN': NN,
            'alpha': alpha,
            'W': W,
            'lam': lam,
            'tau': tau,
            'mu': mu,
            'phi': phi,
            'rho_hat': rho_hat,
            'gamma': gamma,
            'beta': beta,
            'theta': theta,
            'zeta': zeta,
            'sigma': sigma,
            'phi': phi,
            'w': w,
            'W_all': W_all,
        }


    NN = data['NN']
    n = data['n']
    S = data['S']
    IO_table = data['IO_table']

    if verbose:
        print(" ")
        print("-------------------------- Given datum --------------------------")
        print(f"NN = {NN}")
        print(f"n = {n}")
        print(f"S = {S}")
        print(f"IO_table = {IO_table}")

    calibration = calibrate_parameters(data, verbose = verbose, TOL_F = TOL_F)
    phi0, w0 = calibration['phi'], calibration['w']

    values = compute_equilibrium(calibration, phi_initial=phi0, w_initial=w0, verbose = True, TOL_F = TOL_F)
    valid = values['valid']
    phi, w, n = values['phi'], values['w'], values['n']
    NN = values['NN']
    S = values['S']
    ESS1 = values['ESS']

    i_changed = data['i_changed']
    a_changed = data['a_changed']
    n_changed = data['n_changed']

    values2 = compute2_phi_and_w(calibration, phi1=phi, w1=w, n1=n, NN1=NN, verbose = verbose, TOL_F = TOL_F, i_changed = i_changed, a_changed = a_changed, n_changed=n_changed)
    valid2 = values2['valid']
    Pi = values2['Pi']
    ESS2 = values2['ESS']

    REV = (ESS2- ESS1) / ESS1

    print(" ")
    print(" 最終計算結果")
    print(f"Pi/S = {Pi/S}")
    print(f"REV = {REV}")

    return {
        # 'Pi': Pi,
        # 'REV': REV,
        'Pi/S': Pi/S,
        'valid': valid and valid2
    }







I = 7
A = 2
i_changed = 4
a_changed = 1

sigma0 = 5
sigma = np.ones(I) * sigma0

random = True
initial_computed = False
seed = 1
verbose = True
TOL_F = 1e-5
delta = 1.0
data_for_tau = None





NN = np.array([
    [610,	643],
    [30965,	20492],
    [946,	297],
    [7422,	246],
    [3468,	642],
    [186663,	48259],
    [8816,	1864]
])

n = np.array([
    [70,	57],
    [2805,	1584],
    [37,	25],
    [352,	25],
    [183,	65],
    [17543,	6078],
    [133,	81],
])

n_changed = n[i_changed, a_changed]

IO_table = np.array([
    [20000, 83372, 	0, 	0, 	99, 	14797, 	0, 	41658],
    [59469, 	2108068, 	103599, 	14970, 	5936, 	421588, 	601, 	610286 ],
    [2685, 	125165, 	13264, 	3720, 	5303, 	87283, 	32, 	83130 ],
    [1227, 	23172, 	3575, 	81066, 	2701, 	107994, 	920, 	159904 ],
    [0, 	0, 	0, 	2599, 	462, 	294, 	6, 	81182 ],
    [40965, 	549349, 	48762, 	89534, 	18047, 	995463, 	14566, 	2033503 ],
    [85307, 	819616, 	78263, 	109140, 	50174, 	1529950, 	32948, 	129 ],
    [26284, 	623715, 	21944, 	50469, 	20374, 	1591522, 	611, 0]	
])

S_relative = np.array([
    [0.034524, 	0.045555], 
    [0.110750, 	0.188225 ],
    [0.503762, 	0.113730 ],
    [0.503762, 	0.113730 ],
    [0.448661, 	0.049421 ],
    [0.503762, 	0.113730 ],
    [0.503762, 	0.113730 ]
])

S = np.empty((I,A))
for i in range(I):
    for a in range(A):
        S[i,a] = np.sum(IO_table[:, i]) * S_relative[i,a]

Tau = np.array([-4.36, -7.2, -7.86, -7.86, -7.86, -7.86, -7.86])
epsilon = np.array([-2.88, -2.38, -3.23, -3.23, -3.23, -3.23, -3.23])
t = np.array([
    [0.0, 73/6000],
    [73/6000, 0.0]
])

data_for_tau = {
    'Tau': Tau,
    'epsilon': epsilon,
    't': t,
}

data = {
    'I': I,
    'A': A,
    'NN': NN,
    'n': n,
    'S': S,
    'IO_table': IO_table,
    'sigma': sigma,
    'data_for_tau': data_for_tau,
    'i_changed': i_changed,
    'a_changed': a_changed,
    'n_changed': n_changed,
}


changes = compute_changes(data=data, verbose=verbose, TOL_F=TOL_F)
valid = changes['valid']
print(" ")
print("----------------------- 計算結果 -----------------------")
if valid:
    Pi = changes['Pi']
    print("  Pi =", Pi)
else:
    print("  計算失敗")






# ランダムにパラメータ生成して数値計算する
# population = 100
# parameters = generate_parameters(I, A, sigma0 = sigma0, population = population, random = random, seed = seed, verbose=verbose)
# values = compute_equilibrium(parameters, verbose = verbose, initial_computed=initial_computed, TOL_F = TOL_F, delta=delta)
# phi, w, n = values['phi'], values['w'], values['n']
# values2 = compute2_phi_and_w(parameters, phi1=phi, w1=w, n1=n, verbose = verbose, TOL_F = TOL_F)

