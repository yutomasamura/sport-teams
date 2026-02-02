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
fp = fm.FontProperties(fname=font_path, size=26)
fplegend = fm.FontProperties(fname=font_path, size=24)








def compute_changes(data, verbose: bool = False, TOL_F = 1e-10):
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
                alpha_labor[i,a] = max(1 - np.sum(alpha[:,i,a]), 1e-5)

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
                verbose: bool = False,
                initial_computed: bool = False,
                TOL_F = 1e-8
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


                F = np.empty((2, I, A))
                

                for i in range(I):
                    for a in range(A):
                        f1_right = gamma[i,a] * (w[i,a] / alpha_labor[i,a]) ** alpha_labor[i,a] * np.prod((rho_hat[:,a] / alpha[:,i,a]) ** alpha[:,i,a])

                        f1 = phi[i,a] - f1_right

                        f2_left = sigma[i] * phi[i,a] ** sigma[i]

                        f2_right_term = np.empty((I, A))
                        for j in range(I):
                            for b in range(A):
                                f2_right_term[j,b] = (tau[i,a,b] / rho_hat[i,b]) ** (1 - sigma[i]) * (mu[i] * (1 - lam[b]) + alpha[i,j,b] / alpha_labor[j,b]) * NN[j,b] * w[j,b]

                        f2_right = np.sum(f2_right_term)

                        f2 = f2_left - f2_right
                        
                        F[0,i,a] = f1
                        F[1,i,a] = f2

                return F.ravel()

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


                J = np.zeros((2*I*A, 2*I*A - 1))

                def idx(var_block, i, a):
                    return var_block*I*A + i*A + a


                for i in range(I):
                    for a in range(A):
                        f1_right = gamma[i,a] * (w[i,a] / alpha_labor[i,a]) ** alpha_labor[i,a] * np.prod((rho_hat[:,a] / alpha[:,i,a]) ** alpha[:,i,a])

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
                print("      ValueError 発生")
                return {
                    'phi': phi0,
                    'w': w0,
                    'norm': 1.0,
                    'valid': False
                }

            res_norm = np.linalg.norm(sol.fun, ord=np.inf)

            x1 = np.append(sol.x, 1.0).reshape(2, I, A)
            phi, w = x1[0], x1[1]

            return {
                'phi': phi,
                'w': w,
                'norm': res_norm,
                'valid': True
            }

        def compute_NN(phi_0, w_0, parameters, verbose: bool = False, initial_computed: bool = False, delta = 0.1, TOL_F = 1e-8):
            N = parameters['N']
            zeta = parameters['zeta']
            theta = parameters['theta']
            lam = parameters['lam']

            NN0 = parameters['NN']
            phi0 = phi_0
            w0 = w_0
            parameters0 = parameters
            initial_comp = initial_computed

            for _ in range(10000):
                if verbose:
                    print(f"    {_+1} 回目の logit 計算中")
                computation = compute_phi_and_w(parameters=parameters0, phi_initial=phi0, w_initial=w0, verbose=verbose, initial_computed=initial_comp, TOL_F = TOL_F)

                valid = computation['valid']

                phi = computation['phi']
                w = computation['w']
                norm = computation['norm']
                

                w_hat = np.empty((I,A))
                for i in range(I):
                    for a in range(A):
                        w_hat[i,a] = (1 - lam[a]) * w[i,a]

                NN = np.empty((I,A))
                for i in range(I):
                    for a in range(A):
                        NN = NN0 + delta * ((w_hat[i,a] * zeta[i,a]) ** theta[a] / np.sum(w_hat[:,a] * zeta[:,a]) ** theta[a] * N[a] - NN0[i,a])
                if verbose:
                    print(f"      誤差 {np.linalg.norm(NN-NN0):.1e}")
                
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
            print("------------ Solutions ------------")
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
                alpha_labor[i,a] = max(1 - np.sum(alpha[:,i,a]), 1e-2)

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
        
        NN = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                NN[i,a] = N[a] * (zeta[i,a] * w[i,a]) ** theta[a] / np.sum((zeta[:,a] * w[:,a]) ** theta[a])

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
            'NN': NN,
            's': s,
        }






    def compute2_phi_and_w(parameters, phi1, w1, n1,
                        verbose: bool = False,
                            TOL_F = 1e-8):

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
            n = params['n']


            aux_params = compute2_aux_parameters(phi, w, params)

            psi = aux_params['psi']
            alpha_labor = aux_params['alpha_labor']
            w_hat = aux_params['w_hat']
            gamma = aux_params['gamma']
            rho_term = aux_params['rho_term']
            rho_hat = aux_params['rho_hat']
            NN = aux_params['NN']
            s = aux_params['s']
            
            s = np.zeros((I,A))
            for i in range(I):
                for a in range(A):
                    s[i,a] = 1 / beta[i] * (NN[i,a] * w[i,a] / (alpha_labor[i,a] * n[i,a] * phi[i,a]) - 1)





            F = np.empty(2*I*A - 1)
            

            for i in range(I):
                for a in range(A):
                    f1_right = gamma[i,a] * (w[i,a] / alpha_labor[i,a]) ** alpha_labor[i,a] * np.prod((rho_hat[:,a] / alpha[:,i,a]) ** alpha[:,i,a])
                    
                    F[i*A + a] = phi[i,a] - f1_right
            

                    if not (i==0 and a==0):
                        f2_left = psi[i] * s[i,a] * phi[i,a] ** sigma[i]

                        f2_right_term = np.zeros((I, A))
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
            n = params['n']



            NN = np.zeros((I,A))
            w_hat = np.zeros((I,A))
            alpha_labor = np.zeros((I,A))
            psi = np.zeros(I)
            

            for i in range(I):
                psi[i] = sigma[i] * beta[i] / (sigma[i]-1)
                for a in range(A):
                    alpha_labor[i,a] = max(1 - np.sum(alpha[:,i,a]), 1e-2)

                    w_hat[i,a] = (1 - lam[a]) * w[i,a]
            
            if 'gamma' in params:
                gamma = params['gamma']
            else:
                gamma = np.zeros((I,A))
                for i in range(I):
                    for a in range(A):
                        gamma[i,a] = np.prod(psi[:] ** alpha[:,i,a])

            
            for i in range(I):
                for a in range(A):
                    NN[i,a] = N[a] * ((w_hat[i,a] * zeta[i,a]) ** theta[a]) / np.sum((w_hat[:,a] * zeta[:,a]) ** theta[a])
            
            rho_term = np.zeros((I,A,A))
            rho_hat = np.zeros((I,A))
            for i in range(I):
                for a in range(A):
                    for b in range(A):
                        rho_term[i,a,b] = n[i,b] * (tau[i,b,a] * phi[i,b]) ** (1 - sigma[i])

                    rho_hat[i,a] = np.sum(rho_term[i,a,:]) ** (1 / (1 - sigma[i]))
            
            s = np.zeros((I,A))
            for i in range(I):
                for a in range(A):
                    s[i,a] = 1 / beta[i] * (NN[i,a] * w[i,a] / (alpha_labor[i,a] * n[i,a] * phi[i,a]) - 1)



            diff_NN_w = np.zeros((I, A, I))
            diff_rho_hat_phi = np.zeros((I, A, A))
            diff_s_phi = np.zeros((I,A))
            diff_s_w = np.zeros((I,A,I))


            for i in range(I):
                for a in range(A):
                    for j in range(I):
                        diff_NN_w[i,a,j] = theta[a] * (float(i==j) - NN[j,a] / N[a]) * NN[i,a] / w[j,a]

                    for b in range(A):
                        diff_rho_hat_phi[i,a,b] = rho_hat[i,a] ** sigma[i] / phi[i,b] * rho_term[i,a,b]

                    diff_s_phi[i,a] = - NN[i,a] * w[i,a] / (beta[i] * alpha_labor[i,a] * n[i,a] * phi[i,a] ** 2)
            
            for i in range(I):
                for a in range(A):
                    for j in range(I):
                        diff_s_w[i,a,j] = 1 / (beta[i] * alpha_labor[i,a] * n[i,a] * phi[i,a]) * (diff_NN_w[i,a,j] * w[i,a] + float(i==j) * NN[i,a])



            J = np.zeros((2*I*A - 1, 2*I*A - 1))


            for i in range(I):
                for a in range(A):
                    u1 = i*A + a
                    u2 = I*A + i*A + a - 1



                    f1_right = gamma[i,a] * (w[i,a] / alpha_labor[i,a]) ** alpha_labor[i,a] * np.prod((rho_hat[:,a] / alpha[:,i,a]) ** alpha[:,i,a])

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

                            J[u2, i*A + b] = float(a==b) * psi[i] * (diff_s_phi[i,a] * phi[i,a] ** sigma[i] + sigma[i] * s[i,a] * phi[i,a] ** (sigma[i] - 1)) - np.sum(Ju2_phi_term)

                        
                            for j in range(I):
                                if not (j==I-1 and b==A-1):
                                    Ju2_w_term = np.zeros(I)
                                    for k in range(I):
                                        Ju2_w_term[k] = f2_right_term[k,b] * (diff_NN_w[k,b,j] / NN[k,b] + float(j==k) / w[j,b])
                                    
                                    J[u2, I*A + j*A + b] = float(a==b) * psi[i] * diff_s_w[i,a,j] * phi[i,a] ** sigma[i] - np.sum(Ju2_w_term)

            return J


        n = n1
        n[0,0] = n[0,0] + 1


        # if verbose:
        print(f"    企業数の変化：n[0,0]: {n[0,0]-1}  >  {n[0,0]}")

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

        sol2 = root(function2, x1, args=(params,), jac=jacobian2, method='hybr')


        res_norm2 = np.linalg.norm(sol2.fun, ord=np.inf)


        valid = True
        if res_norm2 > TOL_F:
            valid = False

        x2 = np.append(sol2.x, 1.0).reshape(2, I, A)
        phi, w = x2[0], x2[1]

        w_hat = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                w_hat[i,a] = (1 - lam[a]) * w[i,a]

        NN = np.empty((I,A))
        for i in range(I):
            for a in range(A):
                NN[i,a] = N[a] * ((w_hat[i,a] * zeta[i,a]) ** theta[a]) / np.sum((w_hat[:,a] * zeta[:,a]) ** theta[a])
        
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





    @dataclass
    class BetaSolverOptions:
        method: str = "trf"  # method for scipy.optimize.least_squares: 'trf','dogbox','lm'
        jac: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None
        bounds = (0, np.inf)
        extra_options: Dict[str, Any] = None



    def calibrate_parameters(NN: np.ndarray, n: np.ndarray, W_hat: np.ndarray, S: np.ndarray,
                    M: np.ndarray, T: np.ndarray, sigma: np.ndarray,
                    beta_initial: Optional[np.ndarray] = None,
                    beta_options: Optional[BetaSolverOptions] = None,
                    data_for_tau: Optional[Dict[str, np.ndarray]] = None,
                    verbose: bool = False, TOL_F = 1e-8) -> Dict[str, np.ndarray]:
        """Top-level function: validate inputs, compute all intermediate arrays,
        solve for beta and theta, and return results in a dictionary.

        Returns a dictionary with keys:
            'alpha', 'lam', 'tau', 'sigma', 'mu', 'gamma', 'beta', 'theta'
        """


        def _validate_shapes(NN: np.ndarray, n: np.ndarray, W_hat: np.ndarray, S: np.ndarray,
                            M: np.ndarray, T: np.ndarray, sigma: np.ndarray) -> (int, int):
            """Validate input shapes and return (I, A). Raises ValueError on mismatch."""
            # if not (isinstance(NN, np.ndarray) and isinstance(n, np.ndarray) and
            #         isinstance(W_hat, np.ndarray) and isinstance(S, np.ndarray) and
            #         isinstance(M, np.ndarray) and isinstance(T, np.ndarray) and
            #         isinstance(sigma, np.ndarray)):
            #     raise ValueError("All inputs must be numpy arrays.")

            if NN.shape != n.shape or NN.shape != W_hat.shape or NN.shape != S.shape:
                raise ValueError("NN, n, W, S must have identical shapes (I, A).")
            I, A = NN.shape

            if M.shape != (I, I, A):
                raise ValueError(f"M must have shape (I, I, A) = ({I},{I},{A}), got {M.shape}.")

            if T.shape != (I, A, A):
                raise ValueError(f"T must have shape (I, A, A) = ({I},{A},{A}), got {T.shape}.")

            return I, A


        def compute_N(NN: np.ndarray, n: np.ndarray, W_hat: np.ndarray, S: np.ndarray,
                        M: np.ndarray, T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
            N = np.empty(A)
            for a in range(A):
                N[a] = np.sum(NN[:,a])

            return N


        def compute_alpha(NN: np.ndarray, n: np.ndarray, W_hat: np.ndarray, S: np.ndarray,
                        M: np.ndarray, T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
            I, A = NN.shape
            alpha = np.empty((I, I, A), dtype=float)

            for a in range(A):
                for i in range(I):
                    for j in range(I):
                        alpha[i, j, a] = M[i, j, a] / S[j,a]

            return alpha


        def compute_W(NN: np.ndarray, n: np.ndarray, W_hat: np.ndarray, S: np.ndarray,
                        M: np.ndarray, T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
            I, A = NN.shape
            W = np.empty((I, A), dtype=float)

            for a in range(A):
                for i in range(I):
                    W[i,a] = S[i,a] - np.sum(M[:,i,a])
                    W[i,a] = max(1, W[i,a])

            return W


        def compute_lambda(NN: np.ndarray, n: np.ndarray, W_hat: np.ndarray, S: np.ndarray,
                        M: np.ndarray, T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
            I, A = NN.shape
            W = compute_W(NN, n, W_hat, S, M, T, sigma)
            lam = np.empty(A, dtype=float)

            for a in range(A):
                lam[a] = 1 - np.sum(W_hat[:,a]) / np.sum(W[:,a])

            return lam


        def compute_tau_hat(NN: np.ndarray, n: np.ndarray, W_hat: np.ndarray, S: np.ndarray,
                        M: np.ndarray, T: np.ndarray, sigma: np.ndarray,
                        data_for_tau: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
            I, A = NN.shape
            tau_hat = np.zeros((I, A, A), dtype=float)

            if data_for_tau is None:
                for i in range(I):
                    for a in range(A):
                        for b in range(A):
                            tau_hat[i, a, b] = (T[i,a,b] * T[i,b,a] / (T[i,a,a] * T[i,b,b])) ** (1/2)
                        
            else:
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

            return tau_hat


        def compute_tau(NN: np.ndarray, n: np.ndarray, W_hat: np.ndarray, S: np.ndarray,
                        M: np.ndarray, T: np.ndarray, sigma: np.ndarray,
                        data_for_tau: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
            I, A = NN.shape
            tau_hat = compute_tau_hat(NN, n, W_hat, S, M, T, sigma, data_for_tau)

            tau = np.empty((I,A,A))
            for i in range(I):
                for a in range(A):
                    for b in range(A):
                        tau[i, a, b] = tau_hat[i, a, b] ** (1 / (1 - sigma[i]))

            return tau


        def compute_mu(NN: np.ndarray, n: np.ndarray, W_hat: np.ndarray, S: np.ndarray,
                            M: np.ndarray, T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
            I, A = NN.shape
            mu = np.empty(I, dtype=float)
            W = compute_W(NN, n, W_hat, S, M, T, sigma)

            for i in range(I):
                mu[i] = (np.sum(S[i,:]) - np.sum(M[i])) / np.sum(W)

            return mu


        def compute_phi(NN: np.ndarray, n: np.ndarray, W_hat: np.ndarray, S: np.ndarray,
                            M: np.ndarray, T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
            I, A = NN.shape
            phi = np.empty((I,A), dtype=float)

            for i in range(I):
                for a in range(A):
                    phi[i,a] = S[i,a] / (sigma[i] * n[i,a])

            return phi


        def compute_rho_hat(NN: np.ndarray, n: np.ndarray, W_hat: np.ndarray, S: np.ndarray,
                            M: np.ndarray, T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
            I, A = NN.shape
            rho_hat = np.empty((I,A), dtype=float)

            tau_hat = compute_tau_hat(NN, n, W_hat, S, M, T, sigma)
            phi = compute_phi(NN, n, W_hat, S, M, T, sigma)

            for i in range(I):
                for a in range(A):
                    rho_hat[i,a] = np.sum(tau_hat[i,:,a] * n[i] * phi[i] ** (1 - sigma[i])) ** (1 / (1 - sigma[i]))

            return rho_hat


        def compute_gamma(NN: np.ndarray, n: np.ndarray, W_hat: np.ndarray, S: np.ndarray,
                        M: np.ndarray, T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
            I, A = NN.shape
            gamma = np.empty((I, A), dtype=float)

            alpha = compute_alpha(NN, n, W_hat, S, M, T, sigma)
            phi = compute_phi(NN, n, W_hat, S, M, T, sigma)
            rho_hat = compute_rho_hat(NN, n, W_hat, S, M, T, sigma)
            for i in range(I):
                for a in range(A):
                    gamma[i, a] = phi[i,a] * (S[i,a] / NN[i,a]) ** (-1 + np.sum(alpha[:,i,a])) * np.prod((rho_hat[:,a] / alpha[:,i,a]) ** (-alpha[:,i,a]))

            return gamma


        def solve_beta(initial_beta: Optional[np.ndarray], context: Dict[str, Any],
                    options: Optional[BetaSolverOptions] = None):
            """Solve the nonlinear system for beta.

            Parameters
            ----------
            initial_beta : Optional[np.ndarray]
                Initial guess for beta (shape (I,)). If None, a default is created.
            context : dict
                A dictionary with precomputed arrays available to the residual and jacobian.
            options : BetaSolverOptions
                Solver options and optional Jacobian callable with signature J(beta, context)->(I,I)

            Returns
            -------
            beta : np.ndarray
                Solution array of shape (I,).

            Notes
            -----
            - If a Jacobian is available, set `options.jac` to a callable that accepts
            (beta, context) and returns an (I,I) array.
            """
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


            if options is None:
                options = BetaSolverOptions()

            I = context['I']
            if initial_beta is None:
                initial_beta = np.ones(I, dtype=float)

            def wrapped_residual(x):
                return beta_residual(x, context)

            jac = None
            if options.jac is not None:
                def wrapped_jac(x):
                    return options.jac(x, context)
                jac = wrapped_jac

            lsq_kwargs = {}
            if options.extra_options:
                lsq_kwargs.update(options.extra_options)

            if jac is None:
                sol = least_squares(wrapped_residual, initial_beta, method=options.method, bounds=options.bounds, **lsq_kwargs)
            else:
                sol = least_squares(wrapped_residual, initial_beta, jac=jac, method=options.method, bounds=options.bounds, **lsq_kwargs)

            if not sol.success:
                raise RuntimeError(f"Beta solver failed: {sol.message}")
            
            res_norm = np.linalg.norm(sol.fun, ord=np.inf)

            return sol.x, res_norm


        def solve_theta(context: Dict[str, Any], TOL_F = 1e-8):
            
            def theta_residual(theta, context: Dict[str, Any]) -> np.ndarray:
                I = context['I']
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
                        print(f"警告: theta[{a}] の解法は失敗判定だが残差が十分小さいため受理します (||F||_inf={res_norms[a]:.1e}, 閾値={TOL_F:.1e}).")
                    else:
                        raise RuntimeError(f"theta[{a}] solver failed: {sol_a.message}, ||F||_inf={res_norms[a]:.1e}")

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



        I, A = _validate_shapes(NN, n, W_hat, S, M, T, sigma)
        N = compute_N(NN, n, W_hat, S, M, T, sigma)
        alpha = compute_alpha(NN, n, W_hat, S, M, T, sigma)
        W = compute_W(NN, n, W_hat, S, M, T, sigma)
        lam = compute_lambda(NN, n, W_hat, S, M, T, sigma)
        tau = compute_tau(NN, n, W_hat, S, M, T, sigma, data_for_tau)
        mu = compute_mu(NN, n, W_hat, S, M, T, sigma)
        phi = compute_phi(NN, n, W_hat, S, M, T, sigma)
        rho_hat = compute_rho_hat(NN, n, W_hat, S, M, T, sigma)
        gamma = compute_gamma(NN, n, W_hat, S, M, T, sigma)
        w = W / NN
        W_all = np.sum(W)

        context = {
            'I': I,
            'A': A,
            'NN': NN,
            'n': n,
            'W_hat': W_hat,
            'S': S,
            'M': M,
            'T': T,
            'sigma': sigma,
            'alpha': alpha,
            'W': W,
            'lam': lam,
            'tau': tau,
            'mu': mu,
            'phi': phi,
            'rho_hat': rho_hat,
            'gamma': gamma,
        }

        if verbose:
            print("------------ Computed parameters ------------")
            print("alpha =", alpha)
            print("lambda =", lam)
            print("tau =", tau)
            print("mu =", mu)
            print("gamma =", gamma)
            print("N =", N)
            print("w =", w)

        beta, beta_res_norm = solve_beta(beta_initial, context, options=beta_options)
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
    W_hat = data['W_hat']
    S = data['S']
    M = data['M']
    T = data['T']
    sigma = data['sigma']
    data_for_tau = data['data_for_tau']

    if verbose:
        print(f"    NN = {NN}")
        print(f"    n = {n}")
        print(f"    W_hat = {W_hat}")
        print(f"    S = {S}")
        print(f"    M = {M}")
        print(f"    T = {T}")

    calibration = calibrate_parameters(NN, n, W_hat, S, M, T, sigma, data_for_tau = data_for_tau, verbose = verbose, TOL_F = TOL_F)
    phi0, w0 = calibration['phi'], calibration['w']

    values = compute_equilibrium(calibration, phi_initial=phi0, w_initial=w0, verbose = False, TOL_F = TOL_F)
    valid = values['valid']
    phi, w, n = values['phi'], values['w'], values['n']
    S0 = values['S']
    ESS1 = values['ESS']

    values2 = compute2_phi_and_w(calibration, phi1=phi, w1=w, n1=n, verbose = False, TOL_F = TOL_F)
    valid2 = values2['valid']
    Pi = values2['Pi']
    ESS2 = values2['ESS']
    Pi_S= Pi / S0

    REV = (ESS2- ESS1) / ESS1
    # if verbose:
    #     print(" ")
    #     print("    REV =", REV)

    return {
        'Pi/S': Pi_S,
        'REV': REV,
        'valid': valid and valid2
    }





def draw_industry():
    I = 2
    A = 2
    sigma0 = 5
    population = 100
    random = True
    initial_computed = False
    seed = 1
    verbose = False
    TOL_F = 10.0
    delta = 0.01
    data_for_tau = None


    NN0 = np.ones((I, A)) * 1000
    n0 = np.ones((I, A)) * 202
    W_hat0 = np.ones((I, A)) * 1000
    S0 = np.ones((I, A)) * 3000
    M0 = np.ones((I,I,A)) * 1000
    T0 = np.ones((I,A,A)) * 1500

    sigma = np.ones(I) * sigma0


    p_list = np.linspace(0.01, 0.99, 401)
    solutions = []
    p_used = []

    sol_rev = []
    p_used_rev = []


    for p in p_list:
        print(f"p={p}の計算開始")

        # prop_ia = np.ones((I,A)) / 2
        prop_ia = np.array([
            [p, p],
            [1-p, 1-p]
        ])

        NN = NN0 * prop_ia
        n = n0 * prop_ia
        W_hat = W_hat0 * prop_ia
        S = S0 * prop_ia

        M = M0
        for i in range(I):
            for j in range(I):
                for a in range(A):
                    M[i,j,a] = S[j,a] / 3

        T = T0 * np.array([
            [[p, p],
            [p, p]],
            [[1-p, 1-p],
            [1-p, 1-p]]
        ])




        # print(f"    NN = {NN}")

        data = {
            'NN': NN,
            'n': n,
            'W_hat': W_hat,
            'S': S,
            'M': M,
            'T': T,
            'sigma': sigma,
            'data_for_tau': data_for_tau,
        }

        changes = compute_changes(data=data, verbose=verbose, TOL_F=TOL_F)
        valid = changes['valid']
        if valid:
            Pi_S = changes['Pi/S']
            REV = changes['REV']
            print("  Pi/S =", Pi_S)
            print("  REV =", REV)
            print(" ")

            solutions.append(Pi_S)
            p_used.append(p)

            if not np.any(np.isnan(REV)):
                sol_rev.append(REV)
                p_used_rev.append(p)


        changes['valid'] = False


    # グラフ描画
    solutions = np.array(solutions)
    sol_rev = np.array(sol_rev)

    plt.figure()
    for a in range(I):
        for i in range(A):
            plt.scatter(p_used, solutions[:,i,a], label=f"地域{a+1}・産業{i+1}")
    plt.xlabel("産業1の経済規模の割合", fontproperties=fp)
    plt.ylabel("企業の利潤変化/売上", fontproperties=fp)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(prop=fplegend)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Pi_via_industry.pdf")
    plt.close()

    plt.figure()
    for a in range(A):
        plt.scatter(p_used_rev, sol_rev[:, a], label=f"地域{a+1}")
    plt.xlabel("産業1の経済規模の割合", fontproperties=fp)
    plt.ylabel("地域の便益の変化率", fontproperties=fp)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(prop=fplegend)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Pi_via_industry_rev.pdf")
    plt.close()


def draw_area():
    I = 2
    A = 2
    sigma0 = 5
    population = 100
    random = True
    initial_computed = False
    seed = 1
    verbose = False
    TOL_F = 10.0
    delta = 0.01
    data_for_tau = None


    NN0 = np.ones((I, A)) * 1000
    n0 = np.ones((I, A)) * 202
    W_hat0 = np.ones((I, A)) * 1000
    S0 = np.ones((I, A)) * 3000
    M0 = np.ones((I,I,A)) * 1000
    T0 = np.ones((I,A,A)) * 1500

    sigma = np.ones(I) * sigma0


    p_list = np.linspace(0.05, 0.99, 401)
    solutions = []
    p_used = []

    sol_rev = []
    p_used_rev = []


    for p in p_list:
        print(f"p={p}の計算開始")

        # prop_ia = np.ones((I,A)) / 2
        prop_ia = np.array([
            [p, 1-p],
            [p, 1-p]
        ])

        NN = NN0 * prop_ia
        n = n0 * prop_ia
        W_hat = W_hat0 * prop_ia
        S = S0 * prop_ia

        M = M0
        for i in range(I):
            for j in range(I):
                for a in range(A):
                    M[i,j,a] = S[j,a] / 3

        T = T0 * np.array([
            [[p**2, p*(1-p)],
            [p*(1-p), (1-p)**2]],
            [[p**2, p*(1-p)],
            [p*(1-p), (1-p)**2]]
        ])




        # print(f"    NN = {NN}")

        data = {
            'NN': NN,
            'n': n,
            'W_hat': W_hat,
            'S': S,
            'M': M,
            'T': T,
            'sigma': sigma,
            'data_for_tau': data_for_tau,
        }

        changes = compute_changes(data=data, verbose=verbose, TOL_F=TOL_F)
        valid = changes['valid']
        if valid:
            Pi_S = changes['Pi/S']
            REV = changes['REV']
            print("  Pi/S =", Pi_S)
            print("  REV =", REV)
            print(" ")

            solutions.append(Pi_S)
            p_used.append(p)

            if not np.any(np.isnan(REV)):
                sol_rev.append(REV)
                p_used_rev.append(p)


        changes['valid'] = False


    # グラフ描画
    solutions = np.array(solutions)
    sol_rev = np.array(sol_rev)

    plt.figure()
    for a in range(I):
        for i in range(A):
            plt.scatter(p_used, solutions[:,i,a], label=f"地域{a+1}・産業{i+1}")
    plt.xlabel("地域1の経済規模の割合", fontproperties=fp)
    plt.ylabel("企業の利潤変化/売上", fontproperties=fp)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(prop=fplegend)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Pi_via_area.pdf")
    plt.close()

    plt.figure()
    for a in range(A):
        plt.scatter(p_used_rev, sol_rev[:, a], label=f"地域{a+1}")
    plt.xlabel("地域1の経済規模の割合", fontproperties=fp)
    plt.ylabel("地域の便益の変化率", fontproperties=fp)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(prop=fplegend)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Pi_via_area_rev.pdf")
    plt.close()


def draw_medium12():
    I = 2
    A = 2

    q = 0.5

    sigma0 = 5
    population = 100
    random = True
    initial_computed = False
    seed = 1
    verbose = False
    TOL_F = 10.0
    delta = 0.01
    data_for_tau = None


    NN0 = np.ones((I, A)) * 1000
    n0 = np.ones((I, A)) * 202
    W_hat0 = np.ones((I, A)) * 1000
    S0 = np.ones((I, A)) * 3000
    M0 = np.ones((I,I,A)) * 1000
    T0 = np.ones((I,A,A)) * 1500

    sigma = np.ones(I) * sigma0


    p_list = np.linspace(0.01, 0.99, 401)
    solutions = []
    p_used = []

    sol_rev = []
    p_used_rev = []


    for p in p_list:
        print(f"p={p}の計算開始")

        # prop_ia = np.ones((I,A)) / 2
        prop_ia = np.array([
            [q, q],
            [1-q, 1-q]
        ])

        NN = NN0 * prop_ia
        n = n0 * prop_ia
        W_hat = W_hat0 * prop_ia
        S = S0 * prop_ia

        # M = M0 * 2 * np.array([
        #     [[q-p+p*q, q-p+p*q],
        #     [p*(1-q), p*(1-q)]],
        #     [[p*(1-q), p*(1-q)],
        #     [(1-p)*(1-q), (1-p)*(1-q)]]
        # ])

        M = M0 * np.array([
            [[1/2, 1/2],
            [p, p]],
            [[1/2, 1/2],
            [1-p, 1-p]]
        ])

        T = T0 / 2




        # print(f"    NN = {NN}")

        data = {
            'NN': NN,
            'n': n,
            'W_hat': W_hat,
            'S': S,
            'M': M,
            'T': T,
            'sigma': sigma,
            'data_for_tau': data_for_tau,
        }

        changes = compute_changes(data=data, verbose=verbose, TOL_F=TOL_F)
        valid = changes['valid']
        if valid:
            Pi_S = changes['Pi/S']
            REV = changes['REV']
            print("  Pi/S =", Pi_S)
            print("  REV =", REV)
            print(" ")

            solutions.append(Pi_S)
            p_used.append(p)

            if not np.any(np.isnan(REV)):
                sol_rev.append(REV)
                p_used_rev.append(p)


        changes['valid'] = False


    # グラフ描画
    solutions = np.array(solutions)
    sol_rev = np.array(sol_rev)

    plt.figure()
    for a in range(I):
        for i in range(A):
            plt.scatter(p_used, solutions[:,i,a], label=f"地域{a+1}・産業{i+1}")
    plt.xlabel("産業2の投入中間財1の割合", fontproperties=fp)
    plt.ylabel("企業の利潤変化/売上", fontproperties=fp)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(prop=fplegend)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Pi_via_medium12.pdf")
    plt.close()

    plt.figure()
    for a in range(A):
        plt.scatter(p_used_rev, sol_rev[:, a], label=f"地域{a+1}")
    plt.xlabel("産業2の投入中間財1の割合", fontproperties=fp)
    plt.ylabel("地域の便益の変化率", fontproperties=fp)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(prop=fplegend)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Pi_via_medium12_rev.pdf")
    plt.close()


def draw_medium21():
    I = 2
    A = 2
    sigma0 = 5
    population = 100
    random = True
    initial_computed = False
    seed = 1
    verbose = False
    TOL_F = 10.0
    delta = 0.01
    data_for_tau = None


    NN0 = np.ones((I, A)) * 1000
    n0 = np.ones((I, A)) * 202
    W_hat0 = np.ones((I, A)) * 1000
    S0 = np.ones((I, A)) * 3000
    M0 = np.ones((I,I,A)) * 1000
    T0 = np.ones((I,A,A)) * 1500

    sigma = np.ones(I) * sigma0


    p_list = np.linspace(0.01, 0.99, 401)
    solutions = []
    p_used = []

    sol_rev = []
    p_used_rev = []


    for p in p_list:
        print(f"p={p}の計算開始")

        prop_ia = np.ones((I,A)) / 2
        # prop_ia = np.array([
        #     [p, p],
        #     [1-p, 1-p]
        # ])

        NN = NN0 * prop_ia
        n = n0 * prop_ia
        W_hat = W_hat0 * prop_ia
        S = S0 * prop_ia

        M = M0 * np.array([
            [[1-p, 1-p],
            [1/2, 1/2]],
            [[p, p],
            [1/2, 1/2]]
        ])

        T = T0 / 2




        # print(f"    NN = {NN}")

        data = {
            'NN': NN,
            'n': n,
            'W_hat': W_hat,
            'S': S,
            'M': M,
            'T': T,
            'sigma': sigma,
            'data_for_tau': data_for_tau,
        }

        changes = compute_changes(data=data, verbose=verbose, TOL_F=TOL_F)
        valid = changes['valid']
        if valid:
            Pi_S = changes['Pi/S']
            REV = changes['REV']
            print("  Pi/S =", Pi_S)
            print("  REV =", REV)
            print(" ")

            solutions.append(Pi_S)
            p_used.append(p)

            if not np.any(np.isnan(REV)):
                sol_rev.append(REV)
                p_used_rev.append(p)


        changes['valid'] = False


    # グラフ描画
    solutions = np.array(solutions)
    sol_rev = np.array(sol_rev)

    plt.figure()
    for a in range(I):
        for i in range(A):
            plt.scatter(p_used, solutions[:,i,a], label=f"地域{a+1}・産業{i+1}")
    plt.xlabel("産業1の投入中間財2の割合", fontproperties=fp)
    plt.ylabel("企業の利潤変化/売上", fontproperties=fp)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(prop=fplegend)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Pi_via_medium21.pdf")
    plt.close()

    plt.figure()
    for a in range(A):
        plt.scatter(p_used_rev, sol_rev[:, a], label=f"地域{a+1}")
    plt.xlabel("産業1の投入中間財2の割合", fontproperties=fp)
    plt.ylabel("地域の便益の変化率", fontproperties=fp)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(prop=fplegend)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Pi_via_medium21_rev.pdf")
    plt.close()


def draw_transported():
    I = 2
    A = 2
    sigma0 = 5
    population = 100
    random = True
    initial_computed = False
    seed = 1
    verbose = False
    TOL_F = 10.0
    delta = 0.01
    data_for_tau = None


    NN0 = np.ones((I, A)) * 1000
    n0 = np.ones((I, A)) * 202
    W_hat0 = np.ones((I, A)) * 1000
    S0 = np.ones((I, A)) * 3000
    M0 = np.ones((I,I,A)) * 1000
    T0 = np.ones((I,A,A)) * 1500

    sigma = np.ones(I) * sigma0


    p_list = np.linspace(0.01, 0.99, 401)
    solutions = []
    p_used = []

    sol_rev = []
    p_used_rev = []


    for p in p_list:
        print(f"p={p}の計算開始")

        prop_ia = np.ones((I,A)) / 2
        # prop_ia = np.array([
        #     [p, p],
        #     [1-p, 1-p]
        # ])

        NN = NN0 * prop_ia
        n = n0 * prop_ia
        W_hat = W_hat0 * prop_ia
        S = S0 * prop_ia

        M = M0 / 2

        T = T0 * np.array([
            [[1-p, p],
            [p, 1-p]],
            [[1-p, p],
            [p, 1-p]]
        ])




        # print(f"    NN = {NN}")

        data = {
            'NN': NN,
            'n': n,
            'W_hat': W_hat,
            'S': S,
            'M': M,
            'T': T,
            'sigma': sigma,
            'data_for_tau': data_for_tau,
        }

        changes = compute_changes(data=data, verbose=verbose, TOL_F=TOL_F)
        valid = changes['valid']
        if valid:
            Pi_S = changes['Pi/S']
            REV = changes['REV']
            print("  Pi/S =", Pi_S)
            print("  REV =", REV)
            print(" ")

            solutions.append(Pi_S)
            p_used.append(p)

            if not np.any(np.isnan(REV)):
                sol_rev.append(REV)
                p_used_rev.append(p)


        changes['valid'] = False


    # グラフ描画
    solutions = np.array(solutions)
    sol_rev = np.array(sol_rev)

    plt.figure()
    for a in range(I):
        for i in range(A):
            plt.scatter(p_used, solutions[:,i,a], label=f"地域{a+1}・産業{i+1}")
    plt.xlabel("産業1の投入中間財2の割合", fontproperties=fp)
    plt.ylabel("企業の利潤変化/売上", fontproperties=fp)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(prop=fplegend)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Pi_via_medium21.pdf")
    plt.close()

    plt.figure()
    for a in range(A):
        plt.scatter(p_used_rev, sol_rev[:, a], label=f"地域{a+1}")
    plt.xlabel("産業1の投入中間財2の割合", fontproperties=fp)
    plt.ylabel("地域の便益の変化率", fontproperties=fp)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(prop=fplegend)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Pi_via_medium21_rev.pdf")
    plt.close()





I = 2
A = 2
sigma0 = 5
population = 100
random = True
initial_computed = False
seed = 1
verbose = True
TOL_F = 10.0
delta = 0.01
data_for_tau = None



# draw_industry()
# draw_area()
draw_medium12()
# draw_medium21()