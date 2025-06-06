import numpy as np 

def soft_threshold(x, threshold):
    def method_1(x, threshold):
        # threshold = np.abs(threshold)
        if x > threshold:
            return x - threshold
        if x < -threshold:
            return x + threshold
        return 0

    def method_2(x, threshold):
        # threshold = np.abs(threshold)
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    return method_2(x, threshold)

def J(Y, lmbda):
    norm_two = np.linalg.norm(Y, 2)
    norm_inf = np.linalg.norm(Y, np.inf) / lmbda
    return np.max([norm_two, norm_inf])

def get_svp(S, mu):
    # svp = np.sum(S > 1 / mu)
    svp = np.count_nonzero(S > 1 / mu)
    return svp

def update_sv(sv, svp, d):
    if svp < sv:
        sv = np.min([svp + 1, d])
    else:
        sv = np.min([svp + np.round(0.05 * d), d])
    return sv

def svd_threshold(X, threshold, mu):
    U, S, V = np.linalg.svd(X, full_matrices=False)
    svp = get_svp(S, mu)
    A_update = np.dot(np.dot(U[:, :svp], np.diag(soft_threshold(S[:svp], threshold))), V[:svp, :])
    return svp, A_update

def IALM(D, lmbda=0.01, mu=None,  rho=1.6, sv=10., tol=1e-7, maxIter=1000, max_mu=1e7):
    """
    D: Data matrix
    A: Low-ramk matrix component
    E: Error matrix, Sparse component
    Y: Lagrange multiplier
    J: Dual norm
    k: Iteration
    sv: Predicted dimension
    svp:  # singular values in the sv singular values that are larger than 1 / µ
    primal_error: D - (A + E)
    """

    # Initialize variables
    dual_norm = J(D, lmbda)
    Y = D / dual_norm
    A = np.zeros_like(Y)
    E = np.zeros_like(Y)
    D_norm = np.linalg.norm(D, 'fro')
    D_norm_two = np.linalg.norm(D, 2)
    mu = 1.25 / D_norm_two if mu is None else mu
    d = Y.shape[1]

    for k in range(maxIter):
        
        # U, S, V = np.linalg.svd(D - E + (1 / mu) * Y, full_matrices=False)
        # svp = get_svp(S, mu)
        # A_update = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        svp, A = svd_threshold(D - E + (1 / mu) * Y, 1 / mu, mu)
        
        E = soft_threshold(D - A + (1 / mu) * Y, lmbda / mu)
        sv = update_sv(sv, svp, d)
        primal_error = D - (A + E)
        Y = Y + mu * primal_error
        mu = np.min([mu * rho, max_mu])

        # Check convergence
        stop_criterion = (np.linalg.norm(primal_error, 'fro') / D_norm) < tol
        if stop_criterion:
            break
    return A, E

def ALM(D, lmbda=0.01, mu=None, rho=6, sv=10., tol_inner=1e-6, tol_outer=1e-7, maxIter=1000, max_mu=1e7):
    """
    D: Data matrix
    A: Low-ramk matrix component
    E: Error matrix, Sparse component
    Y: Lagrange multiplier
    J: Dual norm
    j: Inner iteration
    k: Outer iteration
    sv: Predicted dimension
    svp:  # singular values in the sv singular values that are larger than 1 / µ
    primal_error: D - (A + E)
    """

    # Initialize variables
    dual_norm = J(D, lmbda)
    Y = D / dual_norm
    A_prev = np.zeros_like(Y)
    E_prev = np.zeros_like(Y)
    D_norm = np.linalg.norm(D, 'fro')
    mu = 0.5 / np.linalg.norm(np.sign(D), 2) if mu is None else mu
    d = Y.shape[1]

    for k in range(maxIter):
        for j in range(maxIter):
            # U, S, V = np.linalg.svd(D - E + (1 / mu) * Y, full_matrices=False)
            # svp = get_svp(S, mu)
            # A_update = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
            A = A_prev
            E = E_prev
            svp, A = svd_threshold(D - E + (1 / mu) * Y, 1 / mu, mu)
            E = soft_threshold(D - A + (1 / mu) * Y, lmbda / mu)
            sv = update_sv(sv, svp, d)

            # Check convergence
            stop_criterion_inner = ((np.linalg.norm(A - A_prev, 'fro') / D_norm) < tol_inner) or ((np.linalg.norm(E - E_prev, 'fro') / D_norm) < tol_inner)
            if stop_criterion_inner:
                break

        sv = np.min([svp + np.round(0.1 * d), d])
        primal_error = D - (A + E)
        Y = Y + mu * primal_error
        mu = np.min([mu * rho, max_mu])

        # Check convergence
        stop_criterion_outer = (np.linalg.norm(primal_error, 'fro') / D_norm) < tol_outer
        if stop_criterion_outer:
            break

    return A, E



class RPCA_ALM():

    """
    D: Data matrix
    A: Low-ramk matrix component
    E: Error matrix, Sparse component
    Y: Lagrange multiplier
    J: Dual norm
    k: Iteration
    sv: Predicted dimension
    svp:  # singular values in the sv singular values that are larger than 1 / µ
    primal_error: D - (A + E)
    """
    
    def __init__(self, lmbda=0.01, mu=None, rho=6, sv=10., tol_inner=1e-6, tol_outer=1e-7, maxIter=1000, max_mu=1e7, verbose=False):
        self.lmbda = lmbda
        self.mu = mu
        self.rho = rho
        self.sv = sv
        self.tol_inner = tol_inner
        self.tol_outer = tol_outer
        self.maxIter = maxIter
        self.max_mu = max_mu
        self.verbose = verbose

    def soft_threshold(self, x, threshold):
        def method_1(x, threshold):
            # threshold = np.abs(threshold)
            if x > threshold:
                return x - threshold
            if x < -threshold:
                return x + threshold
            return 0

        def method_2(x, threshold):
            # threshold = np.abs(threshold)
            return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
        return method_2(x, threshold)

    def J(self, Y, lmbda):
        norm_two = np.linalg.norm(Y, 2)
        norm_inf = np.linalg.norm(Y, np.inf) / lmbda
        return np.max([norm_two, norm_inf])

    def get_svp(self, S, mu):
        # svp = np.sum(S > 1 / mu)
        svp = np.count_nonzero(S > 1 / mu)
        return svp

    def update_sv(self, sv, svp, d):
        if svp < sv:
            sv = np.min([svp + 1, d])
        else:
            sv = np.min([svp + np.round(0.05 * d), d])
        return sv

    def svd_threshold(self, X, threshold, mu):
        U, S, V = np.linalg.svd(X, full_matrices=False)
        svp = get_svp(S, mu)
        A_update = np.dot(np.dot(U[:, :svp], np.diag(soft_threshold(S[:svp], threshold))), V[:svp, :])
        return svp, A_update

    def inexact_augmented_lagrange_multiplier(self):
        
        # Initialize variables
        Y = self.Y
        A = np.zeros_like(Y)
        E = np.zeros_like(Y)
        D_norm_two = np.linalg.norm(self.D, 2)
        mu = 1.25 / D_norm_two if self.mu is None else self.mu
        sv = self.sv

        for k in range(self.maxIter):
            
            # U, S, V = np.linalg.svd(self.D - E + (1 / mu) * Y, full_matrices=False)
            # svp = self.get_svp(S, mu)
            # A_update = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
            svp, A = self.svd_threshold(self.D - E + (1 / mu) * Y, 1 / mu, mu)
            
            E = self.soft_threshold(self.D - A + (1 / mu) * Y, self.lmbda / mu)
            sv = self.update_sv(sv, svp, self.d)
            primal_error = self.D - (A + E)
            Y = Y + mu * primal_error
            mu = np.min([mu * self.rho, self.max_mu])

            # Check convergence
            stop_criterion = (np.linalg.norm(primal_error, 'fro') / self.D_norm) < self.tol_outer
            if stop_criterion:
                break
        return A, E

    def augmented_lagrange_multiplier(self):

        # Initialize variables
        Y = self.Y
        A_prev = np.zeros_like(Y)
        E_prev = np.zeros_like(Y)
        mu = 0.5 / np.linalg.norm(np.sign(self.D), 2) if self.mu is None else self.mu
        sv = self.sv

        for k in range(self.maxIter):
            for j in range(self.maxIter):
                # U, S, V = np.linalg.svd(self.D - E + (1 / mu) * Y, full_matrices=False)
                # svp = self.get_svp(S, mu)
                # A_update = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
                A = A_prev
                E = E_prev
                svp, A = self.svd_threshold(self.D - E + (1 / mu) * Y, 1 / mu, mu)
                E = self.soft_threshold(self.D - A + (1 / mu) * Y, self.lmbda / mu)
                sv = update_sv(sv, svp, self.d)

                # Check convergence
                stop_criterion_inner = ((np.linalg.norm(A - A_prev, 'fro') / self.D_norm) < self.tol_inner) or ((np.linalg.norm(E - E_prev, 'fro') / self.D_norm) < self.tol_inner)
                if stop_criterion_inner:
                    break

            sv = np.min([svp + np.round(0.1 * self.d), self.d])
            primal_error = self.D - (A + E)
            Y = Y + mu * primal_error
            mu = np.min([mu * self.rho, self.max_mu])

            # Check convergence
            stop_criterion_outer = (np.linalg.norm(primal_error, 'fro') / self.D_norm) < self.tol_outer
            if stop_criterion_outer:
                break

        return A, E

    def fit(self, D, method='ALM'):
        # Initialize variables
        self.D = D
        self.dual_norm = J(D, self.lmbda)
        self.Y = D / self.dual_norm
        self.D_norm = np.linalg.norm(D, 'fro')
        self.d = self.Y.shape[1]

        if 'in' in method.lower() or 'ialm' == method.lower():
            return self.inexact_augmented_lagrange_multiplier()
        else:
            return self.augmented_lagrange_multiplier()


if __name__ == '__main__':
    # Example usage
    # Generate a low-rank matrix with sparse noise
    n = 100  # Number of samples
    d = 50  # Dimensionality
    r = 5  # Rank
    p = 0.1  # Proportion of sparse noise
    X = np.random.randn(n, r) @ np.random.randn(r, d)
    E = np.random.randn(n, d)
    mask = np.random.rand(n, d) < p
    X += mask * E

    E_ori = mask * E


    rpca = RPCA(lmbda=1.0)


    # Apply RPCA using ALM
    A, E = IALM(X, lmbda=1.0)

    # Print the recovered low-rank and sparse components
    # print("Low-rank component:")
    # print(A)
    # print(np.shape(A))
    # print("\nSparse component:")
    # print(E)
    # print(np.shape(E))
    # print('\nprimal_error')
    print(np.max(X - (A + E)))
    print(np.max(E - E_ori))
    A, E = rpca.fit(X, method='IALM')
    print(np.max(X - (A + E)))
    print(np.max(E - E_ori))

    # Apply RPCA using ALM
    A, E = ALM(X, lmbda=1.0)
    # print("Low-rank component:")
    # print(np.max(A - L))
    # print("\nSparse component:")
    # print(np.max(E - S))
    # print("\n component:")
    # print(np.max(E - S))
    # print('\nprimal_error')
    print(np.max(X - (A + E)))
    print(np.max(E - E_ori))
    A, E = rpca.fit(X, method='ALM')
    print(np.max(X - (A + E)))
    print(np.max(E - E_ori))
