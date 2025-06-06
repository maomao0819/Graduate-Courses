import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Lasso

class MNIST():
    def __init__(self):
        # Load MNIST dataset
        self.dataset = fetch_openml('mnist_784', as_frame=False)   # as_frame: True for Pandas.Dataframe, False for numpy.array
        self.data = self.dataset['data']
        self.target = self.dataset['target']


    def showInfo(self):
        print(self.data.keys())
        print(self.data.shape)
        print(self.target.shape)


    def showExampleImage(self):
        fig = plt.figure(figsize=(15, 3))
        fig.patch.set_facecolor('white')
        for i in range(9):
            plt.subplot(191 + i)
            plt.imshow(self.data[i].reshape(28, 28), 'gray')
            plt.title(self.target[i])
            plt.axis('off')

    
    def mean(self, data=None):
        if data is None:
            return np.mean(self.data, axis=0)
        return np.mean(data, axis=0)
    

    # Filter images labeled
    def extractData(self, data=None, target=None, label='5'):
        if data is None or target is None:
            return self.data[self.target == label]
            # return self.data[np.where(self.target == label)]
        return data[target == label]
        
    
    def extractFirstData(self, data=None, target=None, num=10000):
        if data is None or target is None:
            return self.data[:num], self.target[:num]
        return data[:num], target[:num]


class PCA():
    def __init__(self, data):
        self.data = data


    def mean(self, data=None):
        if data is None:
            return np.mean(self.data, axis=0)
        return np.mean(data, axis=0)
    
    
    def get_data_centered(self, data=None):
        # Calculate the mean of the data
        # data_mean = self.mean(data)
        data_mean = self.mean()
        # mean = np.mean(X, axis=0)
        if data is None:
            # Center the data
            data_centered = self.data - data_mean
        else:
            # Center the data
            data_centered = data - data_mean
        return data_centered
    

    def getEigenvaluesEigenvectors(self, n_components=None):
        data_centered = self.get_data_centered()
        
        # Compute the covariance matrix
        covariance_matrix = np.cov(data_centered.T)
        # covariance_matrix = np.cov(data_centered, rowvar=False)
        
        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        eigenvalues = eigenvalues.astype(np.float64)
        eigenvectors = eigenvectors.astype(np.float64)
        
        # Sort eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # Select the desired number of components
        if n_components:
            selected_eigenvalues = sorted_eigenvalues[:n_components]
            selected_eigenvectors = sorted_eigenvectors[:, :n_components]
        else:
            selected_eigenvalues = sorted_eigenvalues
            selected_eigenvectors = sorted_eigenvectors

        return selected_eigenvalues, selected_eigenvectors.T
    

    def centered_pca(self, data=None, eigenvectors=None, n_components=None, data_id=None):
        # Center the data
        if data is None:
            data_centered = self.get_data_centered()
        else:
            data_centered = self.get_data_centered(data)
        
        if eigenvectors is None:
            _, eigenvectors = self.getEigenvaluesEigenvectors(n_components=n_components)
        
        if data_id is None:
            selected_data_centered = data_centered
        else:
            selected_data_centered = data_centered[data_id]
        # Project the centered data onto the selected eigenvectors
        data_pca = np.dot(selected_data_centered, eigenvectors.T)

        return data_pca
    

    def reconstruct(self, eigenvectors=None, data_pca=None, n_components=None, data_id=None):
        # Calculate the mean of the data
        data_mean = self.mean()
        # mean = np.mean(X, axis=0)
        
        if eigenvectors is None:
            _, eigenvectors = self.getEigenvaluesEigenvectors(n_components=n_components)
        if data_pca is None:
            data_pca = self.centered_pca(eigenvectors=eigenvectors, n_components=n_components, data_id=data_id)

        # Reconstruct the data from the PCA-transformed representation
        data_reconstruct = np.dot(data_pca, eigenvectors) + data_mean

        return data_reconstruct


# def centered_pca(X, n_components=None, n_samples=None):
#     # Calculate the mean of the data
#     mean = X.mean()
#     # mean = np.mean(X, axis=0)
    
#     # Center the data
#     X_centered = X - mean
    
#     # Compute the covariance matrix
#     covariance_matrix = np.cov(X_centered.T)
#     # covariance_matrix = np.cov(X_centered, rowvar=False)
    
#     # Perform eigenvalue decomposition
#     eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
#     eigenvalues = eigenvalues.astype(np.float64)
#     eigenvectors = eigenvectors.astype(np.float64)
    
#     # Sort eigenvalues in descending order
#     sorted_indices = np.argsort(eigenvalues)[::-1]
#     sorted_eigenvalues = eigenvalues[sorted_indices]
#     sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
#     # Select the desired number of components
#     if n_components:
#         selected_eigenvalues = sorted_eigenvalues[:n_components]
#         selected_eigenvectors = sorted_eigenvectors[:, :n_components]
#     else:
#         selected_eigenvalues = sorted_eigenvalues
#         selected_eigenvectors = sorted_eigenvectors
    
#     if n_samples:
#         selected_X_centered = X_centered[:n_samples]
#     else:
#         selected_X_centered = X_centered
    
#     # Project the centered data onto the selected eigenvectors
#     X_pca = np.dot(selected_X_centered, selected_eigenvectors)
    
#     # Reconstruct the data from the PCA-transformed representation
#     X_reconstructed = np.dot(X_pca, selected_eigenvectors.T) + mean
    
#     return selected_eigenvectors.T, selected_eigenvalues, X_pca, X_reconstructed


def L2norm(image1, image2):
    return np.sqrt(np.sum(np.square(image1 - image2)))



class OMP():
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix


    def normalized(self):
        self.data_matrix = self.data_matrix / (np.linalg.norm(self.data_matrix, axis = 0))


    def baseConstruct(self, signal, sparsity, tolerance=1e-6):
        '''
        data_matrix: [n_feature, n_samples] = [base_dim * n_base]
        signal: [n_feature] = [base_dim]
        coef: [n_base] = [n_samples]
        bases: [base_dim, sparsity (n_bases)]
        '''
        # self.normalized()
        base_dim, n_base = self.data_matrix.shape
        coef = np.zeros(n_base)
        residual = signal.copy()
        bases_idx = []
        
        for _ in range(sparsity):
            # Find the index of the maximum inner product
            base_idx = np.argmax(np.abs(np.dot(self.data_matrix.T, residual)))
            
            # Add the index to the bases_idx
            bases_idx.append(base_idx)
            
            # Update the least-squares estimate
            bases = self.data_matrix[:, bases_idx]
            coef_hat = np.linalg.lstsq(bases, signal, rcond=None)[0]
            coef[bases_idx] = coef_hat
            
            # Calculate the residual
            residual = signal - np.dot(bases, coef_hat)
            
            # Stop if the residual is below a threshold
            if np.linalg.norm(residual) < tolerance:
                break

        return coef, bases.T
    

    def reconstruct(self, coef):
        return self.data_matrix @ coef



class LASSO():
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix


    def normalized(self):
        self.data_matrix = self.data_matrix / (np.linalg.norm(self.data_matrix, axis = 0))

    # Shrinks the input values towards zero based on the threshold
    def soft_threshold(self, data, threshold):
        return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)
        # if data > threshold:
        #     return data - threshold
        # elif data < -threshold:
        #     return data + threshold
        # else:
        #     return 0


    def lasso_coordinate_descent(self, signal, alpha=0.1, max_iter=1000, tolerance=1e-4):
        self.normalized()
        n_features, n_samples = self.data_matrix.shape
        coef = np.zeros(n_samples)
        for _ in range(max_iter):
            prev_coef = np.copy(coef)
            # Updates the weights for each feature 
            for sample_id in range(n_samples):
                project = np.dot(self.data_matrix, coef)
                curr_project = self.data_matrix[:, sample_id] * coef[sample_id]
                # Compute the partial residuals. Residual is a kind of distance or error.
                residual = signal - (project - curr_project)
                # Compute the simple least squares coefficient of these residuals on jth predictor. 
                coef_lstsq_r = np.dot(self.data_matrix[:, sample_id], residual) / n_features
                # Update the weights Beta by soft-thresholding from residual to fix reduce the distance between 
                # signal and the difference of project and curr_project 
                coef[sample_id] = self.soft_threshold(coef_lstsq_r, alpha)
            #  Until convergence
            if np.linalg.norm(coef - prev_coef) < tolerance:
                break
        return coef


    def reconstruct(self, coef):
        return coef @ self.data_matrix.T


class Solve():
    def __init__(self):
        os.makedirs('fig', exist_ok=True)
        self.mnist = MNIST()
        
        self.Q1()
        self.Q2()
        self.Q3()
        self.Q4()
        self.Q5()
        self.Q6()
        self.Q7()
        self.Bonus()
        
    # Q1: Show the mean of all 70,000 images.
    def Q1(self):
        mean_image = self.mnist.mean()
        plt.title('Q1: Show the mean of all 70,000 images.')
        plt.imshow(mean_image.reshape(28, 28), 'gray')
        plt.savefig('fig/Q1.jpg')
        plt.close()
        return


    # Q2: Extract all the “5” images (6313 vectors). Use centered PCA (5's center) to decompose. Show the eigenvectors with the three largest 
    # eigenvalues. Show the corresponding eigenvalues as well.
    def Q2(self):
        data_5 = self.mnist.extractData(label='5')
        pca = PCA(data_5)
        eigenvalues, eigenvectors = pca.getEigenvaluesEigenvectors(n_components=3)
        # eigenvectors, eigenvalues, _, _ = centered_pca(data_5, n_components=3)
        order_name = ['', '2nd', '3rd']
        fig = plt.figure(figsize=(15, 5))
        fig.patch.set_facecolor('white')
        for i in range(3):
            # print(f'The {order_name[i]} largest eigenvalues "{eigenvalues[i]}" and its corresponding eigenvectors "{eigenvectors[i]}".')
            plt.subplot(131 + i)
            plt.title(f'The {order_name[i]} largest eigenvalues: {eigenvalues[i]:.4f}.')
            plt.imshow(eigenvectors[i].reshape(28, 28), 'gray')
            plt.axis('off')
        plt.savefig('fig/Q2.jpg')
        plt.close()
        return


    # Q3: Extract all the “5” images. Use centered PCA and the top {3,10,30,100} eigenvectors to reconstruct the first “5” image. 
    # Explain your results.
    def Q3(self):
        n_eigenvectors = [3, 10, 30, 100]
        data_5 = self.mnist.extractData(label='5')
        pca = PCA(data_5)
        fig = plt.figure(figsize=(20, 5))
        fig.patch.set_facecolor('white')
        plt.subplot(151)
        plt.title('Original')
        plt.imshow(data_5[0].reshape(28, 28), 'gray')
        plt.axis('off')
        for i in range(4):
            reconstruction = pca.reconstruct(n_components=n_eigenvectors[i], data_id=1-1)
            plt.subplot(152 + i)
            plt.title(f'Reconstructed with {n_eigenvectors[i]} eigenvectors.')
            plt.imshow(reconstruction.reshape(28, 28), 'gray')
            plt.axis('off')
        plt.savefig('fig/Q3.jpg')
        plt.close()
        return


    # Q4: Extract the first 10,000 images. Next, extract all the “1”, “3”, and "6" (from the 10,000 images). 
    # Use centered PCA ([1,3,6]'s center) to reduce the dimension from 784 to 2 (the two largest eigenvalues). 
    # Plot those points in a 2-D plane using plt.scatter function with different colors. Explain your results.
    def Q4(self):
        data, target = self.mnist.extractFirstData(num=10000)
        plt.title('Q4: Reduce the dimension from 784 to 2.')
        data_1 = self.mnist.extractData(data=data, target=target, label='1')
        data_3 = self.mnist.extractData(data=data, target=target, label='3')
        data_6 = self.mnist.extractData(data=data, target=target, label='6')
        data_1_3_6 = np.concatenate((data_1, data_3, data_6), axis=0)
        pca = PCA(data_1_3_6)
        _, eigenvectors = pca.getEigenvaluesEigenvectors(n_components=2)
        data_1_pca = pca.centered_pca(data=data_1, eigenvectors=eigenvectors)
        data_3_pca = pca.centered_pca(data=data_3, eigenvectors=eigenvectors)
        data_6_pca = pca.centered_pca(data=data_6, eigenvectors=eigenvectors)
        plt.scatter(data_1_pca[:, 0], data_1_pca[:, 1], alpha=0.3, c='b')
        plt.scatter(data_3_pca[:, 0], data_3_pca[:, 1], alpha=0.3, c='r')
        plt.scatter(data_6_pca[:, 0], data_6_pca[:, 1], alpha=0.3, c='g')
        plt.legend(('1', '3', '6'), loc='lower right')
        plt.savefig('fig/Q4.jpg')
        plt.close()
        return


    # Q5: Define the first 10,000 images as training set. Find the 5 bases of the #10001 image ("3") with sparsity = 5. 
    # Show the 5 bases. What do you observe?
    def Q5(self):
        n_train_images = 10000
        data_matrix = self.mnist.data[:n_train_images].T
        # data_matrix /= np.linalg.norm(data_matrix, axis=0)
        data_matrix = data_matrix / (np.linalg.norm(data_matrix, axis=0))
        image_id = 10001
        signal = self.mnist.data[image_id-1]
        omp = OMP(data_matrix=data_matrix)
        sparsity = 5
        _, bases = omp.baseConstruct(signal, sparsity)
        
        # Plot the bases
        fig = plt.figure(figsize=(20, 4))
        fig.patch.set_facecolor('white')
        for i in range(sparsity):
            plt.subplot(1, sparsity, i+1)
            plt.title('Basis {}'.format(i+1))
            plt.imshow(bases[i].reshape(28, 28), cmap='gray')
            plt.axis('off')
        plt.savefig('fig/Q5.jpg')
        plt.close()
        return


    # Q6: Define the first 10,000 images as training set. Find the bases of the #10002 image ("8") with sparsity = {5,10,40,200}. 
    # Show the reconstruction images. Calculate their reconstruction errors using L-2 norm (Euclidean distance). Explain your results.
    def Q6(self):
        n_train_images = 10000
        data_matrix = self.mnist.data[:n_train_images].T
        # data_matrix /= np.linalg.norm(data_matrix, axis=0)
        data_matrix = data_matrix / (np.linalg.norm(data_matrix, axis=0))
        image_id = 10002
        signal = self.mnist.data[image_id-1]
        omp = OMP(data_matrix=data_matrix)
        sparsities = [5, 10, 40, 200]
        fig = plt.figure(figsize=(20, 5))
        fig.patch.set_facecolor('white')
        plt.subplot(151)
        plt.title('Original')
        plt.imshow(signal.reshape(28, 28), cmap='gray')
        plt.axis('off')
        for i, sparsity in enumerate(sparsities):
            coef, _ = omp.baseConstruct(signal, sparsity)
            reconstruct_image = omp.reconstruct(coef)
            reconstruct_image = reconstruct_image.reshape(28, 28)
            reconstruction_error = L2norm(reconstruct_image, signal.reshape(28, 28))
            plt.subplot(1, 5, i+2)
            plt.title(f'Reconstructed with {sparsities[i]} bases\nThe reconstruction errors:{reconstruction_error:.4f}.')
            plt.imshow(reconstruct_image.reshape(28, 28), cmap='gray')
            plt.axis('off')
        plt.savefig('fig/Q6.jpg')
        plt.close()
        return


    # Q7: Extract all the "8" images from the dataset (6825 vectors).
    # 1. Use centered PCA to reconstruct the last “8”. (Remain 5 largest eigenvalues.) 
    # 2. Use the first 6824 images as the base set. Use OMP to find the base and reconstruct the last "8". (Sparsity=5)
    # 3. As 2, use "lasso" to find the bases and reconstruct the images.
    # 4. Adjust the lasso parameters. Explain your experiments and results.
    def Q7(self):
        data_8 = self.mnist.extractData(label='8')
        pca = PCA(data_8)
        reconstruction_PCA = pca.reconstruct(n_components=5, data_id=-1).reshape(28, 28)

        n_train_images = 6824
        data_matrix = data_8[:6824].T
        # data_matrix /= np.linalg.norm(data_matrix, axis=0)
        data_matrix = data_matrix / (np.linalg.norm(data_matrix, axis=0))
        signal = data_8[-1]
        omp = OMP(data_matrix=data_matrix)
        coef, _ = omp.baseConstruct(signal, sparsity=5)
        reconstruction_omp = omp.reconstruct(coef).reshape(28, 28)
        reconstruction_loss_OMP = L2norm(reconstruction_omp, signal.reshape(28, 28))

        # Apply Lasso to find sparse coefficients for the target image
        lasso_alpha_01 = Lasso(alpha=0.1)
        lasso_alpha_01.fit(data_matrix, signal)
        coefficients = lasso_alpha_01.coef_
        # Reconstruct the target image using the sparse coefficients and the base set
        reconstruction_lasso_alpha_01 = np.dot(data_matrix, coefficients).reshape(28, 28)
        reconstruction_loss_lasso_alpha_01 = L2norm(reconstruction_lasso_alpha_01, signal.reshape(28, 28))

        lasso_alpha_05 = Lasso(alpha=0.5)
        lasso_alpha_05.fit(data_matrix, signal)
        coefficients = lasso_alpha_05.coef_
        reconstruction_lasso_alpha_05 = np.dot(data_matrix, coefficients).reshape(28, 28)
        reconstruction_loss_lasso_alpha_05 = L2norm(reconstruction_lasso_alpha_05, signal.reshape(28, 28))

        lasso_alpha_05_no_intercept = Lasso(alpha=0.5, fit_intercept=False)
        lasso_alpha_05_no_intercept.fit(data_matrix, signal)
        coefficients = lasso_alpha_05_no_intercept.coef_
        reconstruction_lasso_alpha_05_no_intercept = np.dot(data_matrix, coefficients).reshape(28, 28)
        reconstruction_loss_lasso_alpha_05_no_intercept= L2norm(reconstruction_lasso_alpha_05_no_intercept, signal.reshape(28, 28))

        lasso_alpha_09_no_intercept = Lasso(alpha=0.9, fit_intercept=False)
        lasso_alpha_09_no_intercept.fit(data_matrix, signal)
        coefficients = lasso_alpha_09_no_intercept.coef_
        reconstruction_lasso_alpha_09_no_intercept = np.dot(data_matrix, coefficients).reshape(28, 28)
        reconstruction_loss_lasso_alpha_09_no_intercept = L2norm(reconstruction_lasso_alpha_09_no_intercept, signal.reshape(28, 28))

        fig = plt.figure(figsize=(30, 5))
        fig.patch.set_facecolor('white')
        plt.subplot(171)
        plt.title('Original.')
        plt.imshow(signal.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.subplot(172)
        plt.title(f'Reconstructed with centered PCA.')
        plt.imshow(reconstruction_PCA, cmap='gray')
        plt.axis('off')
        plt.subplot(173)
        plt.title(f'Reconstructed with OMP.\nReconstruction errors: {reconstruction_loss_OMP:.4f}.')
        plt.imshow(reconstruction_omp, cmap='gray')
        plt.axis('off')
        plt.subplot(174)
        plt.title(f'Reconstructed with Lasso\n(alpha = 0.1 and fit intercept).\nReconstruction errors: {reconstruction_loss_lasso_alpha_01:.4f}.')
        plt.imshow(reconstruction_lasso_alpha_01, cmap='gray')
        plt.axis('off')
        plt.subplot(175)
        plt.title(f'Reconstructed with Lasso\n(alpha = 0.5 and fit intercept).\nReconstruction errors: {reconstruction_loss_lasso_alpha_05:.4f}.')
        plt.imshow(reconstruction_lasso_alpha_05, cmap='gray')
        plt.axis('off')
        plt.subplot(176)
        plt.title(f'Reconstructed with Lasso\n(alpha = 0.5 no fit intercept).\nReconstruction errors: {reconstruction_loss_lasso_alpha_05_no_intercept:.4f}.')
        plt.imshow(reconstruction_lasso_alpha_05_no_intercept, cmap='gray')
        plt.axis('off')
        plt.subplot(177)
        plt.title(f'Reconstructed with Lasso\n(alpha = 0.9 no fit intercept).\nReconstruction errors: {reconstruction_loss_lasso_alpha_09_no_intercept:.4f}.')
        plt.imshow(reconstruction_lasso_alpha_09_no_intercept, cmap='gray')
        plt.axis('off')
        plt.savefig('fig/Q7.jpg')
        plt.close()
        return


    # Bonus: Write the lasso function (handcraft) using coordinate descent.
    # Show your code fragment in the report.
    # Explain your implementation in the report.
    def Bonus(self):
        data_8 = self.mnist.extractData(label='8')
        n_train_images = 6824
        data_matrix = data_8[:6824].T
        # data_matrix /= np.linalg.norm(data_matrix, axis=0)
        data_matrix = data_matrix / (np.linalg.norm(data_matrix, axis=0))
        signal = data_8[-1]
        lasso = LASSO(data_matrix)
        alpha = 0.1
        coefficients = lasso.lasso_coordinate_descent(signal, alpha)
        reconstructed_image = lasso.reconstruct(coefficients)

        fig = plt.figure(figsize=(20, 10))
        fig.patch.set_facecolor('white')
        plt.subplot(121)
        plt.title('Original.')
        plt.imshow(signal.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.subplot(122)
        plt.title(f'Reconstructed with handcraft Lasso.')
        plt.imshow(reconstructed_image.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.savefig('fig/Bonus.jpg')
        plt.close()
        return


def main():
    Solve()


if __name__ == '__main__':
    main()
