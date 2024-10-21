import numpy as np
from kmeans import KMeans
from numpy.linalg import LinAlgError
from tqdm import tqdm
SIGMA_CONST = 1e-06
LOG_CONST = 1e-32
FULL_MATRIX = True


class GMM(object):

    def __init__(self, X, K, max_iters=100):
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        self.N = self.points.shape[0]
        self.D = self.points.shape[1]
        self.K = K

    def softmax(self, logit):
        """		
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error.
        """
        # raise NotImplementedError
        logit = np.exp(logit - np.max(logit, axis=1, keepdims=True))
        return logit / np.sum(logit, axis=1, keepdims=True)

    def logsumexp(self, logit):
        """		
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        s = np.log(np.sum(np.exp(logit - np.max(logit, axis=1, keepdims=True)), axis=1, keepdims=True))
        s = np.add(s, np.max(logit, axis=1, keepdims=True))

        
        return s
        # raise NotImplementedError

    def normalPDF(self, points, mu_i, sigma_i):
        """		
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian
        
        Hint:
            np.diagonal() should be handy.
        """
        pdf = np.ones(points.shape[0])
        for d in range(points.shape[1]):
            sigma = np.sqrt(np.diagonal(sigma_i))
            product = (1 / (np.sqrt(2 * np.pi * sigma[d]**2))) * (np.exp((-1 / (2 * sigma[d]**2)) * ((points[:, d] - mu_i[d])**2)))
            pdf *= product
        return pdf
        # raise NotImplementedError

    def multinormalPDF(self, points, mu_i, sigma_i):
        """		
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian
        
        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. Note the value in self.D may be outdated and not correspond to the current dataset.
            3. You may wanna check if the matrix is singular before implementing calculation process.
        """
        raise NotImplementedError

    def create_pi(self):
        """		
        Initialize the prior probabilities
        Args:
        Return:
        pi: numpy array of length K, prior
        """
        # raise NotImplementedError
        return np.ones(self.K) / self.K
    
    def create_mu(self):
        """		
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        mu = np.zeros((self.K, self.D))
        mu = self.points[np.random.choice(self.N, self.K, replace=True), :]
        return mu
        # raise NotImplementedError

    def create_sigma(self):
        """		
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """
        sig = np.zeros((self.K, self.D, self.D))	
        for k in range(self.K):
            sig[k] = np.eye(self.D)
        return sig

    def _init_components(self, **kwargs):
        """		
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case
        
            Hint: np.random.seed(5) must be used at the start of this function to ensure consistent outputs.
        """
        # raise NotImplementedError
        np.random.seed(5)
        pi = self.create_pi()
        mu = self.create_mu()
        sigma = self.create_sigma()
        return pi, mu, sigma

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """		
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        
        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        likelihood_matrix = np.zeros((self.N, self.K))
        for k in range(self.K):
            if full_matrix is not True:
                pdf = self.normalPDF(self.points, mu[k], sigma[k]) + LOG_CONST
                log_pdf = np.log(pdf)
            temp = pi[k] + LOG_CONST
            likelihood_matrix[:, k] = log_pdf + np.log(temp)
        return likelihood_matrix
                
        
        # raise NotImplementedError

    def _E_step(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """		
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        
        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # raise NotImplementedError
        matrix = np.zeros((self.N, self.K))
        if full_matrix is not True:
            matrix = self.softmax(self._ll_joint(pi, mu, sigma, full_matrix))
        return matrix
        

    def _M_step(self, tau, full_matrix=FULL_MATRIX, **kwargs):
        """		
        Args:
            tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
        
        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        
        # raise NotImplementedError
        if full_matrix is not True:
            pi = np.sum(tau, axis=0) / self.N
            mu = np.zeros((self.K, self.D))
            sigma = np.zeros((self.K, self.D, self.D))
            for k in range(self.K):
                mu[k, :] = np.dot(tau[:, k], self.points) / np.sum(tau[:, k])
                difference = self.points - mu[k, :]
                weighted = difference * tau[:, k][:, np.newaxis]
                sigmaK = np.dot(weighted.T, difference) / np.sum(tau[:, k])
                sigma[k, :, :] = np.diag(np.diag(sigmaK))
            return pi, mu, sigma

    def __call__(
        self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs
    ):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        prev_loss = None
        for it in pbar:
            # E-step
            tau = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(tau, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description("iter %d, loss: %.4f" % (it, loss))
        return tau, (pi, mu, sigma)


def cluster_pixels_gmm(image, K, max_iters=10, full_matrix=True):
    """	
    Clusters pixels in the input image
    
    Each pixel can be considered as a separate data point (of length 3),
    which you can then cluster using GMM. Then, process the outputs into
    the shape of the original image, where each pixel is its most likely value.
    
    Args:
        image: input image of shape(H, W, 3)
        K: number of components
        max_iters: maximum number of iterations in GMM. Default is 10
        full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
    Return:
        clustered_img: image of shape(H, W, 3) after pixel clustering
    
    Hints:
        What do mu and tau represent?
    """
    # raise NotImplementedError
    image = image.astype(np.float32)
    height, width, channel = image.shape
    hw3_img = np.reshape(image, [-1, channel]).astype(np.float32)
    gmm = GMM(hw3_img, K, max_iters)
    tau, (pi, mu, sigma) = gmm(full_matrix)
    cluster = np.argmax(tau, axis=1)
    center = mu
    img = np.reshape(center[cluster], (height, width, channel))
    return img.astype(np.float32)
    
