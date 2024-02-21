
import torch



class Kernel:

    def __add__(self, other):
        return AdditiveKernel(self, other)

    def __mul__(self, other):
        if isinstance(other, (float, torch.Tensor)):
            other = other if torch.is_tensor(other) else other
            return ProdKernel(kernel1=self, weight1=other)
        elif isinstance(other, Kernel):
            return ProdKernel(kernel1=self, kernel2=other)
        else:
            raise NotImplementedError
    
class AdditiveKernel(Kernel):
    
    def __init__(self, kernel1, kernel2) -> None:
        self.kernel1 = kernel1
        self.kernel2 = kernel2
    
    def __call__(self, X, x=None):
        kernel = self.kernel1(X, x,) +  self.kernel2(X, x)
        
        return kernel


class ProdKernel(Kernel):
    
    def __init__(self, kernel1, weight1=1.0, kernel2=None, weight2=1.0) -> None:
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.weight1 = weight1
        self.weight2 = weight2
    
    def __call__(self, X, x=None):
        kernel = self.weight1 * self.kernel1(X, x,) 
        if self.kernel2 is not None:
            kernel *= (self.weight2 * self.kernel2(X, x))
        # print('Prod  ', X.size(), kernel.size())
        
        return kernel
    


class ConstantKernel(Kernel):
    
    def __init__(self, sigma):
        self.sigma = sigma if torch.is_tensor(sigma) else torch.as_tensor(sigma)
    
    def __call__(self, X, x=None):
        if x is None:
            x = X
        # print('Cosntant ', X.size(), )
        sigma = torch.zeros((x.size(0), X.size(0))).type_as(X).fill_diagonal_(torch.exp(self.sigma).item())
        # print("Constamt sigma: ", sigma)
        return sigma.detach()


class LinearKernel(Kernel):

    def __init__(self, sigma, bias=None) -> None:
        self.sigma = sigma if torch.is_tensor(sigma) else torch.as_tensor(sigma)
        self.bias = bias
        if bias is not None:
            self.bias = bias if torch.is_tensor(bias) else torch.as_tensor(bias)
    
    def __call__(self,  X, x=None):
        if x is None:
            x = X
        lin = torch.exp(2 * self.sigma) * (x * X.T) 
        if self.bias is not None:
            assert lin.size(0) == lin.size(1) or lin.size(0) == 1
            bias = torch.diag(self.bias.repeat(lin.size(0)))
            mask_positions = torch.arange(0, lin.size(0)).view(1, -1)
            mask = mask_positions != torch.arange(0, lin.size(0)).view(-1, 1)
            bias[mask] = -65535.
            # print(bias)
            # print(torch.exp(bias))
            lin = lin + torch.exp(bias)
        # print('Linear ', X.size(), x.size(), lin.size())
        # print(lin)
        
        return lin
        

class CosineKernel(Kernel):
    
    def __init__(self, p=1., scale=1.0) -> None:
        self.p = p if torch.is_tensor(p) else torch.as_tensor(p)
        self.scale = scale if torch.is_tensor(scale) else torch.as_tensor(scale)
                
    def __call__(self, X, x=None):
        if x is None:
            x = X
        p = torch.exp(self.p)
        assert len(X.shape) == len(x.shape) == 2
        norm =  torch.abs(x[:, None, ...] - X[None, ...]).sum(axis=-1)
        r = torch.cos( 2 * torch.pi * norm / p)
        # print('Coised ', X.size(), x.size(), ( torch.exp(2 * self.scale) * r).size())
        
        return torch.exp(2 * self.scale) * r
        

class ExpSquaredKernel(Kernel):
    
    def __init__(self, length=1.0, scale=1.0) -> None:
        super().__init__()
        self.scale = scale if torch.is_tensor(scale) else torch.as_tensor(scale)
        self.length = length if  torch.is_tensor(length) else torch.as_tensor(length)
    
    def __call__(self, X, x=None):
        if x is None:
            x = X
        length = torch.exp( 2 * self.length)
        scale = torch.exp(self.scale)
        assert len(X.shape) == len(x.shape) == 2
        norm = torch.square(x[:, None, ...] - X[None, ...]).sum(axis=-1)
        assert list(norm.size()) == list([x.shape[0], X.shape[0]])
        r = torch.exp(-norm / (2 * length))
        # print('ExpSaured ', X.size(), x.size(), (scale * r).size())
        # print("Exp kernel ", scale * r)
        return scale * r
        


class GaussianProcess:
    
    def __init__(self, kernel, X, diag=None, mean=None, mean_function=None):
        self.kernel = kernel
        self.X = X
        self.diag = diag
        self.mean = mean
        self.mean_function = mean_function
    @staticmethod
    def _compute_kernel_fn( X1, X2, kernel_fn,):
        kernel = kernel_fn(X1, X2) 
        return kernel
        

    def compute_kernel(self, X1, X2, mode='inference'):
        kernel = self._compute_kernel_fn(X1, X2, self.kernel)
        # print('Compyte kernel ', kernel)
        if self.diag is not None and mode != 'inference':
            diag = torch.diag(self.diag.repeat(kernel.size(0)))
            mask_positions = torch.arange(0, kernel.size(0)).view(1, -1)
            mask = mask_positions != torch.arange(0, kernel.size(0)).view(-1, 1)
            diag[mask] = -65535.
            diag = torch.exp(self.diag)
            kernel = kernel + diag
        return kernel
    
    def sample(self, n=1):
        kernel  = self.compute_kernel(self.X, self.X) 
        chol = torch.linalg.cholesky(kernel)
        epsilon = torch.randn((self.X.clone().detach().size(0), n), dtype=kernel.dtype)
        samples = chol @ epsilon
        # print("Samples: ", n)
        if self.mean is not None:
            samples = self.mean + samples
        
        if self.mean_function is not None:
            # print(self.X.size(), samples.size())
            # print(self.X.dtype, samples.dtype)
            samples = self.mean_function(self.X) + samples            
             
        return samples
    
    def _log_probability(self, y):
        # Instable
        N = y.size(0)
        kernel = self.compute_kernel(self.X, self.X) 
        y = y.ravel()
        mean = self.mean.ravel()
        mahalanobis_dst = -0.5 * (y - mean)[None, :] @ torch.linalg.inv(kernel) @ (y - mean)
        det = torch.linalg.det(kernel.to(torch.float64))
        return mahalanobis_dst - 0.5 * torch.log(det.float() + 1e-8) - (N / 2) * torch.log(torch.tensor(2*torch.pi))
    
    def log_probability(self, y, mean):
        # Use cholesky
        N = y.size(0)
        kernel = self.compute_kernel(self.X, self.X) 
        alpha = self.compute_alpha(y, mean, kernel)
        chol = torch.linalg.cholesky(kernel)
        mahalanobis_dst = torch.sum(-0.5 * (y - mean) * alpha, axis=0)
        det = chol.diagonal().sum()
        return mahalanobis_dst - 0.5 * torch.log(det.float() ) - (N / 2) * torch.log(torch.tensor(2*torch.pi))
    
    def compute_alpha(self, y, mean, kernel):
        x = y - mean
        x = x.view(y.shape[0], -1)
        chol = torch.linalg.cholesky(kernel)
        # kernel = L * L.transpose
        # kernel.inverse = L.transpose.inverse * L.inverse
        # L.transpose.inverse * L.inverse * x = alpha
        #  L.inverse * x = x1
        # x = L * x1
        # L.transpose * alpha = x1
        
        x1 = torch.linalg.solve_triangular(chol.unsqueeze(0), x.unsqueeze(0), upper=False, left=True)
        
        alpha = torch.linalg.solve_triangular(chol.T.unsqueeze(0), x1, upper=True, left=True)
        return alpha.squeeze()
        