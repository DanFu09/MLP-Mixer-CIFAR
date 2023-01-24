###Standalone code for the monarch conv###

import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from einops import rearrange, repeat
import math
import opt_einsum as oe
contract = oe.contract

def Activation(activation=None, size=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.Softplus()
    elif activation == 'modrelu':
        return Modrelu(size)
    elif activation in ['sqrelu', 'relu2']:
        return SquaredReLU()
    elif activation == 'laplace':
        return Laplace()
    elif activation == 'ln':
        return TransposedLN(dim)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))
####General Module to register parameters######################################

class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


###Learnable monarch FFT########################################################
'''PyTorch version of the block FFT convolution as described in the H3 paper.'''


def ref_dft_matrix(N, H=1):
    """Compute the DFT matrix of size N x N.
    
    This is where we could add extra compute for free."""
    # n = torch.arange(N)
    n = torch.arange(N).cuda()
    k = n.view(-1, 1)
    M = torch.exp(-2j * torch.pi * n * k / N)
    return torch.view_as_real(M.repeat(H, 1, 1))

def compute_twiddle_factors(n, m):
    """Compute the twiddle factors of size n x m"""
    # n_a = torch.arange(n).view(-1, 1)
    # m_a = torch.arange(m)
    n_a = torch.arange(n).cuda().view(-1, 1)
    m_a = torch.arange(m).cuda()
    N = n * m
    M = torch.exp(-2j * torch.pi * n_a * m_a / N)
    return torch.view_as_real(M)

def _cooley_tukey(
    k, n, m, 
    dft_matrix=ref_dft_matrix,
    max_m=16,
    activation=None,
):
    '''
    Compute the FFT using the general Cooley-Tukey algorithm:
        * Reshape to (m, n)
        * Do n m-length FFTs along the rows
        * Transpose to (n, m), multiply by twiddle factors
        * Do m n-length FFTs along the rows

    This function assumes that m <= 16 and recurses on n.
    The base case is n <= 16 (we are simulating tensor cores of 16x16 mm).
    The dft_matrix function is overwriteable
    so that we can replace it with learnable parameters in a model.
    '''
    assert m <= max_m

    if activation is not None:
        act_fn = Activation(activation)

    k = rearrange(k, '... (m n) -> ... m n', m=m, n=n) # (m, n)

    # do n m-length FFTs
    if activation is None:
        mat = torch.view_as_complex(dft_matrix(m))
        k_f = torch.einsum('... m o, ... o n -> ... m n', mat, k) # (..., m, n)
    else:
        mat = torch.view_as_complex(dft_matrix(m))
        k_f = torch.view_as_complex(act_fn(
            torch.view_as_real(torch.einsum('... m o, ... o n -> ... m n', mat, k))
        )) # (..., m, n)

    # multiply by twiddle factors
    twi = torch.view_as_complex(compute_twiddle_factors(n, m)) # (n, m)
    k_f = torch.einsum('n m, ... m n -> ... n m', twi, k_f) # (..., n, m)

    if n <= max_m:
        # do m n-length FFTs
        if activation is None:
            mat = torch.view_as_complex(dft_matrix(n))
            k_f = torch.einsum('... n o, ... o m -> ... n m', mat, k_f) # (.., n, m)
        else:
            mat = torch.view_as_complex(dft_matrix(n))
            k_f = torch.view_as_complex(act_fn(
                torch.view_as_real(torch.einsum('... n o, ... o m -> ... n m', mat, k_f))
            )) # (.., n, m)
    else:
        # recurse
        k_f = rearrange(k_f, '... h n m -> ... m h n')
        k_f = _cooley_tukey(k_f, n // max_m, max_m, dft_matrix, max_m, activation)
        k_f = rearrange(k_f, '... m h n -> ... h n m')

    # reshape for the output
    k_f = rearrange(k_f, '... n m -> ... (n m)') # (..., n*m)

    return k_f

def block_fft(
    k, N,
    dft_matrix=ref_dft_matrix,
    max_m=16,
    **kwargs,
):
    '''
    Compute the FFT of size N of the vector k, using _block_fft_recurse.
    
    The dft_matrix function is overwriteable
    so that we can replace it with learnable parameters in a model.
    '''
    if not math.log(N, 2).is_integer():
        N = int(2 ** math.ceil(math.log(N, 2)))
    # pad k with zeros if necessary (e.g. for causality)
    if k.shape[-1] != N:
        k = nn.ConstantPad1d((0, N - k.shape[-1]), 0)(k)
    
    if N <= max_m:
        mat = torch.view_as_complex(dft_matrix(m))
        return torch.einsum('... n o, ... o -> ... n', mat, k) # (.., n, m)
    n = N // max_m
    m = max_m
    return _cooley_tukey(k, n, m, dft_matrix, max_m, **kwargs)


class BlockFFT(OptimModule):
    '''
    Learnable Block FFT module.

    Args:
        learn_dft_matrix (bool): If True, learn a different DFT matrix for lengths 2, 4, 8, and 16. If False, this module computes a normal FFT.
    '''
    def __init__(self, learn_dft_matrices=True, H=1, max_m=16, dft_lr=0.001, dropout=0, learn_additive=False, **block_fft_args):
        super().__init__()
        self.learn_dft_matrices = learn_dft_matrices
        self.block_fft_args = block_fft_args
        self.max_m=max_m
        self.drop = torch.nn.Dropout(p=dropout)
        self.learn_additive=learn_additive
        # get the powers of 2 up to max_m
        assert math.log(max_m, 2).is_integer(), 'max_m must be a power of 2'

        self.powers = [ 2 ** (i + 1) for i in range(int(math.log(max_m, 2))) ]

        if learn_dft_matrices:
            assert dft_lr>0,"If learn_dft_matrices=True dft_lr must be positive"
            self.dft_matrices = nn.ParameterList()
            for n in self.powers:
                setattr(self,f"mat_{n}",nn.Parameter(
                    0.01 * torch.randn(H, n, n, 2) if self.learn_additive
                    else ref_dft_matrix(n, H=H),
                    requires_grad=True))
                self.register(f"mat_{n}",getattr(self,f"mat_{n}"),dft_lr)
                self.dft_matrices.append(getattr(self,"mat_{}".format(n)))

    def compute_dft_matrix(self, n):
        if not self.learn_dft_matrices:
            return ref_dft_matrix(n)
        else:
            assert n in self.powers
            if self.learn_additive:
                mat = ref_dft_matrix(n)
                return mat + self.drop(self.dft_matrices[int(math.log(n, 2) - 1)])
            else:
                return self.drop(self.dft_matrices[int(math.log(n, 2) - 1)])

    def forward(self, x, N,forward=True):
        '''Compute an FFT (forward=True) or iFFT (forward=False) of length N over x.'''
        if forward:
            return block_fft(x, N, dft_matrix=self.compute_dft_matrix, **self.block_fft_args)
        else:
            return (1/(N))*torch.conj(block_fft(torch.conj(x), N, dft_matrix=self.compute_dft_matrix, **self.block_fft_args))


####LONG CONV #########################################################################

def get_kernel(base,scale, L,d):
    repeats = L // d
    if repeats * d != L:
        repeats += 1
    L_scale = repeats * d
    extended_base = einops.repeat(base,"c H d -> c H (repeat d)",repeat=repeats)             
    Kernel = torch.einsum("c H L, L H-> c H L",extended_base,scale[:L_scale, ...])
    return Kernel[..., :L]


class LongConvKernel(OptimModule):
    def __init__(
        self, 
        H, 
        N, 
        L,
        channels=1, 
        learning_rate=None, 
        lam=0.1, 
        device='cuda', 
        dtype=torch.float,
        causal=True, 
        kernel_dropout=0,
        weight_init="random",
        sequence_d=64,
        hidden_state_d=1, 
        scale_factor=1,
        use_prox=False,
        use_prox_lam = 0.001,
        **kwargs
    ):
        super().__init__()
       
        self.drop = torch.nn.Dropout(p=kernel_dropout)
        self.sg = kwargs.get("sg_cfg",None)
        self.H = H
        self.weight_init = weight_init
        self.causal = causal
        self.scale=None
        self.sequence_d = sequence_d
        self.hidden_state_d = hidden_state_d
        if self.hidden_state_d > 1:
            assert(self.weight_init == "const_recursive")
        if not causal:
            self.N = L*2 # CHANGE THIS LATER!!
            self.L = L*2
        else:
            self.N,self.L=L,L
        
        if use_prox:
            self.use_prox = nn.Identity()
            self.use_prox.lam = use_prox_lam

        self.channels = channels
        self.lam = lam
        self.device = device
        self.kernel_params = torch.nn.Parameter(self._parameter_initialization()) #(c,H,L) 
        # self.zeros = torch.zeros(self.channels, self.H, self.L).to(self.device)
        if weight_init == "const_recursive" or weight_init=="const_recursive_vnorm":
            #const_recursive is the "a" parameters in the constant recursion -> rename recursion_params-  
            self.recursion_params = torch.nn.Parameter(torch.randn(self.channels, self.H, self.hidden_state_d, self.sequence_d) * 0.002) # .to(self.device)
            if weight_init=="const_recursive_vnorm":
                self.scale_factor = scale_factor
                assert scale_factor<10, "Too large scale factor, this leads to unstable kernels"
                self.vnorm = torch.tensor([self.scale_factor*10*self.sequence_d*1/pow(H,i/H) for i in range(H)],dtype=torch.float32) # .to(self.device)


        self.register("kernel_params", self.kernel_params, learning_rate) 
    
    def _get_scale(self,L,d,H):
        # assert L%d==0, "The self similar kernel length must divide the sequence length"
        if L % d == 0:
            L_scale = L
        else:
            L_scale = (L // d + 1) * d
        phase= torch.FloatTensor(H).uniform_(0.7,0.71)
        decay=torch.tensor([1/(H)*torch.pow(torch.tensor(H),torch.tensor(i/H)) for i in range(H)])
        alpha = torch.pow(torch.exp(torch.tensor(1)),phase*1j)*torch.pow(torch.exp(torch.tensor(1)),-decay)
        scale = torch.ones((L,H),dtype=torch.cfloat)
        for i in range(d,L_scale,d):
            scale[i:i+d,:] =torch.einsum("H, L H -> L H ",alpha,scale[i-d:i,:])
        self.scale = scale.real.cuda()

    def _parameter_initialization(self):
        if self.weight_init=="random":
            return torch.randn(self.channels, self.H, self.N) * 0.002
        elif self.weight_init=="hippo":
            from src.models.sequence.ss.kernel import SSKernel
            assert self.causal==True, "The model needs to be causal to use hippo init"
            config = {"H":self.H,"L":self.L,"mode":"nplr","measure":"legs","lr":0.1,"channels":self.channels}
            kernel = SSKernel(**config)
            kernel = (kernel.kernel(L=self.L)[0]).detach().numpy()
            return torch.tensor(kernel,dtype=torch.float32).cuda()
        elif self.weight_init=="double_exp":
            K = torch.randn(self.channels, self.H, self.L,dtype=torch.float32) * 0.02
            double_exp = torch.zeros((self.H,self.L),dtype=torch.float32)
            for i in range(self.H):
                for j in range(self.L):
                    double_exp[i,j] = torch.exp(-(j/self.L)*torch.pow(torch.tensor(int(self.H/2)),torch.tensor(i/self.H)))
            K = torch.einsum("c h l, h l -> c h l",K,double_exp)
            return torch.tensor(K,dtype=torch.float32).cuda()
        elif self.weight_init=="sg":
            from src.models.sequence.ss.gconv_standalone import GConv
            assert self.sg!=None
            model = GConv(self.H,mode = self.sg.mode,channels=self.channels, l_max=self.L,**self.sg.args)
            u = torch.randn((1,self.H,self.L))
            y,k=model(u,return_kernel=True)
            return torch.tensor(k,dtype=torch.float32).cuda()
        elif self.weight_init=="self_similar":
            C = torch.randn(self.channels, self.H, self.sequence_d,dtype=torch.float32) * 0.02
            return torch.tensor(C,dtype=torch.float32).cuda()
        elif self.weight_init=="const_recursive" or self.weight_init=="const_recursive_vnorm":
            C = torch.randn(self.channels, self.H, self.hidden_state_d, self.sequence_d,dtype=torch.float32) * 0.02
            return torch.tensor(C,dtype=torch.float32).cuda()
        else: raise NotImplementedError(f"{self.weight_init} is not valid") 
    
    def _get_kernel(self,L,**kwargs):
        if self.weight_init=="self_similar":
            if self.scale==None:
                self._get_scale(self.L,self.sequence_d,self.H)
            kernel = get_kernel(self.kernel_params,self.scale,L,self.sequence_d)
        elif self.weight_init=="const_recursive" or self.weight_init=="const_recursive_vnorm": 
            d = self.sequence_d
            repeats = L // d
            if repeats * d != L:
                repeats += 1
            kernel = einops.repeat(self.kernel_params, "c H hd d -> c H hd (repeat d)", repeat = repeats)
            a = (self.recursion_params / torch.sum(torch.abs(self.recursion_params), dim=-1, keepdim=True))*0.1
            if self.weight_init=="const_recursive_vnorm":
                a = torch.einsum("c H z d, H -> c H z d",a,self.vnorm).to(self.device)

            a_f = torch.fft.rfft(a, n = 4 * d) / (4 * d)

            for i in range(1, repeats):
                if i == 1:
                    k_f = torch.fft.rfft(kernel[..., :d], n = 4 * d)
                    kernel[..., d:2*d] = torch.fft.irfft(k_f * a_f, n = 4 * d)[..., :d]
                else:
                    k_f = torch.fft.rfft(kernel[..., d * (i - 2):d*i], n=4 * d)
                    kernel[..., i*d:(i+1)*d] = torch.fft.irfft(k_f * a_f, n=4 * d)[..., d:2*d]
            kernel = torch.sum(kernel, dim=2)[..., :L]
            
        else:
            kernel= self.kernel_params
        return kernel

    def forward(self, L=None, **kwargs):
        k = self._get_kernel(L,**kwargs)
        prox_kernel = F.relu(torch.abs(k)-self.lam)*torch.sign(k)
        dropped_kernel = self.drop(prox_kernel)
        return dropped_kernel, None

    @property
    def d_output(self):
        return self.H
####Forward pass in BlockCONV

def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False, # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
    ):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """
    # Construct core module
    # linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, d_output, dim=1 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear

####Learnable Block Conv################################################################################
#first arg: Hidden, second arg: sequence dim

class MonarchConv(OptimModule):
    def __init__(self,H,L,channels=1, learn_ifft=True,verbose=False,
                bidirectional=False,dft_lr=0.0001,fft_dropout=0,learn_dft_mat = True,m_max=32,learning_rate=0.001, lam=0.003, 
                device="cuda",causal=True, kernel_dropout=0.2,weight_init="random",sequence_d=128, hidden_state_d=1,
                scale_factor=1, use_prox=False, use_prox_lam=0.001,forward_drop=0):
        super().__init__()
        self.H =H
        self.L = L
        self.channels = channels
        self.learn_ifft = learn_ifft
        self.bidirectional = bidirectional
        self.lam = lam
        self.m_max = m_max
        self.learning_rate = learning_rate
        if self.bidirectional:
            self.channels*=2
        self.block_fft_conv_args = {"dft_lr":dft_lr,"dropout":fft_dropout,"learn_dft_matrices": learn_dft_mat, "max_m":m_max,"learn_additive":True}
        self.kernel_args ={"channels":self.channels,"learning_rate": learning_rate,"lam":lam,"device":device,
                    "dtype":torch.float,"causal": causal,"kernel_dropout":kernel_dropout,"weight_init": weight_init,
                    "sequence_d":sequence_d,"hidden_state_d":hidden_state_d, "scale_factor": scale_factor, "use_prox":use_prox,"use_prox_lam":use_prox_lam}
        
        # SSM Kernel
        self.forward_drop = nn.Dropout(p=forward_drop)
        self.D = nn.Parameter(torch.randn(channels, self.H)).cuda()
        self.activation = nn.GELU()
        self.kernel = LongConvKernel(self.H, N=0, L=self.L, **self.kernel_args)
        
        self.block_fft_u = BlockFFT(**self.block_fft_conv_args)
        self.block_fft_k = BlockFFT(**self.block_fft_conv_args)
        self.output_linear = LinearActivation(
                self.H * self.channels,
                self.H,
                transposed=False,
                initializer=None,
                activation="glu",
                activate=True,
                weight_norm=False,
            ).cuda()

    
    def forward(self, u, **kwargs):
        _kernel =  self.kernel()
        if len(_kernel) == 2:
            k, k_state = _kernel
            k_bias = None
        else:
            k, k_state, k_bias = _kernel

        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, self.L)) \
                    + F.pad(k1.flip(-1), (self.L, 0))

            if k_bias is not None:
                k_bias_0, k_bias_1 = rearrange(k_bias, '(s c) h l -> s c h l', s=2)
                k_bias = k_bias_0 + k_bias_1.flip(-1)
        k = k.cuda()
        u = u.cuda()

        #generalized block conv
        k_f = self.block_fft_k(k.to(torch.complex64), N=2*self.L) # (C H L)
        u_f = self.block_fft_u(u.to(torch.complex64), N=2*self.L) # (B H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f)
        if self.learn_ifft:
            y = self.block_fft_u(y_f, N=2*self.L,forward=False).real[..., :self.L]
        else:
            y = torch.fft.ifft(y_f, n=2*self.L, dim=-1).real[..., :self.L] # (B C H L)
        
        y = rearrange(y, '... c h l -> ... (c h) l')
        y = self.activation(y)
        y = self.forward_drop(y)

        return y

def main():

    L = 1024
    B = 50
    H =256
    ###Blockconv can be a replacement for either matmul or convolutions
    genLC = MonarchConv(H,L) #Creates a sequence to sequence map: (batch,head,length) ->(batch,head,length)
    
    u = torch.randn(B,H,L)
    y = genLC(u)
    print(y)



if __name__=="__main__":
    main()