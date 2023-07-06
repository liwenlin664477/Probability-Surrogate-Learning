import numpy as np
import string, random, os, time
from scipy import interpolate
import subprocess

from hdf5storage import loadmat
import sympy as sy

from infras.randutils import *
from infras.misc import *

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def run_command(cmd):
    
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')

    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)
        
        
class Burgers_exact():
    def __init__(self):
        # 1D Burgers
        x, t, u, nu = sy.symbols('x, t, u, nu')
        m = 2*u*sy.sqrt(nu*t) # u substitution
        dmdu_conversion = 2*sy.sqrt(nu*t)
        integral_1 = sy.sin(sy.pi*(x-m))*sy.exp(-sy.cos(sy.pi*(x-m))/(2*sy.pi*nu))*dmdu_conversion
        integral_2 = sy.exp(-sy.cos(sy.pi*(x-m))/(2*sy.pi*nu))*dmdu_conversion
        # Hermite quadrature
        n = 100
        u_vals, w = np.polynomial.hermite.hermgauss(n)
        integral_1_sum = 0
        integral_2_sum = 0
        integral_total = 0
        for i in range(n):
            integral_1_sum += integral_1.subs(u,u_vals[i])*w[i]
            integral_2_sum += integral_2.subs(u,u_vals[i])*w[i]
        self.burgers_1_u = sy.lambdify([x, t, nu],integral_1_sum,'numpy')
        self.burgers_2_u = sy.lambdify([x, t, nu],integral_2_sum,'numpy')
        
    def burgers_sample(self, meshsize, nu_val):
        x_val = np.linspace(-1,1, meshsize)
        t_val = np.linspace(0,1, meshsize)
        u_vals = np.zeros((x_val.size,t_val.size))
        for i in range(t_val.size): 
            if i == 0: # Take t = 0 as the initial condtions otherwise you divide by 0
                u_vals[:,i] = -np.sin(x_val*np.pi) # u(x, 0) = -sin(pi*x). 
            else:
                u_vals[:,i] = -(self.burgers_1_u(x_val,t_val[i],nu_val))/(self.burgers_2_u(x_val,t_val[i],nu_val))
        return u_vals.T, x_val, t_val
    
class Burgers:
    
    def __init__(self,):

        self.dim = 1
        self.bounds = ((0.001,0.05))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
        
        self.pde = Burgers_exact()
        
    def _solve_one_inst(self, X, fidelity):
        
        nu = np.squeeze(X)
        uvals, _, _ = self.pde.burgers_sample(meshsize=fidelity, nu_val=nu)
        
        #print(u.shape)
            
        return uvals

    def solve(self, X, fidelity):
        assert X.ndim == 2
        y_tensors = []
        N = X.shape[0]
        for n in range(N):
            y = self._solve_one_inst(X[n,:], fidelity)
            y_tensors.append(np.expand_dims(y, 0))
        #
        
        Y = np.concatenate(y_tensors)
        
        return Y


class Poisson:
    def __init__(self,):

        self.dim = 5
        self.bounds = ((0.1, 0.9),(0.1, 0.9),(0.1, 0.9),(0.1, 0.9),(0.1, 0.9))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def _solver(self, fidelity, u_0_x, u_1_x, u_y_0, u_y_1, u_dirac):
        x = np.linspace(0,1,fidelity)
        dx = x[1]-x[0]
        y = np.linspace(0,1,fidelity)
        u = np.zeros((fidelity+2,fidelity+2)) # Initial u used to create the b-vector in Ax = b
        # BC's and dirac delta
        u[0,:] = u_0_x
        u[-1,:] = u_1_x
        u[:,0] = u_y_0
        u[:,-1] = u_y_1
        if fidelity%2 == 0:
            u[int((fidelity+2)/2-1):int((fidelity+2)/2+1),int((fidelity+2)/2-1):int((fidelity+2)/2)+1] =\
            u_dirac
        else:
            u[int((fidelity+1)/2),int((fidelity+1)/2)] = u_dirac

        # 5-point scheme
        A = np.zeros((fidelity**2,fidelity**2))
        for i in range(fidelity**2):
            A[i,i] = 4
            if i < fidelity**2-1:
                if i%fidelity != fidelity-1:
                    A[i,i+1] = -1
                if i%fidelity != 0 & i-1 >= 0:
                    A[i,i-1] = -1
            if i < fidelity**2-fidelity:
                A[i,i+fidelity] = -1
                if i-fidelity >= 0:
                    A[i,i-fidelity] = -1

        # Boundry conditions
        g = np.zeros((fidelity,fidelity))
        for i in range(1,fidelity+1):
            for j in range(1,fidelity+1):
                g[i-1,j-1] = u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]

        b = dx**2*g.flatten()
        
        # Sparse solver
        A_s = csc_matrix(A, dtype=float) # s for sparse
        b_s = csc_matrix(b, dtype=float)
        x_s = spsolve(A_s,b_s.T)
        u_s = x_s.reshape(fidelity,fidelity)

        return u_s

    def _solve_one_inst(self, X, fidelity):
        
        X = np.squeeze(X)
        u = self._solver(fidelity, X[0], X[1], X[2], X[3], X[4])
            
        return u

    def solve(self, X, fidelity):
        y_tensors = []
        N = X.shape[0]
        for n in range(N):
            y = self._solve_one_inst(X[n], fidelity)
            y_tensors.append(y)
        #

        Y = np.array(y_tensors)
        
        return Y
    
    
class Heat:
    
    def __init__(self,):

        self.dim = 3
        self.bounds = ((0,1),(-1,0),(0.01, 0.1))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def _thomas_alg(self, a, b, c, d):
        n = len(b)
        x = np.zeros(n)
        for k in range(1,n):
            q = a[k]/b[k-1]
            b[k] = b[k] - c[k-1]*q
            d[k] = d[k] - d[k-1]*q
        q = d[n-1]/b[n-1]
        x[n-1] = q
        for k in range(n-2,-1,-1):
            q = (d[k]-c[k]*q)/b[k]
            x[k] = q
        return x
    
    def _solver(self, fidelity,alpha,neumann_0,neumann_1):
        x = np.linspace(0,1,fidelity)
        t = np.linspace(0,1,fidelity)
        u = np.zeros((fidelity+1,fidelity+2))
        dx = x[1]-x[0]
        dt = t[1]-t[0]

        # Set heaviside IC
        for i in range(fidelity):
            if i*dx >= 0.25 and i*dx <= 0.75:
                u[0,i+1] = 1

        for n in range(0,fidelity): # temporal loop
            a = np.zeros(fidelity); b = np.zeros(fidelity); c = np.zeros(fidelity); d = np.zeros(fidelity)
            for i in range(1,fidelity+1): # spatial loop
                # Create vectors for a, b, c, d
                a[i-1] = -alpha*dt/dx**2
                b[i-1] = 1+2*alpha*dt/dx**2
                c[i-1] = -alpha*dt/dx**2
                d[i-1] = u[n,i]

            # Neumann coniditions 
            d[0] = (d[0] - ((alpha*dt/dx**2)*2*dx*neumann_0))/2 # Divide by 2 to keep symmetry
            d[-1] = (d[-1] + ((alpha*dt/dx**2)*2*dx*neumann_1))/2
            a[0] = 0
            b[0] = b[0]/2
            c[-1] = 0
            b[-1] = b[-1]/2

            # Solve
            u[n+1,1:-1] = self._thomas_alg(a,b,c,d)
        v = u[1:,1:-1]
        return v, x
        
    def _solve_one_inst(self, X, fidelity):

        X = np.squeeze(X)
        u, x = self._solver(fidelity, X[2], X[0], X[1])
            
        return u

    def solve(self, X, fidelity):
        y_tensors = []
        N = X.shape[0]
        for n in range(N):
            y = self._solve_one_inst(X[n], fidelity)
            y_tensors.append(y)
        #
        
        Y = np.array(y_tensors)

        return Y
    
    
class TopOpt:
    
    def __init__(self,):

        self.dim = 2
        self.bounds = ((0.1,0.9),(0.1,0.9))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def query(self, X, fidelity):
        
        query_key = generate_random_string(16)

        buff_path = os.path.join('data/__buff__', query_key)
        create_path(buff_path, verbose=False)
        
        mat_file_name = os.path.join(buff_path, 'outputs.mat')
        #print(mat_file_name)

        mat_input = '['
        for i in range(X.shape[0]):
            s = np.array2string(X[i,:], separator=',',floatmode='maxprec',max_line_width=10000000)
            mat_input += s
            if i < X.shape[0] - 1:
                mat_input += ';'
            #
        #
        mat_input += ']'

        
        matlab_cmd = 'addpath(genpath(\'data/matlab_solvers/Topology-Optimization\'));'
        matlab_cmd += 'query_client_topopt(' + mat_input  + ',' + str(fidelity) + ', \'' + mat_file_name + '\'' + ');'
        matlab_cmd += 'quit force'
        
        #print(matlab_cmd)
   
        process = subprocess.Popen(["matlab", "-nodesktop", "-r", matlab_cmd],
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        process.wait()
        
        data = loadmat(mat_file_name, squeeze_me=True, struct_as_record=False, mat_dtype=True)['data']

        Y = data.Y
        
        if Y.ndim == 2:
            Y = np.expand_dims(Y, 2)
            
        tr_Y = np.transpose(Y, [2,0,1])
        
        return tr_Y


    def solve(self, X, fidelity):
        
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        #
        
        Y = self.query(X, fidelity)
        
        return Y
    
    
class NavierStockPRec:
    
    def __init__(self,):
        

        self.dim = 1
        c_lb=0.2
        c_ub=0.8
        self.bounds = ((c_lb,c_ub),(c_lb,c_ub),(c_lb,c_ub),(c_lb,c_ub),(c_lb,c_ub))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def query(self, X, fidelity):
        
        query_key = generate_random_string(16)

        buff_path = os.path.join('data/__buff__', query_key)
        create_path(buff_path, verbose=False)
        
        mat_file_name = os.path.join(buff_path, 'outputs.mat')
        #print(mat_file_name)

        mat_input = '['
        for i in range(X.shape[0]):
            s = np.array2string(X[i,:], separator=',',floatmode='maxprec',max_line_width=10000000)
            mat_input += s
            if i < X.shape[0] - 1:
                mat_input += ';'
            #
        #
        mat_input += ']'

        
        matlab_cmd = 'addpath(genpath(\'data/matlab_solvers/NS-Solver\'));'
        matlab_cmd += 'query_client_ns(' + mat_input  + ',' + str(fidelity+1) + ', \'' + mat_file_name + '\'' + ');'
        matlab_cmd += 'quit force'

        #print(matlab_cmd)
        
        process = subprocess.Popen(["matlab", "-nodesktop", "-r", matlab_cmd],
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        process.wait()
        
        Y = loadmat(mat_file_name, squeeze_me=True, struct_as_record=False, mat_dtype=True)['PRec']
        Xfluid = loadmat(mat_file_name, squeeze_me=True, struct_as_record=False, mat_dtype=True)['Xfluid']
        
        if Y.ndim == 3:
            Y = np.expand_dims(Y, 3)

        tr_Y = np.transpose(Y, [3,2,0,1])
        
        return tr_Y, Xfluid


    def solve(self, X, fidelity):
        
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        #
        
        Y, Xfluid = self.query(X, fidelity)
        
        return Y, Xfluid


class NavierStockURec:
    
    def __init__(self,):

        self.dim = 1
        c_lb=0.2
        c_ub=0.8
        self.bounds = ((c_lb,c_ub),(c_lb,c_ub),(c_lb,c_ub),(c_lb,c_ub),(c_lb,c_ub))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def query(self, X, fidelity):
        
        query_key = generate_random_string(16)

        buff_path = os.path.join('data/__buff__', query_key)
        create_path(buff_path, verbose=False)
        
        mat_file_name = os.path.join(buff_path, 'outputs.mat')
        #print(mat_file_name)

        mat_input = '['
        for i in range(X.shape[0]):
            s = np.array2string(X[i,:], separator=',',floatmode='maxprec',max_line_width=10000000)
            mat_input += s
            if i < X.shape[0] - 1:
                mat_input += ';'
            #
        #
        mat_input += ']'

        
        matlab_cmd = 'addpath(genpath(\'data/matlab_solvers/NS-Solver\'));'
        matlab_cmd += 'query_client_ns(' + mat_input  + ',' + str(fidelity) + ', \'' + mat_file_name + '\'' + ');'
        matlab_cmd += 'quit force'

        #print(matlab_cmd)
        
        process = subprocess.Popen(["matlab", "-nodesktop", "-r", matlab_cmd],
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        process.wait()
        
        Y = loadmat(mat_file_name, squeeze_me=True, struct_as_record=False, mat_dtype=True)['URec']
        Xfluid = loadmat(mat_file_name, squeeze_me=True, struct_as_record=False, mat_dtype=True)['Xfluid']
        cprint('y', X)
        cprint('y', Xfluid)
        
        if Y.ndim == 3:
            Y = np.expand_dims(Y, 3)

        tr_Y = np.transpose(Y, [3,2,0,1])
        
        
        return tr_Y, Xfluid

    def solve(self, X, fidelity):
        
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        #
        
        Y, Xfluid = self.query(X, fidelity)
        
        return Y, Xfluid
    
class NavierStockVRec:
    
    def __init__(self,):

        self.dim = 1
        c_lb=0.2
        c_ub=0.8
        self.bounds = ((c_lb,c_ub),(c_lb,c_ub),(c_lb,c_ub),(c_lb,c_ub),(c_lb,c_ub))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
        
    def query(self, X, fidelity):
        
        query_key = generate_random_string(16)

        buff_path = os.path.join('data/__buff__', query_key)
        create_path(buff_path, verbose=False)
        
        mat_file_name = os.path.join(buff_path, 'outputs.mat')
        #print(mat_file_name)

        mat_input = '['
        for i in range(X.shape[0]):
            s = np.array2string(X[i,:], separator=',',floatmode='maxprec',max_line_width=10000000)
            mat_input += s
            if i < X.shape[0] - 1:
                mat_input += ';'
            #
        #
        mat_input += ']'

        
        matlab_cmd = 'addpath(genpath(\'data/matlab_solvers/NS-Solver\'));'
        matlab_cmd += 'query_client_ns(' + mat_input  + ',' + str(fidelity) + ', \'' + mat_file_name + '\'' + ');'
        matlab_cmd += 'quit force'
        
        process = subprocess.Popen(["matlab", "-nodesktop", "-r", matlab_cmd],
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        process.wait()
        
        Y = loadmat(mat_file_name, squeeze_me=True, struct_as_record=False, mat_dtype=True)['VRec']
        Xfluid = loadmat(mat_file_name, squeeze_me=True, struct_as_record=False, mat_dtype=True)['Xfluid']
        cprint('y', X)
        cprint('y', Xfluid)
        
        if Y.ndim == 3:
            Y = np.expand_dims(Y, 3)

        tr_Y = np.transpose(Y, [3,2,0,1])
        
        return tr_Y, Xfluid


    def solve(self, X, fidelity):
        
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        #
        
        Y, Xfluid = self.query(X, fidelity)
        
        return Y, Xfluid
    


    
    