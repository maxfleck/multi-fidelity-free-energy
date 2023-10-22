import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from alchemlyb.estimators.base import _EstimatorMixOut
from scipy import integrate
import matplotlib.pyplot as plt

import GPy
import emukit.multi_fidelity
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

class MF():
    
    def __init__(self,X,Y,fidelity,colors=["orange","green","magenta"],
                kernel=GPy.kern.Matern32,n_optimization_restarts=1):
        axis = 0
        self.axis = axis
        if axis==0:
            axis_inv=-1
        else:
            axis_inv=0
        self.axis_inv = axis_inv
        # normalize with highest fidelity, X and Y
        self.X = np.atleast_2d(X).T #- np.min(X)
        self.Y = np.atleast_2d(Y).T #- np.min(Y)
        self.fidelity = fidelity
        
        # norm data
        self.X_normer = np.linalg.norm(self.X,axis=axis)
        self.Y_normer = np.linalg.norm(self.Y,axis=axis)       
        #print("X+Y_normer",self.X_normer,self.Y_normer)
        self.X_norm = self.X/self.X_normer
        self.Y_norm = self.Y/self.Y_normer
        
        # mirror data ... mirror_flag ?
        self.X_min = np.min(self.X,axis=axis)
        self.X_max = np.max(self.X,axis=axis)
        #print("X_max",self.X_max)
        Xd = self.X_max-self.X_min
        self.X_train  = np.concatenate(( np.flip(self.X_norm-Xd,axis=axis), 
                                        self.X_norm, 
                                        np.flip(self.X_norm-Xd,axis=axis)), axis=axis)
        #print("X_train",self.X_train.shape)
        #print("Y_norm",self.Y_norm.shape)
        self.Y_train  = np.concatenate( [np.flip(self.Y_norm,axis=axis),self.Y_norm,np.flip(self.Y_norm,axis=axis)],axis=axis )
        fidelity_train = np.concatenate(( np.flip(self.fidelity,axis=axis), 
                                        self.fidelity, 
                                        np.flip(self.fidelity,axis=axis)), axis=axis)     
        #print("Y_train",self.Y_train.shape)
        self.X_train_F = np.concatenate( [ self.X_train, np.atleast_2d(fidelity_train).T ], axis=axis_inv )
        #print("X_train_F",self.X_train_F.shape)
        
        # initialize MF model
        self.dims = self.X.shape[axis_inv]
        #print("dimensions:",self.dims)
        self.unique_fidelities = np.sort(np.unique(fidelity))
        self.no_fidelities = len(self.unique_fidelities)
        #print("no_fidelities:",self.no_fidelities)
        kernels = []
        for _ in range(self.no_fidelities):
            kernels.append( kernel(self.dims) )
        lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(self.X_train_F, self.Y_train, lin_mf_kernel, 
                                                       n_fidelities=self.no_fidelities)       
        self.lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, self.no_fidelities, 
                                                  n_optimization_restarts=n_optimization_restarts)
        # now you can manipulate hyperparas etc :P
                
        # names for plotting
        self.fidelity_labels=[]
        for fi in self.unique_fidelities:
            self.fidelity_labels.append( "fidelity "+str(fi)+" data" )
        self.fidelity_labels[0] = "low fidelity data"
        self.fidelity_labels[-1] = "high fidelity data"
        # plot opts???
        self.colors=colors
        self.msize=12 
        self.fsize=15
        self.alpha=0.3
        self.lsize = 4        
        
        return
        
    def train(self):
        """
        trains the multi fidelity object.
        
        Adjust and fix hyperparameters and stuff between initialisation and training of the model
        """
        self.lin_mf_model.optimize()    
        return

    def predict(self, x ):
        """
        predicts un-normed high fidelity results for a trained multi fidelity model
        
        x:  np.array
            un-normed input data (values between 0 and 1)
            
        returns mean predictions and corresponding variances as np.arrays
        """        
        X  = x/self.X_normer  
        X  = convert_x_list_to_array([X]*self.no_fidelities)
        X  =  X[ int(len(x)): ]
        hf_mean, hf_var = self.lin_mf_model.predict(X)
        return hf_mean*self.Y_normer, hf_var*self.Y_normer
    
    def predict_all_fidelities(self, x ):
        """
        predicts all un-normed results for a trained multi fidelity model
        
        x:  np.array
            un-normed input data (values between 0 and 1)
            
        returns mean predictions and corresponding variances as np.arrays
        """        
        #X  = np.atleast_2d(x/self.X_normer  )
        X  = x/self.X_normer  
        #print("X pred",X.shape)
        X  = convert_x_list_to_array([X]*self.no_fidelities)
        #print("X pred F",X.shape)
        mean, var = self.lin_mf_model.predict(X)
        return mean*self.Y_normer, var*self.Y_normer
    
    def plot(self,savepath=""):
        n = 100
        msize=12
        fsize=15
        ffsize=20
        alpha=0.3
        lsize = 4        
        
        for i,(mi,ma) in enumerate( zip( self.X_min,self.X_max ) ):
            #print(i,mi,ma)
            helper = np.atleast_2d( np.linspace( mi,ma,n ) ).T
            if i==0:
                dummy = helper
            else:
                dummy = np.concatenate( [dummy, helper ],axis=-1 )
        
        #print("dummy",dummy.shape)
        Y_predict, varY_predict = self.predict_all_fidelities(dummy)        
        #print("Y_predict",Y_predict.shape)
        for j in range(self.dims):
            for i,ii in enumerate(self.unique_fidelities):
                mean = Y_predict[ int(i*n):int((i+1)*n) ,j]
                var  = varY_predict[ int(i*n):int((i+1)*n) ,j]*1.96#*1000
                xx   = dummy[0:n,j]
                #print("mean",mean.shape)
                #print("xx",xx.shape)
                plt.plot(xx, mean,"-",color=self.colors[i],linewidth=lsize)
                plt.fill_between(xx, mean-var, mean+var,alpha=self.alpha,color=self.colors[i])

                p = np.where(self.fidelity == ii)
                plt.plot(self.X[p], self.Y[p],".", label=self.fidelity_labels[i],markersize=msize,color=self.colors[i] )
            plt.ylabel(r"$\langle{\frac{\partial U}{\partial\lambda}}\rangle_{\lambda}$", fontsize=ffsize)
            plt.xlabel( r"$\mathit{\lambda}$", fontsize=ffsize-2)
            plt.legend(frameon=False, fontsize=fsize)
            plt.xlim([np.min(xx),np.max(xx)])
            plt.xticks(fontsize=fsize)
            plt.yticks(fontsize=fsize)
            if savepath:
                plt.savefig(savepath+"MFTI_approx.png", bbox_inches='tight')
                plt.savefig(savepath+"MFTI_approx.pdf", bbox_inches='tight')
            plt.show()
            plt.close()            
        return

class MFTI(BaseEstimator, _EstimatorMixOut):
    """Thermodynamic integration (TI).

    Parameters
    ----------

    verbose : bool, optional
        Set to True if verbose debug output is desired.

    Attributes
    ----------

    delta_f_ : DataFrame
        The estimated dimensionless free energy difference between each state.

    d_delta_f_ : DataFrame
        The estimated statistical uncertainty (one standard deviation) in
        dimensionless free energy differences.

    states_ : list
        Lambda states for which free energy differences were obtained.

    dhdl : DataFrame
        The estimated dhdl of each state.


    .. versionchanged:: 1.0.0
       `delta_f_`, `d_delta_f_`, `states_` are view of the original object.

    """

    def __init__(self, verbose=False, predict_noise=True, n_predict=100,savepath=""):
        self.verbose = verbose
        self.n_predict = n_predict
        self.savepath = savepath
        return
        
    def fit(self, dHdl):
        """
        Compute free energy differences between each state by integrating
        dHdl across lambda values.

        Parameters
        ----------
        dHdl : DataFrame
            dHdl[n,k] is the potential energy gradient with respect to lambda
            for each configuration n and lambda k.

        """
       
        dummy = np.array(dHdl.index.names)
        indexes = dummy[ dummy!="fidelity" ]
        indexes[0] = "fidelity"    
        indexes = list(indexes)
    
        # sort by state so that rows from same state are in contiguous blocks,
        # and adjacent states are next to each other
        dHdl = dHdl.sort_index(level=indexes)

        # get the lambda names
        l_types = np.array(dHdl.index.names[1:])
        l_types = l_types[ l_types != "fidelity" ]
        
        # apply trapezoid rule to obtain DF between each adjacent state
        # NEW: use MF predictions from different sources
        dG_matrix,var_matrix,means,variances = MF_integrator(dHdl,n=self.n_predict,savepath=self.savepath)
        #deltas = np.diff( np.sum(dG_matrix,axis=-1) )
        deltas = np.sum(dG_matrix,axis=-1) 

        # obtain vector of delta lambdas between each state
        # Fix issue #148, where for pandas == 1.3.0
        # dl = means.reset_index()[list(means.index.names[:])].diff().iloc[1:].values
        dl = means.reset_index()[means.index.names[:]].diff().iloc[1:].values        
        
        # build matrix of deltas between each state
        # copied from orig. TI implementation... does whatever it does :P
        ad_delta = np.zeros((len(deltas), len(deltas)))
        for j in range(len(deltas)-1):
            dout = []
            for i in range(len(deltas)-1 - j):

                # Define additional zero lambda
                a = [0.0] * len(l_types)

                # Define dl series' with additional zero lambda on the left and right
                dll = np.insert(dl[i : i + j + 1], 0, [a], axis=0)
                dlr = np.append(dl[i : i + j + 1], [a], axis=0)

                # Get a series of the form: x1, x1 + x2, ..., x(n-1) + x(n), x(n)
                dllr = dll + dlr

                # Append deviation of free energy difference between state i and i+j+1
                dout.append(
                    (dllr**2 * variances.iloc[i : i + j + 2].values / 4)
                    .sum(axis=1)
                    .sum()
                )
            ad_delta += np.diagflat(np.array(dout), k=j + 1)        
        
        # yield standard delta_f_ free energies between each state
        adelta = np.repeat( [deltas], len(deltas), axis=0)      
        self._delta_f_ = pd.DataFrame(
            adelta - adelta.T, columns=means.index.values, index=means.index.values
        )
        self.dhdl = means

        # yield standard deviation d_delta_f_ between each state
        self._d_delta_f_ = pd.DataFrame(
            np.sqrt(ad_delta + ad_delta.T),
            columns=variances.index.values,
            index=variances.index.values,
        )

        self._states_ = means.index.values.tolist()

        self._delta_f_.attrs = dHdl.attrs
        self._d_delta_f_.attrs = dHdl.attrs
        self.dhdl.attrs = dHdl.attrs

        return self
    
    def separate_dhdl(self):
        """
        For transitions with multiple lambda, the attr:`dhdl` would return
        a :class:`~pandas.DataFrame` which gives the dHdl for all the lambda
        states, regardless of whether it is perturbed or not. This function
        creates a list of :class:`pandas.Series` for each lambda, where each
        :class:`pandas.Series` describes the potential energy gradient for the
        lambdas state that is perturbed.

        Returns
        ----------
        dHdl_list : list
            A list of :class:`pandas.Series` such that ``dHdl_list[k]`` is the
            potential energy gradient with respect to lambda for each
            configuration that lambda k is perturbed.
        """
        if len(self.dhdl.index.names) == 1:
            name = self.dhdl.columns[0]
            return [
                self.dhdl[name],
            ]
        dhdl_list = []
        # get the lambda names
        l_types = self.dhdl.index.names
        # obtain bool of changed lambdas between each state
        # Fix issue #148, where for pandas == 1.3.0
        # lambdas = self.dhdl.reset_index()[list(l_types)]
        lambdas = self.dhdl.reset_index()[l_types]
        diff = lambdas.diff().to_numpy(dtype="bool")
        # diff will give the first row as NaN so need to fix that
        diff[0, :] = diff[1, :]
        # Make sure that the start point is set to true as well
        diff[:-1, :] = diff[:-1, :] | diff[1:, :]
        for i in range(len(l_types)):
            if any(diff[:, i]):
                new = self.dhdl.iloc[diff[:, i], i]
                # drop all other index
                for l in l_types:
                    if l != l_types[i]:
                        new = new.reset_index(l, drop=True)
                new.attrs = self.dhdl.attrs
                dhdl_list.append(new)
        return dhdl_list    
    
    
def MF_integrator(dHdl,savepath="",n=100):
    # get index order for sort_index
    dummy = np.array(dHdl.index.names)
    indexes = dummy[ dummy!="fidelity" ]
    indexes[0] = "fidelity"    
    indexes = list(indexes)

    # sort by state so that rows from same state are in contiguous blocks,
    # and adjacent states are next to each other
    dHdl = dHdl.sort_index(level=indexes)

    # obtain the mean and variance of the mean for each state
    # variance calculation assumes no correlation between points
    # used to calculate mean
    means = dHdl.groupby(level=indexes).mean().reset_index()
    # delete lambda paths not used i.e. with zero means
    #means = means.loc[:, (means != 0).any(axis=0)]
    
    # get information about fidelities and high fidelity means
    unique_fidelities = np.unique(means["fidelity"])
    fidelity_max = np.max(unique_fidelities)
    fidelity_num = len(unique_fidelities)
    means_hf = means[ means["fidelity"] == np.max(unique_fidelities) ]    
    
    # NEW: use them to predict noise
    variances = np.square(dHdl.groupby(level=indexes).sem()).reset_index()
    # delete lambda paths not used i.e. with no var
    #variances = variances.loc[:, (variances != 0).any(axis=0)]
    
    # buid dfs containing high fidelity data
    mf_means = means[ means.fidelity==fidelity_max ].drop("fidelity", axis=1).set_index(indexes[1:])
    # hf variances will be overwritten later
    mf_variances = variances[ variances.fidelity==fidelity_max ].drop("fidelity", axis=1).set_index(indexes[1:])
    
    # integrate the different lambda paths with a MF model
    # get array containing fidelity numbers
    fidelity = np.array( means["fidelity"] )
    # get lambda path names/types
    loop = [ ix for ix in means.keys() if "-lambda" in ix ]
    # build structure to save cumulative integration results
    delta_matrix = np.zeros( ( len( np.array(means_hf[loop[0]] ) ),len(loop)  ) ) 
    var_matrix   = np.zeros( ( len( np.array(means_hf[loop[0]] ) ),len(loop)  ) )
    # loop over every different lambda subpath
    for l,idx in enumerate(loop):
        print(idx)
        # extract lambda paths to initialize MF model
        # works only for continuous paths
        x = np.array( means[idx] )
        if np.sum(x) != 0 and len( np.unique(x) ) > 1:
            y = np.array( means[idx.replace("-lambda","")] )
            y_var = np.array( variances[idx.replace("-lambda","")] )
            p = np.array([])
            for fid in unique_fidelities:
                pp  = np.squeeze(np.where(fidelity==fid ))
                dx  = np.gradient(x[ pp ] )
                ppp = np.squeeze(np.where(dx!=0))
                p = np.append(p,pp[ppp]).astype(int)
            if np.sum( np.diff(p) != 1 ) > len(unique_fidelities)-1 :
                print("WARNING: non-continuous lambda-path:",idx)
                #print("\t\t the MF-integration should fail.")
                print("\t\t compare your results to other methods.")
            # get data to ini MF model
            xp = x[p]
            yp = y[p]
            yp_var = y_var[p]
            fp = fidelity[p]
            # ini MF model
            mf = MF(xp,yp,fp)
            # fix noise
            # FUTURE: better solution. Make hyperparas optional
            for fid in unique_fidelities:
                pp  = np.squeeze(np.where(fp==fid ))
                noise = np.mean(yp_var[pp])/mf.Y_normer
                mf.lin_mf_model.gpy_model.param_array[ int(-fidelity_num+fid) ] = noise
                mf.lin_mf_model.gpy_model.mixed_noise.fix()
                
            #mf.lin_mf_model.gpy_model.mixed_noise.Gaussian_noise.fix(0.0000001)
            #mf.lin_mf_model.gpy_model.mixed_noise.Gaussian_noise_1.fix(0.0000001)    
            # train model :)
            mf.train()
            if savepath:
                mf.plot(savepath=savepath+"/"+idx+"_")
            else:
                mf.plot()
            # generate data for integrator
            # get high fidelity grid points
            p_high = np.where( mf.fidelity==fidelity_max )
            dummy = mf.X[p_high]
            # add additional grid points
            mi = np.squeeze(mf.X_min)
            ma = np.squeeze(mf.X_max)
            helper = np.atleast_2d( np.linspace( mi,ma,n ) ).T
            dummy = np.concatenate( [dummy, helper ],axis=0 )
            # predict dG/dl from MF for integrator
            hf_mean, hf_var = mf.predict(  dummy   )
            dummy   = np.squeeze(dummy)
            hf_mean = np.squeeze(hf_mean)
            hf_var  = np.squeeze(hf_var)
            # sort dummy and get reverse sort info
            phf     = np.squeeze(np.argsort(dummy))
            phf_rev = np.argsort(phf)
            # integrate all sorted points
            all_deltas = integrate.cumulative_trapezoid(hf_mean[phf], x=dummy[phf] )
            # get pointer to high fidelity data in initial dataset
            p_high = np.squeeze( np.where( fidelity==np.max(mf.unique_fidelities) ) )
            # get pointer to high fidelity data in MF training dataset
            p_hmf = np.squeeze(np.where( fp==np.max(mf.unique_fidelities)  ))
            # get pointer for filling data
            p_fill = p[p_hmf] - np.min(p_high)
            # extract integrated high fidelity predictions of the initial grid
            filler     = all_deltas[ phf_rev[ : p_fill.shape[0] ] ]
            var_filler = hf_var[ phf_rev[ : p_fill.shape[0] ] ]
            # prepare structure to be filled with MF integrated results
            to_be_filled = np.zeros( means_hf.shape[0] )
            vars_to_be_filled = np.zeros( means_hf.shape[0] )
            # fill in results of the integrated areas of the overall lambda pathway 
            to_be_filled[p_fill]      = filler
            vars_to_be_filled[p_fill] = var_filler
            # apply cumulative filling for integrals
            p_empty = np.setdiff1d( np.arange(len(to_be_filled)),p_fill ) 
            d_mat = np.abs(np.subtract( np.atleast_2d(p_fill), np.atleast_2d(p_empty).T ))
            ps = np.argmin(d_mat,axis=-1)
            to_be_filled[p_empty] = to_be_filled[p_fill[ps]]
            # add MF integrated results to the main structure
            delta_matrix[:,l] = to_be_filled
            var_matrix[:,l]   = vars_to_be_filled
            mf_variances[idx.replace("-lambda","")] = vars_to_be_filled
        else:
            print("passed cause empty",idx)
    return delta_matrix, var_matrix,mf_means, mf_variances     