import pandas as pd, numpy as np, matplotlib.pyplot as plt, emcee, corner
import radvel
import radvel.likelihood
from radvel.plot import orbit_plots, mcmc_plots
from scipy import optimize

#read data
data = pd.read_table("star2_radialvelocities.dat",sep="\s+",names=["julian_time","rv","rv_error"])
#data columns
rv = np.array(data["rv"]); rv_error = np.array(data["rv_error"]); time = np.array(data["julian_time"])

# the values declared are guesses
def initialize_model():
    time_base = time[0]
    params = radvel.Parameters(1,basis='per tp e w k') # number of planets = 1
    params['per1'] = radvel.Parameter(value=800)
    params['tp1'] = radvel.Parameter(value=53000)
    params['e1'] = radvel.Parameter(value=0.4)
    params['w1'] = radvel.Parameter(value=-0.6)
    params['k1'] = radvel.Parameter(value=0.04)

    mod = radvel.RVModel(params, time_base=time_base)
    return mod

#plot the result
def plot_results(like,time):
    fig = plt.figure(figsize=(6.5,5))
    fig = plt.gcf()
    fig.set_tight_layout(True)
    plt.errorbar(like.x, like.model(time)+like.residuals(),yerr=like.yerr, fmt='k.',label='data')
    ti = np.linspace(min(time),max(time),1000)
    plt.plot(ti, like.model(ti),label="Fit",color="b")
    plt.xlabel('Date of observation (Julian Days)')
    plt.ylabel('RV (km/s)')
    plt.title("Fitting the data using RadVel tool")
    plt.legend()
    plt.show()

#initialize model and set the parameters to be varied or not, as well as the gamma parameter (offset)
mod = initialize_model()
like = radvel.likelihood.RVLikelihood(mod, time, rv, rv_error)
like.params['gamma'] = radvel.Parameter(value=40, vary=False, linear=True)
like.params['gamma'].vary = True
like.params['e1'].vary = True
like.params['w1'].vary = True
like.params['per1'].vary = True
like.params['tp1'].vary = True
like.params['k1'].vary = True
print(like)

#posterior probabilities, attending to the priors
post = radvel.posterior.Posterior(like)
post.priors += [radvel.prior.Gaussian( 'k1', 0.04, 0.03)]
post.priors += [radvel.prior.Gaussian( 'gamma', 40, 10)]
post.priors += [radvel.prior.Gaussian( 'per1', 800, 50)]
post.priors += [radvel.prior.Gaussian( 'tp1', 53000, 5000)]
post.priors += [radvel.prior.Gaussian( 'e1', 0.4, 0.2)]
post.priors += [radvel.prior.Gaussian( 'w1', -0.6, 0.5)]
res  = optimize.minimize(
    post.neglogprob_array,     # objective function is negative log likelihood
    post.get_vary_params(),    # initial variable parameters
    method='Nelder-Mead')

plot_results(like,time)         # plot best fit model
print(post)

# these part is supposed to give the corner plot of the fit, but an unknown error occured within the package
df = radvel.mcmc(post,nwalkers=20,nrun=400,savename='rawchains.h5')
Corner = mcmc_plots.CornerPlot(post, df)
Corner.plot()