import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#Normal Distribution from scratch
def normal_pdf(x: float,mu:float = 0,sigma:float=1) ->float:
    return (1/(math.sqrt(2*math.pi)*sigma))*math.exp(-((x-mu)**2)/(2*sigma**2))

mu = 0
sigma = 3

plt.subplot(1,2,1)
xs = np.linspace(mu-3*sigma,mu+3*sigma,100)
plt.plot(xs,[normal_pdf(x,mu,sigma) for x in xs],'-')
plt.title("From Scratch")

#Normal Distribution from scipy.stats
import scipy.stats as stats

mu = 50
sigma = 1

plt.subplot(1,2,2)
xs = np.linspace(mu-3*sigma,mu+3*sigma,100)
plt.plot(xs,stats.norm.pdf(xs,mu,sigma))
plt.title('From scipy.stats')
plt.show()