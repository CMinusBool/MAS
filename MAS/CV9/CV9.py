import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm, expon, powerlaw, poisson, lognorm

# normal distribution
normal_dist = np.random.normal(0, 10, 5500)

normal_fit = norm.fit(normal_dist)
x_axis = np.arange(min(normal_dist), max(normal_dist), 0.01)

# distribution graph
plt.hist(normal_dist, bins=100, density= True, stacked= True)
plt.plot(x_axis, norm.pdf(x_axis, normal_fit[0], normal_fit[1]), 'r')
plt.title('Normal Distribution')
plt.show()

# exponential distribution
exponential_dist = np.random.exponential(1, 5500)

exponential_fit = expon.fit(exponential_dist)
x_axis = np.arange(min(exponential_dist), max(exponential_dist), 0.01)

# distribution graph
plt.hist(exponential_dist, bins=100, density= True, stacked= True)
plt.plot(x_axis, expon.pdf(x_axis, exponential_fit[0], exponential_fit[1]), 'r')
plt.title('Exponential Distribution')
plt.show()

k_test = kstest(exponential_dist, 'expon')

# power law distribution
powerlaw_dist = np.random.power(6, 5500)

powerlaw_fit = powerlaw.fit(powerlaw_dist)

x_axis = np.arange(min(powerlaw_dist), max(powerlaw_dist), 0.01)

# distribution graph
plt.hist(powerlaw_dist, bins=100, density= True, stacked= True)
plt.plot(x_axis, powerlaw.pdf(x_axis, powerlaw_fit[0]), 'r')
plt.title('Power Law Distribution')
plt.show()

# Poisson distribution
poisson_dist = np.random.poisson(lam=8, size=5500)
poisson_fit = np.mean(poisson_dist)
x_axis_poisson = np.arange(min(poisson_dist), max(poisson_dist), 1)

# Distribution graph for Poisson
plt.hist(poisson_dist, bins=20, density=True, label='Histogram')
plt.plot(x_axis_poisson, poisson.pmf(x_axis_poisson, poisson_fit), 'r', label='Fitted Poisson PMF')
plt.title('Poisson Distribution')
plt.show()

# Log-Normal distribution
lognormal_dist = np.random.lognormal(mean=1, sigma=0.5, size=5500)
lognormal_fit = lognorm.fit(lognormal_dist)
x_axis_lognormal = np.linspace(min(lognormal_dist), max(lognormal_dist), 1000)

# Distribution graph for Log-Normal
plt.hist(lognormal_dist, bins=100, density=True, label='Histogram')
plt.plot(x_axis_lognormal, lognorm.pdf(x_axis_lognormal, *lognormal_fit), 'r', label='Fitted Log-Normal PDF')
plt.title('Log-Normal Distribution')
plt.show()

# Ks test for all distributions
k_test = kstest(normal_dist, lambda x: norm.cdf(x, normal_fit[0], normal_fit[1]))
print(f"Ks test for Normal distribution: ")
print(f"D-value: {k_test[0]:4f}, P-value: {k_test[1]:4f}\n")

k_test = kstest(exponential_dist, lambda x: expon.cdf(x, exponential_fit[0], exponential_fit[1]))
print(f"Ks test for Exponential distribution: ")
print(f"D-value: {k_test[0]:4f}, P-value: {k_test[1]:4f}\n")

k_test = kstest(powerlaw_dist, lambda x: powerlaw.cdf(x, powerlaw_fit[0]))
print(f"Ks test for Power Law distribution:")
print(f"D-value: {k_test[0]:4f}, P-value: {k_test[1]:4f}\n")

# ks test je pro kontinualni distribuce a ne pro diskrétní
ks_poisson = kstest(poisson_dist, 'poisson', args=(poisson_fit,))
print("K-S Test for Poisson Distribution:")
print(f"D-value: {ks_poisson.statistic:.4f}, P-value: {ks_poisson.pvalue:.4f}\n")

ks_lognormal = kstest(lognormal_dist, lambda x: lognorm.cdf(x, *lognormal_fit))
print("K-S Test for Log-Normal Distribution:")
print(f"D-value: {ks_lognormal.statistic:.4f}, P-value: {ks_lognormal.pvalue:.4f}\n")