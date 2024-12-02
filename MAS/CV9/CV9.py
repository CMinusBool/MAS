import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm, expon, powerlaw

# normal distribution
normal_dist = np.random.normal(0, 10, 5500)

normal_fit = norm.fit(normal_dist)
x_axis = np.arange(min(normal_dist), max(normal_dist), 0.01)

# distribution graph
plt.hist(normal_dist, bins=100, density= True, stacked= True)
plt.plot(x_axis, norm.pdf(x_axis, normal_fit[0], normal_fit[1]), 'r')
plt.title('Normal Distribution')
plt.show()

k_test = kstest(normal_dist, 'norm', args=(normal_fit[0], normal_fit[1]))

print(f"Normal distribution: {k_test}")

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

#k_test = kstest(powerlaw_dist, 'powerlaw')
print(f"Power Law distribution: {k_test}")