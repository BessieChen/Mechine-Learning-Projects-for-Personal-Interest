from cvxpy import *
import numpy as np
from posdef import *

mu = np.matrix([[-0.01132331],[-0.05061149],[-0.00729714],[-0.06206554],[-0.01079347],[0.0025]])
Sigma = np.array(
	[[2.52879395e-03, 1.85425637e-03, 9.20436767e-04, 1.95914715e-03, 4.28396944e-03, 0.00000000e+00],
	 [1.85425637e-03, 2.15246031e-03, 9.42088127e-04, 1.58969365e-03, 3.30104182e-03, 0.00000000e+00],
	 [  9.20436767e-04, 9.42088127e-04, 2.17054891e-03, 7.91879289e-04, 1.39817285e-03, 0.00000000e+00],
	 [  1.95914715e-03, 1.58969365e-03, 7.91879289e-04, 1.67831424e-03, 3.77524505e-03, 0.00000000e+00],
	 [  4.28396944e-03, 3.30104182e-03, 1.39817285e-03, 3.77524505e-03, 8.72500053e-03, 0.00000000e+00],
	 [  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.73472348e-18]])

print(isPD(Sigma))

min_return = 0.0
n = 6

w = Variable(n)
ret = mu.T*	w
risk = quad_form(w, Sigma)
min_ret = Parameter(sign='positive')
min_ret.value = min_return

prob = Problem(Minimize(risk),
               [ret >= min_ret,
               sum_entries(w) == 1,
               w >= 0])
prob.solve()
print(w.value)


# Compute and plot the optimum risk-return trade-off for 10 assets, restricting ourselves to a long only portfolio

# Generate data for long only portfolio optimization.
import numpy as np
np.random.seed(1)
n = 10								# Number of assets
mu = np.abs(np.random.randn(n, 1))	# Mean returns of n assets
Sigma = np.random.randn(n, n)
Sigma = Sigma.T.dot(Sigma)			# Covarience for n assets

# Long only portfolio optimization.
from cvxpy import *
w = Variable(n)								# Portfolio allocation vector
gamma = Parameter(sign='positive')			# Risk aversion parameter
ret = mu.T*	w
risk = quad_form(w, Sigma)					# w.T * Sigma * w
prob = Problem(Maximize(ret - gamma*risk),	# Restricting to long-only portfolio
			   [sum_entries(w) == 1,
				w >= 0])

# Compute trade-off curve.
SAMPLES = 100
risk_data = np.zeros(SAMPLES)
ret_data = np.zeros(SAMPLES)
gamma_vals = np.logspace(-2, 3, num=SAMPLES)	# SAMPLES evenly spaced values on logspace
for i in range(SAMPLES):
	gamma.value = gamma_vals[i]
	prob.solve()
	risk_data[i] = sqrt(risk).value
	ret_data[i] = ret.value

# Plot long only trade-off curve.
import matplotlib.pyplot as plt

markers_on = [29, 40]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(risk_data, ret_data, 'g-')
for marker in markers_on:
	plt.plot(risk_data[marker], ret_data[marker], 'bs')
	ax.annotate(r"$\gamma = %.2f$" % gamma_vals[marker], xy=(risk_data[marker]+.08, ret_data[marker]-.03))
for i in range(n):
	plt.plot(sqrt(Sigma[i,i]).value, mu[i], 'ro')
plt.xlabel('Standard deviation')
plt.ylabel('Return')
# plt.savefig('tradeoff.png')
plt.show()

# Plot return distributions for two points on the trade-off curve.
import matplotlib.mlab as mlab
plt.figure()
for midx, idx in enumerate(markers_on):
	gamma.value = gamma_vals[idx]
	prob.solve()
	x = np.linspace(-2, 5, 1000)
	plt.plot(x, mlab.normpdf(x, ret.value, risk.value), label=r"$\gamma = %.2f$" % gamma.value)

plt.xlabel('Return')
plt.ylabel('Density')
plt.legend(loc='upper right')
# plt.savefig('return_dist.png')
plt.show()

# The probability of a loss is near 0 for the low risk value and far above 0 for the high risk value.


####################################################################################################


# Portfolio optimization with the leverage limit constraint.
Lmax = Parameter()							# Leverage limit (Leverage = L1 norm of w)
prob = Problem(Maximize(ret - gamma*risk),
			   [sum_entries(w) == 1,
				norm(w, 1) <= Lmax])

# Compute trade-off curve for each leverage limit.
L_vals = [1, 2, 4]
SAMPLES = 100
risk_data = np.zeros((len(L_vals), SAMPLES))
ret_data = np.zeros((len(L_vals), SAMPLES))
gamma_vals = np.logspace(-2, 3, num=SAMPLES)
w_vals = []
for k, L_val in enumerate(L_vals):
	for i in range(SAMPLES):
		Lmax.value = L_val
		gamma.value = gamma_vals[i]
		prob.solve()
		risk_data[k, i] = sqrt(risk).value
		ret_data[k, i] = ret.value

# Plot trade-off curves for each leverage limit.
for idx, L_val in enumerate(L_vals):
	plt.plot(risk_data[idx,:], ret_data[idx,:], label=r"$L^{\max}$ = %d" % L_val)
plt.xlabel('Standard deviation')
plt.ylabel('Return')
plt.legend(loc='lower right')
# plt.savefig('tradeoff_leverage.png')
plt.show()


####################################################################################################


# Portfolio optimization with a leverage limit and a bound on risk.
prob = Problem(Maximize(ret),
			  [sum_entries(w) == 1,
			   norm(w, 1) <= Lmax,
			   risk <= 2])

# Compute solution for different leverage limits.
for k, L_val in enumerate(L_vals):
	Lmax.value = L_val
	prob.solve()
	w_vals.append( w.value )

# Plot bar graph of holdings for different leverage limits. (Negative holdings indicate a short position)
colors = ['b', 'g', 'r']
indices = np.argsort(mu.flatten())
for idx, L_val in enumerate(L_vals):
	 plt.bar(np.arange(1,n+1) + 0.25*idx - 0.375, w_vals[idx][indices], color=colors[idx],
			 label=r"$L^{\max}$ = %d" % L_val, width = 0.25)
plt.ylabel(r"$w_i$", fontsize=16)
plt.xlabel(r"$i$", fontsize=16)
plt.xlim([1-0.375, 10+0.375])
plt.xticks(np.arange(1,n+1))
# plt.savefig('holdings.png')
plt.show()

