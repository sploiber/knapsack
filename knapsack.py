from math import log
from pyqubo import Binary


# From Andrew Lucas, NP-hard combinatorial problems as Ising spin glasses
# Workshop on Classical and Quantum Optimization; ETH Zuerich - August 20, 2014
# based on Lucas, Frontiers in Physics _2, 5 (2014)

class Knapsack(object):

    def __init__(self, names, costs, weights, W):

        self.names = names

        # Lagrangian multiplier
        A = max(costs)

        logw = int(log(W)) + 1
        m = logw - 1

        # Obtain the Pyqubo qubit variables
        qbits_x = {u: Binary("x_{}".format(u)) for u in range(len(weights))}
        qbits_y = {u: Binary("y_{}".format(u)) for u in range(logw)}

        sumy = (W + 1 - (2 ** m)) * qbits_y[m]
        for n in range(m):
            sumy += (2 ** n) * qbits_y[n]

        # Set up the wx term
        wx_val = [weights[k] * qbits_x[k] for k in range(len(weights))]

        HA = (sumy - sum(wx_val)) ** 2
        cx_val = [costs[k] * qbits_x[k] for k in range(len(costs))]
        H = (A * HA) - sum(cx_val)

        # Convert the Hamiltonian from symbols and turn it into a QUBO
        self.model = H.compile()
        self.qubo, self.offset = self.model.to_qubo()

    def get_bqm(self):
        return self.model.to_dimod_bqm(self.offset)

    def get_names(self, solution):
        return [self.names[index] for index, s in enumerate(solution[:len(self.names)]) if s == 1]
