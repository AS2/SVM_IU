from kernels import *
from utils import *

# Constants
EPS = 0.00001
INF = 1e12


class SVMClassifier:
    def __init__(self, kernel=LinearKernel, C=30.0):
        # SVM params
        self.alphas = []
        self.b = 0
        self.x = []
        self.y = []
        self.kernel = kernel

        # learning must-have variables
        self.C = C  # regularization param
        self.Rm = [[0]]  # Inversive Q matrix
        self.g = []  # vector of conditions

        # We must use 'list()' instead 'set()', because set() structure automaticly orders values in it!
        self.S = []  # margin support vectors set S
        self.E = []  # error vectors set E
        self.R = []  # non-support vectors
        pass

    # Note: we calculate betas ONLY for elements of D (we dont calculate beta for new (x_c, y_c) pair) and b-term
    def CalculateBetas(self, x_c, y_c):
        # step 1 - count betas for new C and support vectors
        tmpVec = [y_c] + [self.y[s] * y_c * self.kernel(self.x[s], x_c) for s in self.S]
        betasInS = -1 * np.array(self.Rm) @ np.array(tmpVec)

        # step 2 - build vector of betas for whole training samples
        betas = []
        j = 1
        # 2.1 - firstly, we generate all betas for alphas
        for i in range(len(self.alphas)):
            if i not in self.S:
                betas.append(0)
            else:
                betas.append(betasInS[j])
                j += 1

        # 2.2 - secondly, we store beta for b-term as last element of list
        betas.append(betasInS[0])
        return betas

    # Note: here we calculate betas not only for elements of D, but also for new element (x_c, y_c) pair. We dont calculate it for beta
    def CalculateGammas(self, x_c, y_c, betas):
        gammas = []

        # first - calculate gammas for previous alphas
        for i in range(len(self.alphas)):
            if i not in self.S:
                res = (
                    self.y[i] * y_c * self.kernel(self.x[i], x_c)
                    + self.y[i] * betas[-1]
                )
                for j in self.S:
                    res += (
                        self.y[i]
                        * self.y[j]
                        * self.kernel(self.x[i], self.x[j])
                        * betas[j]
                    )
                gammas.append(res)
            else:
                gammas.append(0)

        # second - calculate gamma for
        gamma_c = y_c * y_c * self.kernel(x_c, x_c) + y_c * betas[-1]
        for j in self.S:
            gamma_c += y_c * self.y[j] * self.kernel(x_c, self.x[j]) * betas[j]
        return gammas, gamma_c

    def CalculateDeltaAc(self, betas, gammas, gamma_c, g_c, alpha_c):
        # step 1 - 1st case
        deltaAmax = []
        for s in self.S:
            if betas[s] > EPS:
                deltaAmax.append((self.C - self.alphas[s]) / betas[s])
            elif betas[s] < -EPS:
                deltaAmax.append(-self.alphas[s] / betas[s])

        # TODO: need to be rewritted as in Laskov
        if len(deltaAmax) != 0:
            deltaAcS = absmin(deltaAmax)
        else:
            deltaAcS = INF

        # step 2 - 2nd case
        deltaARs = []
        for i in range(len(self.alphas)):
            if (i in self.E and gammas[i] > EPS) or (i in self.R and gammas[i] < -EPS):
                deltaARs.append(-self.g[i] / gammas[i])

        # TODO: need to be rewritted as in Laskov
        if len(deltaARs) != 0:
            deltaAcR = min(deltaARs)
        else:
            deltaAcR = INF

        # step 3 - 3rd case
        if gamma_c > EPS:
            deltaAcg = -g_c / gamma_c
        else:
            deltaAcg = INF

        # step 4 - 4th case
        deltaAca = self.C - alpha_c

        return min([deltaAcS, deltaAcR, deltaAcg, deltaAca])

    # TODO: Rewrite this and previous functions, which depends on 'ordered' elements of set S (they can be uordered!)
    # Also rewrite this and write next function like in the Tomaso Poggio Incremental and Decremental Support Vector Machine Learning
    def AddToSetSIndex(self, new_index, new_x, new_y):
        # step 1 - count betas for new appended support vector
        tmpVec = [new_y] + [
            self.y[s] * new_y * self.kernel(self.x[s], new_x) for s in self.S
        ]
        betas = -1 * np.array(self.Rm) @ np.array(tmpVec)

        # step 2 - count gamma
        tmpBetas = betas[1:]
        gamma = new_y * new_y * self.kernel(new_x, new_x) + new_y * betas[0]
        for i, s in enumerate(self.S):
            gamma += new_y * self.y[s] * self.kernel(new_x, self.x[s]) * tmpBetas[i]

        # step 3 - update R
        betas = betas.tolist()
        betas.append(1.0)
        tmpBetas = np.array([[b] for b in betas])
        betas = np.array([betas])  # лютый п****ц по производительности........

        tmpRm = np.pad(
            array=np.array(self.Rm),
            pad_width=((0, 1), (0, 1)),
            mode="constant",
            constant_values=((0, 0), (0, 0)),
        )
        self.Rm = (tmpRm + 1.0 / gamma * tmpBetas @ betas).tolist()

        # step 4 - update set S
        self.S.append(new_index)
        return

    def DeleteFromSetSIndex(self, index_to_delete):
        # step 0 - map index to delete from set 'S' into row&col
        k = (
            self.S.index(index_to_delete) + 1
        )  # '+1' is for b-term [zero row&col is for b-term, remember that]

        # step 1 - recompute matrix
        N = len(self.S) + 1  # '+1' is for b-term
        tmpRm = []
        for i in range(N):
            if i == k:
                continue
            tmpRm.append([])
            for j in range(N):
                if j == k:
                    continue
                tmpRm[-1].append(
                    self.Rm[i][j] - 1.0 / self.Rm[k][k] * self.Rm[i][k] * self.Rm[k][j]
                )

        self.Rm = tmpRm
        self.S.remove(index_to_delete)
        return

    def AddFirstElement(self, x_c, y_c):
        self.alphas.append(0)
        self.b = y_c * 2
        self.x.append(x_c)
        self.y.append(y_c)

        self.R.append(0)  # index of vector
        self.g.append(y_c * self(x_c) - 1)
        self.Rm = [[0]]
        return

    # TODO: implement method to set some
    def TakeCareOfS(self, g_c, x_c, y_c):
        for e in self.E:
            if self.g[e] == 0:
                self.AddToSetSIndex(e, self.x[e], self.y[e])
                self.E.remove(e)
                return g_c, False

        for r in self.R:
            if self.g[r] == 0:
                self.AddToSetSIndex(r, self.x[r], self.y[r])
                self.R.remove(r)
                return g_c, False

        mus_r = []
        for i in range(len(self.alphas)):
            if i in self.E and -(self.y[i] / y_c) > EPS:
                mus_r.append(-self.g[i] / self.y[i])
            elif i in self.R and -(self.y[i] / y_c) < -EPS:
                mus_r.append(-self.g[i] / self.y[i])
        if len(mus_r) != 0:
            mu_r = min(mus_r)
        else:
            mu_r = INF
        mu_c = -g_c / y_c
        mu = min(mu_r, mu_c)

        for i in range(len(self.alphas)):
            self.g[i] += self.y[i] * mu
        g_c += y_c * mu

        for e in self.E:
            if self.g[e] == 0:
                self.AddToSetSIndex(e, self.x[e], self.y[e])
                self.E.remove(e)
                return g_c, False

        for r in self.R:
            if self.g[r] == 0:
                self.AddToSetSIndex(r, self.x[r], self.y[r])
                self.R.remove(r)
                return g_c, False

        self.AddToSetSIndex(len(self.alphas), x_c, y_c)
        self.b += mu
        return g_c, True

    def IncrementPair(self, x_c, y_c):
        # step -1: add first element to init all variables
        if len(self.alphas) == 0:
            self.AddFirstElement(x_c, y_c)
            return

        # step 0 - init alpha and evaluate g_c
        alpha_c = 0
        g_c = y_c * self(x_c) - 1

        # step 1.0 - check while loop conditions:
        while g_c <= 0:
            if len(self.S) == 0:
                g_c, isNeedToLeave = self.TakeCareOfS(g_c, x_c, y_c)
                if isNeedToLeave:
                    return

            # step 1.1 - compute betas and gammas
            betas = self.CalculateBetas(x_c, y_c)
            gammas, gamma_c = self.CalculateGammas(x_c, y_c, betas)

            # step 1.2 - compute max step
            delta_ac = self.CalculateDeltaAc(betas, gammas, gamma_c, g_c, alpha_c)

            # step 1.3 - update variables
            self.b += betas[-1] * delta_ac
            alpha_c += delta_ac
            for i in range(len(self.alphas)):
                self.alphas[i] += betas[i] * delta_ac
            for i in range(len(self.g)):
                self.g[i] += gammas[i] * delta_ac
            g_c += gamma_c * delta_ac

            # step 1.4 - check some conditions:
            if abs(g_c) < EPS:  # append new index of support vector
                self.AddToSetSIndex(len(self.alphas), x_c, y_c)
                break
            elif abs(alpha_c - self.C) < EPS:  # append new index of error vector
                self.E.append(len(self.alphas))
                break
            else:  # Manage indexes bookkeeping in sets
                wasTransfer = False
                for s in self.S:
                    if abs(self.alphas[s]) < EPS or abs(self.alphas[s] - self.C) < EPS:
                        wasTransfer = True
                        self.DeleteFromSetSIndex(s)
                        if abs(self.alphas[s]) < EPS:
                            self.R.append(s)
                        else:
                            self.E.append(s)

                if not wasTransfer:
                    for r in self.R:
                        if abs(self.g[r]) < EPS:
                            self.AddToSetSIndex(r, self.x[r], self.y[r])
                            self.R.remove(r)

                if not wasTransfer:
                    for e in self.E:
                        if abs(self.g[e]) < EPS:
                            self.AddToSetSIndex(e, self.x[e], self.y[e])
                            self.E.remove(e)

            # assert self.CheckAllConditions()

        # save new
        new_index = len(self.x)
        self.x.append(x_c)
        self.y.append(y_c)
        self.alphas.append(alpha_c)
        self.g.append(g_c)
        if new_index not in self.S and new_index not in self.E:
            self.R.append(new_index)

        # ASSERT FOR CORRECTION
        # assert self.CheckAllConditions()

        return

    def CheckAllConditions(self):
        # check alphas
        for s in self.S:
            if abs(self.alphas[s]) <= EPS or abs(self.alphas[s] - self.C) <= EPS:
                return False

        for r in self.R:
            if abs(self.alphas[r]) >= EPS:
                return False

        for e in self.E:
            if abs(self.alphas[e] - self.C) >= EPS:
                return False

        # check g
        for s in self.S:
            if abs(self.g[s]) > EPS:
                return False

        for r in self.R:
            if self.g[r] < -EPS:
                return False

        for e in self.E:
            if self.g[e] > EPS:
                return False

        return True

    # TODO: optimize woth numpy.dot method with 'packing' alpha_i and y_i product result in one vector. Also make 'vectorize' with Kernel func
    def __call__(self, x):
        res = self.b

        for i in range(len(self.alphas)):
            res += self.alphas[i] * self.y[i] * self.kernel(self.x[i], x)

        return res
