from kernels import *

# Constants
EPS = 0.00001

# MARKS
# 1. Maybe we need to change 'set()' with 'list()', because set() structure has automatic sorting (ordering)!
# 2. We need to re-read this code and check, if all these formulas are correct.


class SVMClassifier:
    def __init__(self, kernel=LinearKernel, C=30.0):
        # SVM params
        self.alphas = []
        self.b = 0
        self.x = []
        self.y = []
        self.kernel = LinearKernel

        # learning must-have variables
        self.g = []  # vector of conditions
        self.C = C  # regularization param
        self.R = [[0]]  # Inversive Q matrix
        self.S = set()  # margin support vectors set S
        self.E = set()  # error vectors set E
        self.R = set()  # non-support vectors
        pass

    def CalculateBetas(self, x_c, y_c):
        # step 1 - count betas for new C and support vectors
        tmpVec = [y_c] + [self.y[s] * y_c * self.kernel(self.x[s], x_c) for s in self.S]
        betasInS = -1 * np.array(self.R) @ np.array(tmpVec)

        # step 2 - build vector of betas for whole training samples
        betas = []
        j = 1
        # 2.1 - first - add previous indexes
        for i in range(len(self.alphas)):
            if i not in self.S:
                betas.append(0)
            else:
                betas.append(betasInS[1])
                j += 1

        # 2.2 - second, add beta of new index "c" in the end
        betas.append(betasInS[0])
        return betas

    def CalculateGammas(self, x_c, y_c, betas):
        gammas = []

        # first - add previous indexes
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

        # add gamma of new index "c"
        res = y_c * y_c * self.kernel(x_c, x_c) + y_c * betas[-1]
        for j in self.S:
            res += y_c * self.y[j] * self.kernel(x_c, self.x[j]) * betas[j]
        gammas.append(res)
        return gammas

    def CalculateDeltaAc(self, x_c, betas, gammas, alpha_c):
        # step 0 - init var
        delta_ac = 0

        # step 1 - 1st case
        deltaAmax = []
        for i in self.S:
            if betas[i] > EPS:
                deltaAmax.append(abs((self.C - self.alphas[i]) / betas[i]))
            elif betas[i] < -EPS:
                deltaAmax.append(abs(-self.alphas[i] / betas[i]))
        deltaAcS = min(deltaAmax)

        # step 2 - 2nd case
        deltaARs = []
        for i in range(len(self.alphas)):
            if (i in self.E and gammas[i] > EPS) or (i in self.R and gammas[i] < -EPS):
                deltaARs.append(-self.g[i] / gammas[i])
        deltaAcR = min(deltaARs)

        # step 3 - 3rd case
        deltaAcg = -self.g[i] / gammas[-1]

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
        betas = -1 * np.array(self.R) @ np.array(tmpVec)

        # step 2 - count gamma
        tmpBetas = betas[1:]
        gamma = new_y * new_y * self.kernel(new_x, new_x) + new_y * betas[0]
        for j in self.S:
            gamma += new_y * self.y[j] * self.kernel(new_x, self.x[j]) * tmpBetas[j]

        # step 3 - update R
        betas = np.array(
            betas.tolist().append(1)
        )  # лютый п****ц по производительности........
        tmpBetas = np.array([[b] for b in betas])
        tmpR = np.pad(np.array(self.R), (1, 1), ((0, 1), (0, 1)), "constant", 0)
        self.R = (tmpR + 1.0 / gamma * tmpBetas @ betas).tolist()

        # step 4 - update set S
        self.S.add(new_index)
        return

    def DeleteFromSetSIndex(self, index_to_delete):
        # step 0 - map index to delete from set 'S' into row&col
        k = self.S.tolist().index(index_to_delete) + 1  # '+1' is for b-term

        # step 1 - recompute matrix
        N = len(self.S.tolist())
        tmpR = []
        for i in range(N):
            if i == k:
                continue
            tmpR.append([])
            for j in range(N):
                if j == k:
                    continue
                tmpR[-1].append(
                    self.R[i][j] - 1.0 / self.R[k][k] * self.R[i][k] * self.R[k][j]
                )

        self.R = tmpR
        return

    def IncrementPair(self, x_c, y_c):
        # step 0 - init alpha and evaluate g_c
        alpha_c = 0
        g_c = self(x_c) - 1

        # step 1.0 - check while loop conditions:
        while g_c < 0 and alpha_c < self.C:
            # step 1.1 - compute betas and gammas
            betas = self.CalculateBetas(x_c, y_c)
            gammas = self.CalculateGammas(x_c, y_c, betas)

            # step 1.2 - compute max step
            delta_ac = self.CalculateDeltaAc(x_c, betas, gammas, alpha_c)

            # step 1.3 - update variables
            self.b = betas[-1]
            alpha_c += delta_ac
            for i in self.S:
                self.alphas[i] += betas[i] * delta_ac
            for i in range(len(self.g)):
                self.g[i] += gammas[i] * delta_ac

            # step 1.4 - check some conditions:
            if alpha_c == self.C:  # append new index of error vector
                self.E.add(len(self.alphas))
                break
            elif g_c == 0:  # append new index of error vector
                b_c = betas[-1]
                self.AddToSetSIndex(len(self.alphas), x_c, y_c)
                break
            else:  # Manage indexes bookkeeping in sets
                for s in self.S:
                    if abs(self.g[s]) > EPS:
                        self.DeleteFromSetSIndex(s)
                        if self.g[s] < -EPS:
                            self.E.add(s)
                        else:
                            self.R.add(s)

                for r in self.R:
                    if abs(self.g[r]) < EPS:
                        self.AddToSetSIndex(r, self.x[r], self.y[r])
                        self.R.discard(r)
                    # TODO: As i understood, this case is imposible
                    elif self.g[r] < -EPS:
                        self.R.discard(r)
                        self.E.add(r)

                for e in self.E:
                    if abs(self.g[e]) < EPS:
                        self.AddToSetSIndex(e, self.x[e], self.y[e])
                        self.E.discard(e)
                    # TODO: As i understood, this case is imposible
                    elif self.g[e] > EPS:
                        self.E.discard(e)
                        self.R.add(e)

        # save new
        self.x.append(x_c)
        self.y.append(y_c)
        self.alphas.append(alpha_c)
        self.g.append(g_c)
        return

    # TODO: optimize woth numpy.dot method with 'packing' alpha_i and y_i product result in one vector. Also make 'vectorize' with Kernel func
    def __call__(self, x):
        res = self.b

        for i in range(len(self.alphas)):
            res += self.alphas[i] * self.y[i] * self.kernel(self.x[i], x)

        return res
