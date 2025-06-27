class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]

class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for key in params:
            grad = grads[key]
            self.m[key] = self.betas[0] * self.m[key] + (1 - self.betas[0]) * grad
            self.v[key] = self.betas[1] * self.v[key] + (1 - self.betas[1]) * grad ** 2
            m_hat = self.m[key] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[key] / (1 - self.betas[1] ** self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)