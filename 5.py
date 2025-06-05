import numpy as np
X, y = np.array([[2,9],[1,5],[3,6]],dtype=float), np.array([[92],[86],[89]],dtype=float)
X, y = X/np.amax(X,0), y/100

def sigmoid(x): return 1/(1+np.exp(-x))
def dsigmoid(x): return x*(1-x)

epoch, lr = 5000, 0.1
wh, bh = np.random.rand(2,3), np.random.rand(1,3)
wo, bo = np.random.rand(3,1), np.random.rand(1,1)

for _ in range(epoch):
    hl = sigmoid(np.dot(X, wh) + bh)
    out = sigmoid(np.dot(hl, wo) + bo)
    d_out = (y - out) * dsigmoid(out)
    d_hl = d_out.dot(wo.T) * dsigmoid(hl)
    wo += hl.T.dot(d_out) * lr
    wh += X.T.dot(d_hl) * lr

print("Predicted:\n", out)
