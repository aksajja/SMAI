import matplotlib.pyplot as plt

next_x = -2  # We start the search at x=-2
Eta = 0.1  # Eta
precision = 0.00001  # Desired precision of result
max_iters = 100  # Maximum number of iterations
iterations = list(range(-2, 98))

def df(x):
    return 2*x

def gradient_descent(Eta, next_x, max_iter):
    s = []
    convergence_val = []
    print(Eta)
    for i in range(max_iters):
        current_x = next_x
        next_x = current_x - Eta * df(current_x)
        step = next_x - current_x
        if(step>=precision):
            convergence_val.append(step)
        s.append(step)
    return s, convergence_val

s, convergence_val = gradient_descent(0.1, -2, 100)
plt.plot(iterations, s)
plt.xlabel('iteration number', fontsize=12)
plt.ylabel('error', fontsize=12)
plt.show()

plt.plot(iterations[0:len(convergence_val)], convergence_val)
plt.xlabel('iteration number', fontsize=12)
plt.ylabel('convergence criteria', fontsize=12)
plt.show()

new_etas = [0.01, 1.5, 0.9995]
for i in range(3):
    plt.subplot(221)
    s, convergence_val = gradient_descent(new_etas[i], -2, 100)
    plt.plot(iterations, s)
    plt.xlabel('iteration number', fontsize=12)
    plt.ylabel('convergence', fontsize=12)

    plt.subplot(222)
    s, convergence_val = gradient_descent(new_etas[i+1], -2, 100)
    plt.plot(iterations, s)
    plt.xlabel('iteration number', fontsize=12)
    plt.ylabel('divergence', fontsize=12)

    plt.subplot(212)
    s, convergence_val = gradient_descent(new_etas[i+2], -2, 100)
    plt.plot(iterations, s)
    plt.xlabel('iteration number', fontsize=12)
    plt.ylabel('oscilattions', fontsize=12)
    plt.show()

    break