def circle(n=10000, R=1.0, dim=5):
    #x = np.random.uniform(-R, R, (n, dim))
    x = np.random.vonmises(0, R2, (n, dim))
    r = np.sum(x ** 2, axis=-1)
    x = x[r < R ** 2]

    if len(x) < n:
        x = np.concatenate((x, circle(n=n - len(x), R=R, dim=dim)), axis=0)

    return x

def donut(n=10000, R1=1.0, R2=2.0, dim=5):
    #x = np.random.uniform(-R2 , R2, (n, dim))
    x = np.random.vonmises(0, R2, (n, dim))
    r = np.sum(x ** 2, axis=-1)
    x = x[r < R2 ** 2]
    r = np.sum(x ** 2, axis=-1)
    x = x[r > R1 ** 2]

    if len(x) < n:
        x = np.concatenate((x, donut(n=n - len(x), R1=R1, R2=R2, dim=dim)), axis=0)

    return x