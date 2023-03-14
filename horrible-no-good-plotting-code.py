def plot(f,z):
    global coefficient_data
    x = np.linspace(0,np.max(z),20)
    y = np.linspace(0,np.max(z),20)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            Z[i,j] = f(torch.tensor(X[i,j]),torch.tensor(Y[i,j]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(X,Y,Z)
    
    
    indices = np.array([(i, j) for (i, j, v) in coefficient_data])
    values = np.array([v for (i, j, v) in coefficient_data])

    # extract the coordinates from z using the indices
    x = z[indices[:, 0]]
    y = z[indices[:, 1]]
    z = values
    ax.scatter(x, y, z, color='orange')

    # Make the plot interactive with mouse rotation
    ax.view_init(elev=10, azim=20)
    ax.dist = 10
    ax.mouse_init()
    plt.show()
    
