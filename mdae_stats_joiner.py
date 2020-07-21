def joiner(parameter_list):
    pass
    ############# From past script
    # plotting the mse train
    # setting x and y axis range
    # plotting the mse train
    plt.plot(dimensions, mse_train, label="mse_train")
    plt.plot(dimensions, mse_test, label="mse_test")
    plt.xlabel("Encoding dimension")
    plt.ylabel("Reconstruction error (MSE)")
    # showing legend
    plt.legend()
    plt.savefig("reconstruction_error_mse.pdf")
    plt.savefig("reconstruction_error_mse.png")
    plt.close()
    # plotting the rmse train
    # setting x and y axis range
    plt.plot(dimensions, rmse_train, label="rmse_train")
    plt.plot(dimensions, rmse_test, label="rmse_test")
    plt.xlabel("Encoding dimension")
    plt.ylabel("Reconstruction error (RMSE)")
    # showing legend
    plt.legend()
    plt.savefig("reconstruction_error_rmse.pdf")
    plt.savefig("reconstruction_error_rmse.png")
    plt.close()
