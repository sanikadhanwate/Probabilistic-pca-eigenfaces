import numpy as np
import matplotlib.pyplot as plt


IMG_SIZE = 24 #24x24
DATA_FILE = "eigenfaces.npy"


def load_data():
    X = np.load(DATA_FILE)
    if X.ndim == 3:     # flattening stored images to (n, 576)
        X = X.reshape(X.shape[0], -1)

    return X.astype(float)


# (a)  Train the p-PCA model by computing the MLE for this dataset using the equations above for a latent dimension d. 
def train_ppca(X, d):

    n, m = X.shape
    mu = np.mean(X, axis=0) 
    X_centered = X - mu  

    S = (X_centered.T @ X_centered) / n 

    eigvals, eigvecs = np.linalg.eigh(S) 

    order = np.argsort(eigvals)[::-1] 

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    sigma2 = np.mean(eigvals[d:]) 

    Uq = eigvecs[:, :d]
    Lambda_q = eigvals[:d]

    W = Uq @ np.diag(
        np.sqrt(np.maximum(Lambda_q - sigma2, 0))
    )

    return W, mu, sigma2


def encode(X, W, mu, sigma2):

    d = W.shape[1]

    M = W.T @ W + sigma2 * np.eye(d)

    rhs = W.T @ (X - mu).T

    Z = np.linalg.solve(M, rhs).T

    return Z


def decode(Z, W, mu):

    return Z @ W.T + mu



def save_collage(images, rows, cols, title, filename):

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols, rows)
    )

    axes = np.array(axes).reshape(-1)

    for ax, image in zip(axes, images):

        ax.imshow(
            image.reshape(IMG_SIZE, IMG_SIZE),
            cmap="gray"
        )

        ax.axis("off")

    fig.suptitle(title)

    plt.tight_layout()

    plt.savefig(filename, dpi=200)

    plt.close()


def main():

    np.random.seed(0)

    X = load_data()

    n, m = X.shape

    print("Dataset shape:", X.shape)


    # (a) Train p-PCA

    models = {}

    for d in [2, 16, 32, 64]:

        W, mu, sigma2 = train_ppca(X, d)

        models[d] = (W, mu, sigma2)

        print(
            f"d={d}, "
            f"W shape={W.shape}, "
            f"sigma^2={sigma2:.6f}"
        )

    # (b) Scatter plot for d = 2

    W, mu, sigma2 = models[2]

    Z = encode(X, W, mu, sigma2)

    plt.figure(figsize=(6, 6))

    plt.scatter(
        Z[:, 0],
        Z[:, 1],
        s=8,
        alpha=0.5
    )

    plt.xlabel("z1")
    plt.ylabel("z2")

    plt.title("Latent Feature Vectors for d = 2")

    plt.gca().set_aspect("equal")

    plt.tight_layout()

    plt.savefig(
        "latent_scatter_d2.png",
        dpi=200
    )

    plt.close()


    # (c) Reconstruction for d = 16, 32, 64
    
    indices = np.random.choice(
        n,
        size=25,
        replace=False
    )

    X_selected = X[indices]

    for d in [16, 32, 64]:

        W, mu, sigma2 = models[d]
        
        Z_hat = encode(
            X_selected,
            W,
            mu,
            sigma2
        )

        X_reconstructed = decode(
            Z_hat,
            W,
            mu
        )

        save_collage(
            X_reconstructed,
            5,
            5,
            f"Reconstructions (d = {d})",
            f"reconstructions_d{d}.png"
        )

    # (d) Generate 100 new faces

    d = 64

    W, mu, sigma2 = models[d]

    # Sample z ~ N(0, I)
    Z_new = np.random.randn(100, d)

    # Compute E[x | z] = Wz + mu
    X_generated = decode(
        Z_new,
        W,
        mu
    )

    save_collage(
        X_generated,
        10,
        10,
        "Generated Faces",
        "generated_faces.png"
    )

    # (e) Latent perturbation for d = 16

    W, mu, sigma2 = models[16]

    index = np.random.choice(n)

    x = X[index:index+1]

    z_hat = encode(
        x,
        W,
        mu,
        sigma2
    )[0]

    dimensions = [0, 1, 2, 3, 4]

    perturbations = np.linspace(
        -0.5,
        0.5,
        10
    )

    images = []

    for dim in dimensions:

        for amount in perturbations:

            z_new = z_hat.copy()

            z_new[dim] += amount

            x_new = decode(
                z_new.reshape(1, -1),
                W,
                mu
            )[0]

            images.append(x_new)


    save_collage(
        images,
        5,
        10,
        "Latent Space Perturbations (d = 16)",
        "perturbations_d16.png"
    )


    print("\nDone! All figures have been saved.")


if __name__ == "__main__":
    main()