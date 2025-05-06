import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.hdc_utils import create_pose_matrix
from plot_config import mpl

# Constants
HDC_DIM = 10000  # Dimensionality of Hyperdimensional Computing vectors
SCALES = np.logspace(0, 3, 7, base=2) - 1  # Scaling values for experiments
N_STEPS = 100  # Number of timesteps
KERNELS = ['sinc', 'gaussian', 'triangular']  # Kernels to test
FPE_METHOD = 'sinusoid'  # Fractional Power Expansion mode
K_IDX = 50  # Reference timestep to compare similarity
SAVE_PATH = "."  # Directory to save plots
FONT_SIZE = 24  # Global font size for plots
plt.rcParams.update({'font.size': FONT_SIZE})


def process_kernel(kernel, axes, index):
    """
    Args:
        kernel: The kernel function or name being processed. Determines specific details of the pose matrix computation.
        axes: The array of matplotlib axes objects used for plotting similarity results.
        index: The index of the current plot/axis within the axes array.
    """
    print(f"Processing kernel: {kernel}...")
    for scale in SCALES:
        # Create pose matrix and compute similarity
        pose_matrix = create_pose_matrix(N_STEPS, scale, HDC_DIM, seed=0, kernel=kernel,
                                         fpe_method=FPE_METHOD)
        similarity = cosine_similarity(pose_matrix[K_IDX:K_IDX + 1, :], pose_matrix)
        # Plot similarity on the corresponding axis
        axes[index].plot(similarity[0, :], label=f"β={scale:.2f}")

    # Configure subplot titles, labels, and reference line
    axes[index].set_title(kernel.capitalize())
    axes[index].set_xlabel('Time step')
    axes[index].set_ylabel('Cosine similarity')
    axes[index].axvline(x=K_IDX, color='black', linestyle='--')  # Highlight K_IDX
    if index == len(KERNELS) - 1:  # Place legend outside for the last subplot
        axes[index].legend(loc='center left', bbox_to_anchor=(1, 0.5))


def main_app():
    """
    Main application logic for visualizing cosine similarity across kernels and scales.
    """
    print("Starting plot configuration...")
    # Create subplots for kernels
    figure, axes = plt.subplots(1, len(KERNELS), figsize=(15, 5))
    print("Subplots created.")

    # Process each kernel
    for i, kernel in enumerate(KERNELS):
        process_kernel(kernel, axes, i)

    # Adjust layout, save the plot, and display
    plt.tight_layout()
    output_file = f"{SAVE_PATH}/images/similarity_timesteps.pdf"
    plt.savefig(output_file)
    print(f"Plot saved to: {output_file}")
    plt.show()
    print("Plot displayed successfully.")


if __name__ == '__main__':
    main_app()
