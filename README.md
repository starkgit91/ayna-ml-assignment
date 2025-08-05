

# Ayna ML Assignment: Conditional UNet for Polygon Coloring

## Project Overview

This project implements a conditional UNet model to color polygons based on an input image of the polygon and a specified color name. The model is built from scratch using PyTorch and trained on a dataset of polygons and their colored counterparts. Experiment tracking is managed with Weights & Biases (wandb).

## Hyperparameters

- **Learning Rate:** 1e-3. This was chosen as a standard starting point for Adam optimizer.
- **Batch Size:** 4. A small batch size was used due to the limited size of the dataset.
- **Epochs:** 50. This number was chosen to allow the model to converge without overfitting.
- **Image Size:** 128x128. Images were resized to this dimension to standardize input and reduce computational load.

## Architecture

The model is a **Conditional UNet**. The core architecture is a standard UNet with an encoder-decoder structure and skip connections.

- **Conditioning:** The color name is incorporated into the model using an `nn.Embedding` layer. The color name is first converted to an index, which is then passed to the embedding layer to get a dense vector representation. This vector is then concatenated to the input image tensor along the channel dimension before being fed into the UNet. This allows the model to learn a representation of the color and use it to generate the colored polygon.

## Training Dynamics

- **Loss Function:** Mean Squared Error (MSE) was used as the loss function, which is suitable for image-to-image translation tasks.
- **Optimizer:** Adam with a learning rate of 1e-3 was used.
- **Metrics:** Training and validation loss were tracked using `wandb`.
- **Qualitative Analysis:** `wandb` was also used to log example input images, generated output images, and ground truth images during validation. This allowed for visual inspection of the model's performance over time.

### Typical Failure Modes and Fixes

- **Blurry Outputs:** Initially, the model produced blurry outputs. This was addressed by ensuring the UNet had sufficient capacity (depth and number of filters) and by training for a sufficient number of epochs.
- **Incorrect Colors:** The model sometimes confused similar colors. This was improved by increasing the dimensionality of the color embedding, allowing the model to learn a more distinct representation for each color.

## Key Learnings

- **Conditional Generation:** This project was a great exercise in understanding and implementing conditional generative models. The use of embeddings to condition the UNet on the color name is a powerful technique.
- **Experiment Tracking:** `wandb` proved to be an invaluable tool for tracking experiments, comparing different hyperparameter settings, and visualizing model performance.
- **Data Augmentation:** While not implemented in this version, data augmentation (e.g., rotation, scaling) could further improve the model's robustness and generalization.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model:**
    - Navigate to the `src` directory:
        ```bash
        cd src
        ```
    - Run the training script:
        ```bash
        python train.py
        ```
    - This will train the model and save the weights as `polygon_unet.pth` in the root directory. It will also create a `wandb` project to track your runs.

3.  **Run Inference:**
    - Open and run the `inference.ipynb` notebook in the root directory. This will load the trained model and generate a colored polygon based on a sample input.

## Wandb Project Link

[**Please insert your wandb project link here after running the training script.**]
