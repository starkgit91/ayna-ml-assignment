Conditional Image Generation with a UNet
Project Overview
This project addresses the Ayna ML Assignment by implementing a deep learning model to color polygons. The model, a Conditional UNet built from scratch in PyTorch, takes two inputs: a black-and-white image of a polygon outline and a target color name. It then generates an image of the polygon filled with the specified color.

The development process was iterative, involving several stages of identifying model weaknesses and implementing targeted solutions to improve performance. This report details the final, successful architecture and chronicles the key learnings from each stage of development.

Final Architecture and Hyperparameters
After significant experimentation, the final, most effective configuration was determined to be:

Model: A deep Conditional UNet with 5 downsampling/upsampling blocks to capture complex geometric features.

Conditioning Mechanism: An nn.Embedding layer converts the input color name into a 32-dimensional vector, which is then reshaped and concatenated to the input image tensor along the channel axis.

Loss Function: A Balanced Weighted L1 Loss. This was the key to achieving both a solid color fill and sharp, accurate boundaries. The loss function applies a 5x higher weight to pixels inside the target polygon, forcing the model to prioritize getting the fill correct without distorting the shape.

Data Augmentation: Paired data augmentation (random rotation and horizontal flipping) is applied to both the input and target images simultaneously within the Dataset class. This was crucial for helping the model generalize to different shapes and orientations.

Final Hyperparameters:
Optimizer: Adam

Learning Rate: 2e-4

Epochs: 200

Batch Size: 4

Image Size: 128x128

LR Scheduler: StepLR (gamma=0.5, step_size=50)

Loss Weight (Polygon vs. Background): 5:1

Training Dynamics & Key Learnings: An Iterative Journey
The final model was the result of a systematic process of identifying and fixing issues. Each step provided valuable insights into the model's behavior.

1. Initial Baseline: Blurry Shapes
The first model used a standard MSELoss. While it quickly learned to place color in the correct location, the output was blurry and indistinct, failing to capture the polygon's shape. This is a known characteristic of MSE loss, which tends to favor averaged, low-frequency solutions.

Insight: MSELoss is often suboptimal for tasks requiring sharp image generation.

<img src="https://storage.googleapis.com/gemini-prod/images/image_4a272d.png" width="400"/>

2. Seeking Sharpness: L1 Loss and a Deeper Model
To combat the blurriness, the loss function was switched to L1Loss, which penalizes absolute differences and encourages sharper edges. The model was also made deeper to increase its capacity. This resulted in a much sharper, more solid color fill, but the model still failed to learn the geometry, producing a generic blob.

Insight: While L1Loss improves sharpness, a deeper model is necessary to learn complex features.

<img src="https://storage.googleapis.com/gemini-prod/images/image_4a88c7.png" width="400"/>

3. The Generalization Problem: Failure on Complex Shapes
The model performed reasonably on simple shapes like triangles but failed completely on more complex ones like stars, defaulting to a simple circle. This indicated a failure to generalize, likely due to the small dataset.

Insight: The model was memorizing simple shapes, not learning the underlying concept of "filling a polygon." Data augmentation was needed.

<img src="https://storage.googleapis.com/gemini-prod/images/image_4a8cab.png" width="400"/>

4. The Augmentation Bug: Unpaired Transforms
The first attempt at data augmentation led to catastrophic model failure, producing blank or noisy images. The root cause was a classic bug: the random transformations applied to the input image were different from those applied to the target image. The model was being shown a crooked outline and told to produce a straight filled shape, leading to complete confusion.

Insight: Data augmentation for paired image-to-image tasks must be applied identically to both the input and target.

<img src="https://storage.googleapis.com/gemini-prod/images/image_5ae229.png" width="400"/>

5. A Step Forward: Paired Augmentation and Faint Colors
After fixing the bug by moving augmentation logic into the Dataset class to ensure paired transforms, the model began to correctly learn the shapes of complex polygons. However, the output colors were faint and washed out. The loss from the large white background was "drowning out" the loss from the smaller, more important polygon area.

Insight: The model was correctly learning shape but lacked a strong signal to produce vibrant color inside.

<img src="https://storage.googleapis.com/gemini-prod/images/image_66be70.png" width="400"/>

6. The Final Solution: Balanced Weighted Loss
To solve the faint color issue, a weighted loss function was introduced. This function applies a higher penalty to errors on pixels inside the polygon. An initial high weight of 10x successfully produced a solid fill but caused the shapes to become distorted and "puffy."

By fine-tuning this weight down to a 5x penalty, the perfect balance was achieved. The model was strongly encouraged to create a solid, high-contrast fill while still being constrained enough to preserve the precise, sharp boundaries of the input polygon. This final configuration generalized well across all shapes in the dataset.

Final Insight: A balanced, weighted loss function is a powerful technique to focus a model's attention on the most important regions of an image, leading to high-fidelity results.

<img src="https://storage.googleapis.com/gemini-prod/images/image_67280e.png" width="400"/>

How to Run the Project
1. Installation
Clone the repository and install the required dependencies. You may need to install kornia if you experiment with the SSIM loss function.

git clone <your-repo-link>
cd ayna-ml-assignment
pip install -r requirements.txt

2. Train the Model
Navigate to the src directory and run the training script. This will train the final, stable model and save the weights as polygon_unet.pth in the project's root directory.

cd src
python train.py

3. Run Inference
Once training is complete, open and run the inference.ipynb notebook from the root directory. It will load the trained model and generate a sample output, which you can customize by changing the input image path and color name.

Wandb Project Link
The full training history, including all experiments, metrics, and sample outputs discussed in this report, can be viewed at the following Weights & Biases project:

[<-- Insert Your Public Wandb Project Link Here -->]
