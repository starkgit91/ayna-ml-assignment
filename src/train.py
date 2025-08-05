

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import wandb
# from dataset import PolygonDataset
# from model import ConditionalUNet
# import os
# from torch.optim.lr_scheduler import StepLR
# # NEW: Import torchvision for Perceptual Loss
# import torchvision.models as models

# # --- Hyperparameters ---
# LEARNING_RATE = 1e-4 
# BATCH_SIZE = 4
# EPOCHS = 200 # Perceptual loss can take time to converge
# IMAGE_SIZE = 128
# LR_STEP_SIZE = 50
# LR_GAMMA = 0.5

# # --- NEW: VGG Perceptual Loss ---
# # This loss function ensures the generated image is perceptually similar to the target.
# class VGGPerceptualLoss(torch.nn.Module):
#     def __init__(self, resize=True):
#         super(VGGPerceptualLoss, self).__init__()
#         # Load the VGG19 model pre-trained on ImageNet
#         blocks = []
#         blocks.append(models.vgg19(pretrained=True).features[:4].eval())
#         blocks.append(models.vgg19(pretrained=True).features[4:9].eval())
#         blocks.append(models.vgg19(pretrained=True).features[9:18].eval())
#         blocks.append(models.vgg19(pretrained=True).features[18:27].eval())
#         for bl in blocks:
#             for p in bl.parameters():
#                 p.requires_grad = False
#         self.blocks = torch.nn.ModuleList(blocks)
#         self.transform = torch.nn.functional.interpolate
#         self.resize = resize
#         # The VGG network was trained on images normalized with these values
#         self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#         self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

#     def forward(self, input, target):
#         # Normalize input and target to match VGG's expected input
#         input = (input - self.mean) / self.std
#         target = (target - self.mean) / self.std
#         if self.resize:
#             input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
#             target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        
#         loss = 0.0
#         x = input
#         y = target
#         # Calculate L1 loss between feature maps at different depths of the VGG network
#         for block in self.blocks:
#             x = block(x)
#             y = block(y)
#             loss += torch.nn.functional.l1_loss(x, y)
#         return loss

# def train(model, device, train_loader, optimizer, criterion_l1, criterion_perceptual):
#     model.train()
#     for batch_idx, (data, color, target) in enumerate(train_loader):
#         data, color, target = data.to(device), color.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data, color)

#         # --- FINAL: Combined L1 + Perceptual Loss ---
#         # 1. Calculate L1 Loss (for pixel-level accuracy)
#         loss_l1 = criterion_l1(output, target)

#         # 2. Calculate Perceptual Loss (for high-level feature similarity)
#         # We need to convert images from [-1, 1] to [0, 1] for the VGG model
#         output_0_1 = output * 0.5 + 0.5
#         target_0_1 = target * 0.5 + 0.5
#         loss_p = criterion_perceptual(output_0_1, target_0_1)

#         # 3. Combine the losses.
#         loss = loss_l1 + 0.05 * loss_p
#         # --- End of Final Code ---

#         loss.backward()
#         optimizer.step()
#         wandb.log({"train_loss": loss.item(), "l1_loss": loss_l1.item(), "perceptual_loss": loss_p.item()})

# def validate(model, device, val_loader):
#     model.eval()
#     with torch.no_grad():
#         data, color, target = next(iter(val_loader))
#         data, color, target = data.to(device), color.to(device), target.to(device)
#         output = model(data, color)
        
#         data_vis = data[0].cpu() * 0.5 + 0.5
#         output_vis = output[0].cpu() * 0.5 + 0.5
#         target_vis = target[0].cpu() * 0.5 + 0.5
        
#         wandb.log({
#             "examples": [
#                 wandb.Image(data_vis, caption="Input"),
#                 wandb.Image(output_vis, caption="Generated Output"),
#                 wandb.Image(target_vis, caption="Ground Truth")
#             ]
#         })

# def main():
#     wandb.init(project="ayna-ml-assignment")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     SRC_DIR = os.path.dirname(os.path.abspath(__file__))
#     PROJECT_ROOT = os.path.dirname(SRC_DIR)

#     train_dir = os.path.join(PROJECT_ROOT, 'dataset', 'training')
#     val_dir = os.path.join(PROJECT_ROOT, 'dataset', 'validation')
#     model_save_path = os.path.join(PROJECT_ROOT, 'polygon_unet.pth')

#     train_dataset = PolygonDataset(root_dir=train_dir, image_size=IMAGE_SIZE, is_train=True)
#     val_dataset = PolygonDataset(root_dir=val_dir, image_size=IMAGE_SIZE, is_train=False)

#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
#     num_colors = len(train_dataset.colors)

#     model = ConditionalUNet(n_channels=3, n_classes=3, num_colors=num_colors).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
#     # Define our two loss functions
#     criterion_l1 = nn.L1Loss()
#     criterion_perceptual = VGGPerceptualLoss().to(device)
    
#     scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    
#     wandb.watch(model)
#     wandb.config.update({
#         "learning_rate": LEARNING_RATE,
#         "epochs": EPOCHS,
#         "batch_size": BATCH_SIZE,
#         "loss_function": "Combined L1+Perceptual",
#     })

#     for epoch in range(1, EPOCHS + 1):
#         train(model, device, train_loader, optimizer, criterion_l1, criterion_perceptual)
#         validate(model, device, val_loader)
#         scheduler.step()
#         print(f"Epoch {epoch}/{EPOCHS} completed. Current LR: {scheduler.get_last_lr()[0]}")

#     torch.save(model.state_dict(), model_save_path)
#     print(f"Training finished and model saved to {model_save_path}")

# if __name__ == '__main__':
#     main()


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from dataset import PolygonDataset
from model import ConditionalUNet
import os
from torch.optim.lr_scheduler import StepLR

# --- Hyperparameters ---
LEARNING_RATE = 2e-4 
BATCH_SIZE = 4
EPOCHS = 200 
IMAGE_SIZE = 128
LR_STEP_SIZE = 50
LR_GAMMA = 0.5


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, color, target) in enumerate(train_loader):
        data, color, target = data.to(device), color.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, color)

        # --- FINAL: Reverted to Balanced Weighted L1 Loss for Stability and Generalization ---
        # This is the most robust method we found. It encourages a solid fill
        # without the instability of more complex perceptual losses.
        
        # 1. Get the element-wise loss without reduction
        loss_elementwise = criterion(output, target)

        # 2. Create a weight map from the target image.
        with torch.no_grad():
            # Create a mask for the polygon area (non-white pixels)
            mask = (target.sum(dim=1, keepdim=True) < 2.9).float()
            # Create weights: 5 for polygon pixels, 1 for background.
            weights = mask * 4.0 + 1.0

        # 3. Apply weights and calculate the final mean loss
        weighted_loss = loss_elementwise * weights
        loss = torch.mean(weighted_loss)
        # --- End of Final Code ---

        loss.backward()
        optimizer.step()
        wandb.log({"train_loss": loss.item()})

def validate(model, device, val_loader):
    model.eval()
    with torch.no_grad():
        data, color, target = next(iter(val_loader))
        data, color, target = data.to(device), color.to(device), target.to(device)
        output = model(data, color)
        
        # Un-normalize images for correct visualization in wandb
        data_vis = data[0].cpu() * 0.5 + 0.5
        output_vis = output[0].cpu() * 0.5 + 0.5
        target_vis = target[0].cpu() * 0.5 + 0.5
        
        wandb.log({
            "examples": [
                wandb.Image(data_vis, caption="Input"),
                wandb.Image(output_vis, caption="Generated Output"),
                wandb.Image(target_vis, caption="Ground Truth")
            ]
        })

def main():
    wandb.init(project="ayna-ml-assignment")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)

    train_dir = os.path.join(PROJECT_ROOT, 'dataset', 'training')
    val_dir = os.path.join(PROJECT_ROOT, 'dataset', 'validation')
    model_save_path = os.path.join(PROJECT_ROOT, 'polygon_unet.pth')

    train_dataset = PolygonDataset(root_dir=train_dir, image_size=IMAGE_SIZE, is_train=True)
    val_dataset = PolygonDataset(root_dir=val_dir, image_size=IMAGE_SIZE, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_colors = len(train_dataset.colors)

    model = ConditionalUNet(n_channels=3, n_classes=3, num_colors=num_colors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Use L1Loss with no reduction to get per-pixel loss for weighting
    criterion = nn.L1Loss(reduction='none')
    
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    
    wandb.watch(model)
    wandb.config.update({
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "loss_function": "Balanced Weighted L1Loss",
        "scheduler": "StepLR",
    })

    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, criterion)
        validate(model, device, val_loader)
        scheduler.step()
        print(f"Epoch {epoch}/{EPOCHS} completed. Current LR: {scheduler.get_last_lr()[0]}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Training finished and model saved to {model_save_path}")

if __name__ == '__main__':
    main()
