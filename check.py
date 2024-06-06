import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from aiart import Generator  # Import your Generator class from the model file

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the input image
input_image = Image.open("check.png")

# Define the transform to normalize the image
transform = transforms.Compose([
    transforms.ToTensor()  # Convert to tensor
])
input_tensor = transform(input_image)

# Compute mean and standard deviation
mean = torch.mean(input_tensor, dim=(1, 2))
std = torch.std(input_tensor, dim=(1, 2))

epsilon = 1e-6  # Small epsilon value
transform = transforms.Compose([
    transforms.Normalize(mean, std + epsilon)  # Normalize pixel values
])

# Normalize the input tensor
input_tensor_normalized = transform(input_tensor)

print("Mean shape:", mean.shape)
print("Standard deviation shape:", std.shape)

# Check if input tensor has an alpha channel and remove it if present
if input_tensor_normalized.shape[0] > 3:
    input_tensor_normalized = input_tensor_normalized[:3, :, :]

# Load the trained CycleGAN model
G_AB = Generator(3, 3, n_residual_blocks=9).to(device)  # Instantiate your Generator model
G_AB.load_state_dict(torch.load("/Users/atharva/Desktop/venv1/G_AB.pth", map_location=device))  # Load the trained weights
G_AB.eval()

# Perform image translation
with torch.no_grad():
    translated_image = G_AB(input_tensor_normalized.unsqueeze(0).to(device))

# Postprocess the translated image
translated_image = translated_image.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy array
translated_image = (translated_image + 1) / 2  # Denormalize pixel values
translated_image = (translated_image * 255).astype(np.uint8)  # Convert to uint8

# Convert numpy array to PIL image
translated_image = Image.fromarray(translated_image.transpose(1, 2, 0))

# Display the original and translated images
input_image.show(title="Original Traditional Art")
translated_image.show(title="Translated Modern Visual Context")
