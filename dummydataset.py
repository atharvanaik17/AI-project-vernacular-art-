import os
from PIL import Image
import numpy as np

# Create directories
os.makedirs('data/train/A', exist_ok=True)
os.makedirs('data/train/B', exist_ok=True)
os.makedirs('data/test/A', exist_ok=True)
os.makedirs('data/test/B', exist_ok=True)

# Create dummy images
for i in range(10):
    img_A = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    img_A.save(f'data/train/A/{i}.png')
    
    img_B = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    img_B.save(f'data/train/B/{i}.png')

    img_test_A = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    img_test_A.save(f'data/test/A/{i}.png')
    
    img_test_B = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    img_test_B.save(f'data/test/B/{i}.png')
