# check_env.py
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def check_environment():
    """
    A simple script to check if the PyTorch environment is set up correctly
    for our Machine Unlearning project.
    """
    print("--- Environment Check for Machine Unlearning Project ---")
    print("\n[1] Checking PyTorch and Torchvision versions...")
    
    try:
        print(f"    - PyTorch Version: {torch.__version__}")
        print(f"    - Torchvision Version: {torchvision.__version__}")
        print("    ✅ PyTorch and Torchvision are installed.")
    except ImportError as e:
        print(f"    ❌ Error: {e}")
        print("    Please install PyTorch and Torchvision.")
        return

    print("\n[2] Checking for GPU (CUDA) availability...")
    
    is_cuda_available = torch.cuda.is_available()
    if is_cuda_available:
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        print(f"    ✅ Success! CUDA is available.")
        print(f"    - Number of GPUs: {gpu_count}")
        print(f"    - Current GPU Name: {gpu_name}")
        device = torch.device("cuda")
    else:
        print("    ⚠️ Warning: CUDA not available. The code will run on CPU.")
        print("    - Training will be significantly slower.")
        device = torch.device("cpu")

    print("\n[3] Checking data loading with CIFAR-10...")
    
    try:
        # Define a simple transform
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Try to download CIFAR-10 training set
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                  shuffle=True, num_workers=2)
        
        print("    ✅ CIFAR-10 dataset downloaded and DataLoader created successfully.")
        # Check if we can get a batch of data
        dataiter = iter(trainloader)
        images, labels = next(dataiter)
        print(f"    - Sample batch loaded: Images shape {images.shape}, Labels shape {labels.shape}")

    except Exception as e:
        print(f"    ❌ Error during data loading: {e}")
        print("    - Check your internet connection or file permissions for the './data' directory.")
        return

    print("\n[4] Checking model loading (ResNet-18) and forward pass...")
    
    try:
        # Load a pretrained ResNet-18 model
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.to(device) # Move model to the selected device (GPU or CPU)
        model.eval() # Set to evaluation mode
        
        print("    ✅ ResNet-18 model loaded successfully.")

        # Take the sample batch from the previous step and move it to the device
        images = images.to(device)
        
        # Perform a forward pass
        with torch.no_grad(): # We don't need to calculate gradients for this test
            outputs = model(images)
        
        print("    ✅ Forward pass completed successfully.")
        print(f"    - Output tensor shape: {outputs.shape}")

    except Exception as e:
        print(f"    ❌ Error during model operations: {e}")
        return

    print("\n--- Check Complete ---")
    print("🎉 Your environment seems to be ready for the project! 🎉")

if __name__ == '__main__':
    check_environment()
