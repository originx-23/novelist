import torch

def check_torch_version():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))

if __name__ == "__main__":
    check_torch_version()