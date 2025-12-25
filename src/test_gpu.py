import torch
import numpy as np
print(np.__version__)

def test_gpu():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.get_device_name(0))

        # 进一步测试是否能在 GPU 上运算
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.mm(x, x)
            print("GPU test passed: matrix multiplication completed!")
        except Exception as e:
            print("GPU test failed:", e)
    else:
        print("No GPU detected or CUDA not working.")

if __name__ == "__main__":
    test_gpu()
