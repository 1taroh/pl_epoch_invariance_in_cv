import torch
import pytorch_lightning as pl

def main():
    pl.seed_everything(42)
    print(torch.rand(1)) # tensor([0.8823])
    print(torch.rand(1)) # tensor([0.9150])

if __name__ == "__main__":
    main()
