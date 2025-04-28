import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm
from models.DDPM.model import generate_new_images
from pipeline.movqgan import decode
from pipeline.show import show_images

def training_loop(ddpm, loader, n_epochs, optim, device, model, display=True, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    scheduler = lr_scheduler.ExponentialLR(optim, gamma=0.98)
    best_loss = float("inf")
    n_steps = ddpm.n_steps
    half_loader = len(loader) // 2

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        if epoch >= 19:
            scheduler.step()
            current_lr = optim.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1} start: LR *= 0.98 → {current_lr:.6f}\n")

        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            x0 = batch.to(device)

            eta = torch.randn(x0.shape).to(device) 
            # t = random.randint(0, 999)
            t = random.randint(0, 50)

            noisy_imgs = ddpm(x0, t, eta) 


            # Getting model estimation of noise based on the images and the time-step

            time_tensor = torch.full((x0.shape[0],), t, dtype=torch.long, device=device)

            eta_theta = ddpm.backward(noisy_imgs, time_tensor)

            loss = F.mse_loss(eta, eta_theta) 
            optim.zero_grad()
            loss.backward()
            optim.step()

            if epoch >= 19 and step == half_loader:
                scheduler.step()
                current_lr = optim.param_groups[0]['lr']
                print(f"\nEpoch {epoch+1} mid:   LR *= 0.98 → {current_lr:.6f}\n")

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)
            if step % 10 == 0:
              print(step, loss.item(), t)

        if display:
            show_images(decode(model, generate_new_images(ddpm, c=4, h=22, w=22, device=device)), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"
        store_path2 = "/content/drive/MyDrive/model/project_ddpm_regular.ckpt"
        torch.save(ddpm.state_dict(), store_path2)

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)
