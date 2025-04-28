import random
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm
from pipeline.movqgan import decode
from pipeline.show import show_images

def training_loop(nvp, loader, n_epochs, optim, device, model, inverse_normalize_fn, display=True, store_path="/content/checkpoint/project_nvp.ckpt"):
    scheduler = lr_scheduler.ExponentialLR(optim, gamma=0.98)
    best_loss = float("inf")
    half_loader = len(loader) // 2

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        if epoch >= 19:
            scheduler.step()
            current_lr = optim.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1} start: LR *= 0.98 → {current_lr:.6f}\n")

        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            x = batch.to(device)

            optim.zero_grad()
            loss = -nvp.log_prob(x).mean()
            loss.backward()
            optim.step()

            if epoch >= 19 and step == half_loader:
                scheduler.step()
                current_lr = optim.param_groups[0]['lr']
                print(f"\nEpoch {epoch+1} mid:   LR *= 0.98 → {current_lr:.6f}\n")

            epoch_loss += loss.item() * len(x) / len(loader.dataset)
            if step % 10 == 0:
              print(step, loss.item())
        # Display images generated at this epoch
        if display:
            nvp.eval()
            with torch.no_grad():
                generated_codes = nvp.sample(16)
                show_images(decode(model, inverse_normalize_fn(generated_codes)), f"Latent codes generated at epoch {epoch + 1}")
            nvp.train()

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"
        store_path2 = "/content/checkpoint/project_nvp2.ckpt"
        torch.save(nvp.state_dict(), store_path2)

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(nvp.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)
