import random
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm
from models.GAN.model import generate_new_codes
from models.GAN.model import compute_gradient_penalty
from pipeline.movqgan import decode
from pipeline.show import show_images

def training_loop(generator, critic, loader, n_epochs, optim_g, optim_c, device, model, inverse_normalize_fn, n_critic=5, n_sched=19, display=True, store_path="/content/checkpoints/gan1.ckpt", store_path2='/content/checkpoints/gan2.ckpt'):
    scheduler_g = lr_scheduler.ExponentialLR(optim_g, gamma=0.98)
    scheduler_c = lr_scheduler.ExponentialLR(optim_c, gamma=0.98)
    best_loss = float("inf")
    latent_dim = 128
    half_loader = len(loader) // 2

    for epoch in tqdm(range(n_epochs), desc="Training progress", colour="#00ff00"):
        if epoch >= n_sched:
            scheduler_g.step()
            scheduler_c.step()
            current_lr = optim_g.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1} start: LR *= 0.98 → {current_lr:.6f}\n")

        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            real_samples = batch.to(device)
            batch_size = real_samples.size(0)

            critic_loss = 0.0
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_samples = generator(noise)
            for _ in range(n_critic):

                real_pred = critic(real_samples)
                fake_pred = critic(fake_samples.detach())
                gp = compute_gradient_penalty(critic, real_samples, fake_samples.detach(), device)
                c_loss = fake_pred.mean() - real_pred.mean() + 10 * gp
                optim_c.zero_grad()
                c_loss.backward()
                optim_c.step()
                critic_loss += c_loss.item() / 5

            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_samples = generator(noise)
            fake_pred = critic(fake_samples)
            g_loss = -fake_pred.mean()
            optim_g.zero_grad()
            g_loss.backward()
            optim_g.step()

            if epoch >= n_sched and step == half_loader:
                scheduler_g.step()
                scheduler_c.step()
                current_lr = optim_g.param_groups[0]['lr']
                print(f"\nEpoch {epoch+1} mid:   LR *= 0.98 → {current_lr:.6f}\n")

            epoch_loss += (critic_loss + g_loss.item()) * batch_size / len(loader.dataset)
            if step % 10 == 0:
                print(f"Step {step}, Critic Loss: {critic_loss:.4f}, Generator Loss: {g_loss.item():.4f}")

        if display:
            generated_codes = generate_new_codes(generator, latent_dim, c=4, h=22, w=22, device=device)
            show_images(decode(model, inverse_normalize_fn(generated_codes)), f"Latent codes generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"
        torch.save({
            'generator': generator.state_dict(),
            'critic': critic.state_dict()
        }, store_path2)

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save({
                'generator': generator.state_dict(),
                'critic': critic.state_dict()
            }, store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)
