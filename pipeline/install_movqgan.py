import os
import sys
import git

def clone_movqgan_repo():
    if not os.path.exists('MoVQGAN'):
        print("Cloning MoVQGAN repository...")
        git.Repo.clone_from('https://github.com/ai-forever/MoVQGAN.git', 'MoVQGAN')
    else:
        print("MoVQGAN repository already exists.")

def load_model(model_name='67M', pretrained=True, device='cuda'):
    sys.path.append(os.path.join(os.getcwd(), 'MoVQGAN'))
    from movqgan import get_movqgan_model

    # Загружаем модель
    model = get_movqgan_model(model_name, pretrained=pretrained, device=device)
    return model

def main():
    clone_movqgan_repo()  
    model = load_model()
    return model
