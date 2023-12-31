import torch
import torchvision.utils as vutils
from torch.autograd import Variable, grad
from tqdm import tqdm
import datetime

def log_progresso(log_file, message):
    """Registra uma mensagem com um carimbo de data/hora no arquivo de log especificado."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"

    with open(log_file, "a") as file:
        file.write(log_entry)

def save_checkpoint(epoch, generator, discriminator, optim_g, optim_d, losses_g, losses_d, path="checkpoint.pth"):
    """Salva o estado atual dos modelos e otimizadores no caminho especificado."""
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optim_g.state_dict(),
        'optimizer_d_state_dict': optim_d.state_dict(),
        'losses_g': losses_g,
        'losses_d': losses_d
    }, path)

def gradient_penalty(discriminator, real_data, fake_data, device):
    """Calcula a penalidade do gradiente para o discriminador. Utilizado no WGAN-GP."""
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
    alpha = alpha.expand_as(real_data)

    interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
    interpolated = Variable(interpolated, requires_grad=True).to(device)
    interpolated_prob = discriminator(interpolated)

    gradients = grad(outputs=interpolated_prob, inputs=interpolated,
                     grad_outputs=torch.ones(interpolated_prob.size()).to(device),
                     create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return ((gradients_norm - 1) ** 2).mean()

def train_model(
        generator, 
        discriminator, 
        weights_path,
        n_critic, 
        sample_size,
        sample_dir,
        optim_g, 
        optim_d, 
        data_loader, 
        device, 
        z_dim, 
        num_epochs, 
        last_epoch,
        save_model_at,
        log_dir,
        losses_g=[], 
        losses_d=[]
    ):
    """Treina o gerador e o discriminador, logando o progresso e salvando pontos de controle."""
    lambda_gp = 10  # Valor padrão para WGAN-GP

    fixed_noise = Variable(torch.randn(sample_size, z_dim, 1, 1)).to(device)
    
    for epoch in range(last_epoch, num_epochs + 1):
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, data in pbar:
            images, _ = data
            images = Variable(images).to(device)

            # Atualização do Discriminador
            for _ in range(n_critic):
                z = Variable(torch.randn(images.size(0), z_dim, 1, 1)).to(device)
                fake_images = generator(z)
                
                real_prob = discriminator(images)
                fake_prob = discriminator(fake_images.detach())
                
                real_loss = -torch.mean(real_prob)
                fake_loss = torch.mean(fake_prob)
                
                gp = gradient_penalty(discriminator, images, fake_images, device)
                
                d_loss = real_loss + fake_loss + lambda_gp * gp

                discriminator.zero_grad()
                d_loss.backward()
                optim_d.step()

            # Atualização do Gerador
            z = Variable(torch.randn(images.size(0), z_dim, 1, 1)).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images).squeeze()
            g_loss = -torch.mean(outputs)

            generator.zero_grad()
            g_loss.backward()
            optim_g.step()

            pbar.set_description(f'Epoch {epoch}/{num_epochs}, g_loss: {g_loss.data}, d_loss: {d_loss.data}')
        
        log_progresso(f"{log_dir}/trainning.log", f'Epoch {epoch}/{num_epochs}, g_loss: {g_loss.data}, d_loss: {d_loss.data}')

        losses_g.append(g_loss.data.cpu())
        losses_d.append(d_loss.data.cpu())

        # Salva imagens geradas para inspeção visual
        vutils.save_image(generator(fixed_noise).data, sample_dir + '/fake_samples_epoch_%06d.jpeg' % (epoch), normalize=True)

        # Salva pontos de controle a intervalos especificados
        if (epoch) % save_model_at == 0:
            save_checkpoint(epoch, generator, discriminator, optim_g, optim_d, losses_g, losses_d, f"{weights_path}/checkpoint.pth")

    return losses_g, losses_d
