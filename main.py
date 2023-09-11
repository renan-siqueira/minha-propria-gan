# Importa as bibliotecas necessárias
import json
import torch
import torch.optim as optim
import time

# Importa módulos locais
from app.generator import Generator
from app.discriminator import Discriminator
from app.training import train_model
from app.utils import (print_datetime, check_if_gpu_available, check_if_set_seed, create_dirs, weights_init, dataloader,
                       load_checkpoint, plot_losses)

def main():
    # Registra o momento de início da execução
    time_start = time.time()
    print_datetime()  # Imprime a data e hora atual

    # Carrega os parâmetros de configuração a partir do arquivo JSON
    with open('parameters.json', 'r') as f:
        params = json.load(f)

    # Verifica se uma GPU está disponível e se CUDA pode ser usado
    check_if_gpu_available()
    
    # Configura a semente para geração de números aleatórios, garantindo reprodutibilidade
    check_if_set_seed(params["seed"])
    
    # Cria os diretórios necessários listados nos parâmetros de configuração
    create_dirs(params["directories"])

    # Imprime a quantidade de vezes que o discriminador será treinado em relação ao gerador
    print('Number of repetitions for the discriminator:', params['n_critic'])

    # Configura o dispositivo para treinamento (GPU, se disponível; caso contrário, CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Inicializa o modelo do gerador e aplica inicialização de pesos
    generator = Generator(params["z_dim"], params["channels_img"], params["features_g"]).to(device)
    generator.apply(weights_init)

    # Inicializa o modelo do discriminador e aplica inicialização de pesos
    discriminator = Discriminator(params["channels_img"], params["features_d"], params["alpha"]).to(device)
    discriminator.apply(weights_init)

    # Carrega os dados para treinamento
    data_loader = dataloader(params["dataset_dir"], params["image_size"], params["batch_size"])

    # Configura os otimizadores para o gerador e o discriminador
    optim_g = optim.Adam(generator.parameters(), lr=params["lr_g"], betas=(params['g_beta_min'], params['g_beta_max']))
    optim_d = optim.Adam(discriminator.parameters(), lr=params["lr_d"], betas=(params['d_beta_min'], params['d_beta_max']))

    # Carrega o ponto de controle (checkpoint) do treinamento anterior, se houver
    last_epoch, losses_g, losses_d = load_checkpoint(f'{params["directories"][2]}/checkpoint.pth', generator, discriminator, optim_g, optim_d)

    # Inicia o processo de treinamento dos modelos
    losses_g, losses_d = train_model(
        generator=generator,
        discriminator=discriminator,
        weights_path=params["directories"][2],
        n_critic=params["n_critic"],
        sample_size=params["sample_size"],
        sample_dir=params["directories"][1],
        optim_g=optim_g,
        optim_d=optim_d,
        data_loader=data_loader,
        device=device,
        z_dim=params["z_dim"],
        num_epochs=params["num_epochs"],
        last_epoch=last_epoch,
        save_model_at=params['save_model_at'],
        log_dir=params['directories'][3],
        losses_g=losses_g,
        losses_d=losses_d,
    )

    # Calcula e imprime a duração total da execução do código
    time_end = time.time()
    time_total = (time_end - time_start) / 60
    print(f"The code took {round(time_total, 1)} minutes to execute.")
    print_datetime()  # Imprime a data e hora atual

    # Gera um gráfico das perdas do gerador e discriminador ao longo do treinamento
    plot_losses(losses_g, losses_d, save_plot_image=params["directories"][0])

# Garante que o script seja executado apenas quando chamado diretamente
if __name__ == '__main__':
    main()
