# Minha Própria Implementação de GAN

Minha própria implementação de GAN com Pytorch.
Adaptada para usar GPU se você tiver uma placa gráfica NVIDIA.

A arquitetura escolhida para este projeto foi WGAN-GP.

## Uma breve explicação sobre WGAN:

WGAN (Wasserstein Generative Adversarial Network) é uma variante das redes adversariais generativas (GANs) que usa a distância de Wasserstein para medir a diferença entre a distribuição de dados reais e a distribuição gerada. Esta abordagem foi introduzida para resolver os problemas de convergência das GANs tradicionais, proporcionando um treinamento mais estável.

## Uma breve explicação sobre WGAN-GP:

WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty) é uma extensão da WGAN. O "GP" refere-se à penalidade de gradiente, um termo adicionado à função de perda para garantir que os gradientes sejam limitados e para evitar o problema de "mode collapse" frequentemente visto nas GANs. Esta penalidade força o discriminador a ter gradientes de norma próxima a 1, o que ajuda a garantir um treinamento mais suave e estável.

## Comparação entre as duas arquiteturas:

Ao comparar WGAN e WGAN-GP, a principal razão para optar por WGAN-GP é a introdução da Penalidade de Gradiente. No WGAN original, para garantir que a função crítica (também conhecida como discriminador) seja Lipschitz contínua, é preciso cortar os pesos, um processo conhecido como "weight clipping". No entanto, este método pode levar a problemas de otimização e redes que não são expressivas o suficiente.

WGAN-GP, por outro lado, introduz uma penalidade no gradiente da função crítica, garantindo que seja Lipschitz contínua sem a necessidade de cortar os pesos. Isso resulta em um treinamento mais estável, evitando os problemas associados ao "weight clipping". Em resumo, ao usar WGAN-GP em vez de WGAN, beneficia-se de um treinamento mais suave e consistente e de um modelo potencialmente mais expressivo.

## O que é "Lipschitz contínuo"?

Uma função é dita Lipschitz contínua se existir uma constante "L" (conhecida como constante de Lipschitz) tal que a diferença absoluta entre os valores da função em quaisquer dois pontos seja limitada pelo produto dessa constante e a distância entre esses pontos.

A condição de Lipschitz garante que a função não tenha oscilações muito abruptas, significando que não é muito "íngreme" em qualquer intervalo.

## Por que WGAN-GP?

Treinar GANs tradicionais (Generative Adversarial Networks) muitas vezes enfrenta problemas de estabilidade, que podem resultar em geração de imagens de baixa qualidade ou falhas de convergência do modelo. WGAN, com sua abordagem baseada na distância de Wasserstein, trouxe melhorias significativas em termos de estabilidade do treinamento. No entanto, a necessidade de "weight clipping" no WGAN original pode limitar a capacidade da rede.

Então surge o WGAN-GP, que introduz a penalidade de gradiente. Ao substituir o "weight clipping" pela penalidade de gradiente, WGAN-GP garante que a função crítica permaneça Lipschitz contínua sem restringir o poder expressivo da rede. Esta modificação melhora a estabilidade do treinamento e ajuda a produzir saídas de maior qualidade.

Se você está procurando uma abordagem de GAN que combine treinamento robusto com a capacidade de produzir saídas de alta qualidade, WGAN-GP é, sem dúvida, uma excelente escolha.

# Como usar este projeto

Após clonar este repositório, crie um ambiente virtual python e instale as dependências diretamente do arquivo requirements.txt.

Antes de iniciar o treinamento da GAN, é necessário criar uma pasta chamada "dataset" na raiz do projeto. Por padrão, o pytorch exige pelo menos um rótulo para imagens. 
Para fazer isso, basta criar uma pasta dentro da pasta "dataset", com um nome de sua escolha (pode ser qualquer nome). Em seguida, basta copiar as imagens que deseja usar no treinamento para essa pasta.

O projeto tem um arquivo chamado "parameters.json" onde você pode configurar os parâmetros de treinamento da GAN e os caminhos do projeto.

O arquivo já está configurado com parâmetros otimizados para treinar esta arquitetura, mas, se desejar, sinta-se à vontade para alterá-lo de acordo com sua preferência.

Para iniciar o treinamento, execute o arquivo "main.py".

Após iniciar a execução, você poderá acompanhar o treinamento através do terminal e através do arquivo de log que será criado no diretório configurado em "parameters.json".

Além disso, no final de cada época, você poderá visualizar uma amostra das imagens geradas dentro do diretório configurado, na pasta "samples".

# Como usar GPU

Esta implementação suporta o uso de GPU.
Eu usei CUDA 12+ e, para instalar o torch, usei este comando (após criar o ambiente virtual e instalar as dependências):

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121```
