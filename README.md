# PPO-Lane-Follower

Controlador autônomo de manutenção de faixa (*lane following*) baseado no algoritmo de *Deep Reinforcement Learning* **Proximal Policy Optimization (PPO)**, treinado e validado no simulador **CARLA** (v0.9.16).

## Resumo

O desenvolvimento de sistemas de controle para veículos autônomos apresenta desafios complexos relacionados à dinâmica e à segurança veicular. Este trabalho propõe a implementação de um controlador autônomo de manutenção de faixa baseado no algoritmo de Aprendizado por Reforço **Proximal Policy Optimization (PPO)**. O problema foi modelado como um **Processo de Decisão de Markov (MDP)** com espaços de estados e ações **contínuos**, sendo o treinamento e a validação realizados no simulador CARLA.

Para tornar o modelo mais aderente ao comportamento dinâmico do veículo, adotou-se uma estratégia de representação contínua do espaço de ações, incluindo uma variável unificada para aceleração e frenagem (*throttle_brake*). O estado observado pelo agente é composto por três variáveis contínuas: desvio lateral (*lane offset*), erro de orientação (*heading error*) e velocidade escalar.

Os resultados demonstram que o agente foi capaz de aprender uma política de controle estável, mantendo o veículo dentro dos limites da faixa. Em comparação com abordagens anteriores baseadas em **Q-Learning**, observou-se uma redução significativa nas oscilações laterais da trajetória, evidenciando a superior capacidade do PPO em lidar com espaços contínuos. No entanto, tal melhoria vem acompanhada de um aumento no custo computacional e no tempo de treinamento, inerentes aos métodos de otimização de políticas.

## Demonstração

![Demonstração do Agente](evaluation_pygame.mp4)

*Vídeo completo disponível em: https://1drv.ms/f/c/bacbbaced9b1b624/IgA0dNcJfbVKRJdr9YuhmrDKAeHKatfGBpOT6XESSAQN-hA?e=cYo3by*

## Arquitetura do Código

O projeto é estruturado de forma modular para separar as diferentes responsabilidades da pipeline de Aprendizado por Reforço.

```
src/
├── main.py                  # Ponto de entrada principal
├── config.py                # Hiperparâmetros e configurações centralizadas
├── evaluate.py              # Script de avaliação visual com Pygame
├── agent/
│   └── ppo_agent.py         # Implementação NumPy pura do PPO (ator-crítico)
├── environment/
│   ├── carla_env.py         # Wrapper de baixo nível para o simulador CARLA
│   └── gym_wrapper.py       # Interface Gymnasium para integração com Stable-Baselines3
├── training/
│   ├── train.py             # Loop de treinamento com Stable-Baselines3
│   └── callbacks.py         # Callbacks personalizados (métricas, taxa de sucesso, etc.)
├── reward/
│   └── reward_function.py   # Função de recompensa com reward shaping
├── state/
│   └── state_discretizer.py # Discretização de estados (legado do Q-Learning)
└── utils/
    ├── logger.py            # Configuração de logging
    └── data_logger.py       # Logger de dados de avaliação em CSV
```

### Descrição dos Principais Scripts

- **`src/main.py`**: Ponto de entrada principal. Carrega a configuração e inicia o treinamento do agente PPO chamando a função `train()`.

- **`src/config.py`**: Arquivo centralizado com todos os hiperparâmetros do PPO (taxa de aprendizado, *clip range*, arquitetura da rede neural, etc.), configurações do ambiente CARLA (mapa, modo síncrono, *delta time*) e parâmetros da função de recompensa.

- **`src/training/train.py`**: Orquestra o ciclo completo de treinamento utilizando o **Stable-Baselines3**. Cria o ambiente via `CarlaGymWrapper`, instancia o modelo PPO com política `MlpPolicy` e rede neural de arquitetura `[128, 128]`, e executa o treinamento por um número configurável de *timesteps* (padrão: 15M). Inclui *callbacks* para salvamento de *checkpoints*, registro de métricas por episódio e monitoramento da taxa de sucesso.

- **`src/evaluate.py`**: Script de avaliação visual que carrega um modelo PPO treinado e executa o agente no simulador com renderização Pygame. Gera:
  - Vídeo da performance do agente (`evaluation_pygame.mp4`)
  - Log de dados passo a passo em CSV (velocidade, *lane offset*, *heading error*, comandos de controle)
  - Imagens do mapa com a trajetória percorrida (`evaluation_result_map.png` e `evaluation_result_trajectory.png`)
  - Gravação replay do CARLA (`evaluation_carla.log`)

- **`src/environment/carla_env.py`**: *Wrapper* de baixo nível para o simulador CARLA. Gerencia a conexão com o servidor, *spawn* do veículo (Tesla Model 3), sensores (câmera e colisão), cálculo das observações (*lane offset*, *heading error*, velocidade) e aplicação dos comandos de controle (esterçamento, aceleração/frenagem). Inclui otimizações de performance como modo assíncrono, renderização desabilitada e limpeza eficiente de atores.

- **`src/environment/gym_wrapper.py`**: Adapta o `CarlaEnvironment` para a interface **Gymnasium**, permitindo integração direta com o Stable-Baselines3. Implementa os métodos `reset()` e `step()` com cálculo da recompensa via `reward_function.py`.

- **`src/agent/ppo_agent.py`**: Implementação do algoritmo PPO em **NumPy puro** (sem dependência do Stable-Baselines3), com redes neurais *actor-critic* de duas camadas ocultas (64 neurônios cada). Inclui *rollout buffer* com cálculo de **Generalized Advantage Estimation (GAE)**, função objetivo com *clipping* e atualização em mini-batches com múltiplas épocas.

- **`src/reward/reward_function.py`**: Define a função de recompensa com *reward shaping*, priorizando centralização na faixa, alinhamento angular e progressão em velocidade, com penalidades severas para comportamentos inseguros (saída de pista, colisão, veículo preso).

## Funcionalidades

- **PPO com Espaço de Ações Contínuo**: Controle suave e contínuo de esterçamento (*steer*) e aceleração/frenagem unificada (*throttle_brake*), eliminando as oscilações laterais características de abordagens discretas.
- **Arquitetura Actor-Critic**: Rede neural com duas camadas ocultas de 128 neurônios (configurável) para aproximação da política e da função de valor.
- **Treinamento com Stable-Baselines3**: Integração robusta com a biblioteca SB3, incluindo *TensorBoard logging*, *checkpoints* periódicos e *callbacks* personalizados.
- **Modularidade**: Componentes facilmente substituíveis para ambiente, agente, representação de estado e função de recompensa.
- **Otimizado para Performance**: Modo síncrono com *fixed_delta_seconds* configurável, *no_rendering mode* para treinamento acelerado, limpeza eficiente de atores e sensores.
- **Suíte de Avaliação Completa**: Script com renderização Pygame, gravação de vídeo, geração de mapas de trajetória e logging detalhado em CSV.
- **Implementação NumPy Pura**: Código do PPO também disponível em NumPy puro (`src/agent/ppo_agent.py`) para fins educacionais e de estudo do algoritmo.

## Hiperparâmetros do Treinamento

| Parâmetro | Valor |
|---|---|
| Algoritmo | PPO (Stable-Baselines3) |
| Política | MlpPolicy |
| Arquitetura da rede | [128, 128] |
| *Learning rate* | 3 × 10⁻⁴ |
| Fator de desconto (γ) | 0,99 |
| GAE (λ) | 0,95 |
| *Clip range* (ε) | 0,2 |
| *Entropy coefficient* | 0,01 |
| *Value function coefficient* | 0,5 |
| *Max grad norm* | 0,5 |
| *n_steps* | 4096 |
| *batch size* | 256 |
| *n_epochs* | 20 |
| *Total timesteps* | 15.000.000 |
| Distância de sucesso | 2.000 m |

## Como Usar

### Pré-requisitos

- Python 3.7+
- Simulador CARLA (versão 0.9.16)
- Dependências listadas em `requirements.txt`

### Instalação

```bash
1. Clone o repositório:
   git clone https://github.com/davidsimonmarques/PPO-lane-follower.git
   cd PPO-lane-follower

2. Instale as dependências:
   pip install -r requirements.txt
```

### Treinamento

Para iniciar o treinamento do agente PPO:

```bash
python src/main.py
```

As configurações de treinamento (hiperparâmetros do PPO, ambiente, recompensa) podem ser ajustadas em `src/config.py`. Os *checkpoints* do modelo são salvos periodicamente em `src/logs/checkpoints/` e os logs do TensorBoard em `src/logs/tensorboard/`.

Para visualizar o progresso do treinamento com TensorBoard:

```bash
tensorboard --logdir src/logs/tensorboard
```

### Avaliação

Para avaliar um modelo treinado com renderização visual:

```bash
python src/evaluate.py
```

Este comando:
1. Conecta ao CARLA e carrega o modelo PPO treinado
2. Abre uma janela Pygame com visão do motorista (terceira pessoa) e visão superior (*bird view*)
3. Exibe um HUD com informações em tempo real (velocidade, *lane offset*, *heading error*, comandos de controle)
4. Gera um vídeo MP4 da performance (`evaluation_pygame.mp4`)
5. Salva um log CSV com dados passo a passo
6. Gera imagens do mapa com a trajetória percorrida

O modelo utilizado na avaliação pode ser configurado alterando `self.model_path` na classe `EvaluationConfig` em `src/evaluate.py`.

## Resultados

O agente PPO foi treinado por aproximadamente **6,8 milhões de *timesteps*** (5.000 episódios) e demonstrou:

- **Redução significativa do desvio lateral**: Estabilização com erros médios em torno de **0,1 metros** pós-convergência
- **Controle de direção suave**: Eliminação das oscilações em "zigue-zague" observadas na abordagem Q-Learning
- **Velocidade operacional**: Manutenção de regime acima de **60 km/h** durante a validação
- **Taxa de sucesso**: Atingimento consistente da distância-alvo de **2.000 metros**

### Comparação com Q-Learning

| Aspecto | Q-Learning (Tabular) | PPO (Contínuo) |
|---|---|---|
| Espaço de ações | Discreto (passos fixos) | Contínuo |
| Oscilações laterais | Significativas | Mínimas |
| Estabilidade de treinamento | Moderada | Alta |
| Custo computacional | Baixo | Alto |
| Suavidade do controle | Baixa | Alta |

## Estrutura do Espaço de Estados e Ações

### Estado (observação)

O vetor de estado normalizado é composto por:

```
s_t = [lane_offset / d_max,  heading_error / 180,  speed / v_max]
```

onde:
- `lane_offset`: desvio lateral em relação ao centro da faixa (metros)
- `heading_error`: erro de orientação em relação ao *waypoint* de referência (graus)
- `speed`: velocidade escalar do veículo (m/s)
- `d_max` = 2.0 m, `v_max` = 40/3.6 m/s

### Ação

O espaço de ações contínuo é composto por dois comandos:

- **`throttle_brake`** ([-1.0, 0.6]): Valor unificado onde positivo = aceleração, negativo = frenagem
- **`steer`** ([-1.0, 1.0]): Ângulo de esterçamento (positivo = direita, negativo = esquerda)

### Recompensa

A função de recompensa combina *reward shaping* com penalidades terminais:

- `+5.0` por permanência próxima ao centro da faixa
- `-0.05 × |heading_error|` por erro de orientação
- `+0.05` por progressão em velocidade (até a referência)
- `-200.0` por saída de pista ou colisão
- `-200.0` por veículo preso
- `+400.0` bônus ao atingir a distância-alvo

## Referências

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., & Koltun, V. (2017). *CARLA: An Open Urban Driving Simulator*. Proceedings of the 1st Annual Conference on Robot Learning.
- Marques, D. S. (2026). *Desenvolvimento de um Controlador Q-Learning para a Navegação Autônoma de um Veículo Seguidor de Faixa*.
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para mais detalhes.