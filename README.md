# Q-Lane-Follower

## Resumo

O desenvolvimento de sistemas de controle para veículos autônomos apresenta desafios complexos relacionados à dinâmica e segurança veicular. Este projeto propõe a implementação de um controlador autônomo de manutenção de faixa (lane-following) baseado no algoritmo de Aprendizado por Reforço Q-Learning. O problema foi modelado como um Processo de Decisão de Markov (MDP) com espaços de estados e ações discretizados, sendo o treinamento e a validação realizados no simulador CARLA. Para otimizar a convergência e contornar a inércia do veículo em aceleração, foi aplicada uma estratégia de treinamento em duas fases, transferindo o conhecimento de cenários de baixa para dinâmicas de alta velocidade. Os resultados demonstram que o agente internalizou com sucesso a política de navegação, mantendo-se dentro dos limites da via, embora as oscilações laterais observadas na trajetória validem as limitações inerentes da abordagem tabular.

## Demonstração

![Demonstração do Agente](evaluation_clip.gif)

## Arquitetura do Código

O projeto é estruturado de forma modular para separar as diferentes responsabilidades da pipeline de Aprendizado por Reforço.

-   `src/main.py`: Ponto de entrada principal para iniciar o processo de treinamento.
-   `src/training/train.py`: Contém o loop de treinamento, orquestrando a interação entre o agente e o ambiente ao longo de múltiplos episódios. Também gerencia a persistência do modelo (Q-table).
-   `src/environment/carla_env.py`: Um wrapper para o simulador CARLA que abstrai sua complexidade, fornecendo uma interface padrão similar à do Gym, com métodos `reset()` e `step()`. Gerencia o veículo, sensores e o estado do mundo.
-   `src/agent/q_learning_agent.py`: Implementa o algoritmo de Q-learning tabular. É responsável por escolher ações (exploração vs. explotação) e atualizar a Q-table com base nas experiências.
-   `src/state/state_discretizer.py`: Converte as observações contínuas do ambiente (ex: desvio da faixa, erro de direção) em um espaço de estados discreto, necessário para a abordagem de Q-learning tabular.
-   `src/reward/reward_function.py`: Define a lógica de recompensa. Calcula um valor de recompensa com base no desempenho do agente, como permanecer na faixa, manter a velocidade e evitar colisões.
-   `src/config.py`: Arquivo centralizado para todos os hiperparâmetros, configurações do ambiente, mapeamento de ações e limites para a discretização de estados.
-   `src/evaluate.py`: Script para avaliar visualmente um agente treinado, gravar vídeos da sua performance e gerar gráficos da trajetória.

## Funcionalidades

- **Q-Learning Tabular**: Implementação de um algoritmo clássico de aprendizado por reforço para espaços de estado-ação discretos.
- **Design Modular**: Componentes facilmente substituíveis para o ambiente, agente, representação de estado e função de recompensa.
- **Otimizado para Performance**: Inclui otimizações para treinamento mais rápido, como modo assíncrono, renderização desabilitada e cálculos eficientes.
- **Logging Detalhado**: Registra dados detalhados por episódio (recompensa, velocidade, erros) em um arquivo CSV para análise de treinamento.
- **Suíte de Avaliação**: Um script de avaliação (`src/evaluate.py`) que fornece feedback visual, grava vídeos e gera mapas de trajetória.

## Como Usar

### Pré-requisitos

- Python 3.7+
- Simulador CARLA (versão 0.9.11 ou superior)
- Dependências listadas em `requirements.txt`

### Instalação

```python
1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/Q-lane-follower.git
   cd Q-lane-follower
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

### Treinamento

Para iniciar o treinamento, execute:
```bash
python src/main.py
```
As configurações de treinamento podem ser ajustadas em `src/config.py`. A Q-table resultante será salva em `assets/q_table.npy`.

### Avaliação

Para avaliar um modelo treinado:
```bash
python src/evaluate.py
```
Este comando iniciará uma janela Pygame exibindo a performance do agente e salvará os resultados (vídeo, log de dados, mapa de trajetória).

## Otimizações de Performance

O wrapper do ambiente CARLA inclui otimizações para acelerar o treinamento. A tabela abaixo compara o desempenho com e sem essas melhorias.

### Melhorias Implementadas

- **Modo Assíncrono**: Remove esperas síncronas entre os passos da simulação.
- **Câmera Desabilitada**: Elimina o processamento de imagem desnecessário para o agente.
- **FPS Limitado**: Reduz a carga de CPU através da configuração `max_substep_delta_time`.
- **Limpeza de Atores**: Sensores e veículos órfãos são removidos a cada episódio para evitar vazamento de memória.

### Benchmark de Performance

Execute o benchmark para testar as otimizações:
```bash
python benchmark.py
```
