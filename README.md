# snake-ai

Agentes que jogam o [snake-game](https://github.com/MiguelMochizuki/snake-game), treinados com a rede neural do [nn-cpp](https://github.com/MiguelMochizuki/nn-cpp). A abordagem escolhida é a **imitação**: uma rede pequena aprende a copiar um jogador que segue um ciclo hamiltoniano. Em 30 partidas novas ela faz em média 393,8 pontos de um máximo de 397.

Os resultados e as decisões estão em [docs/REPORT.md](docs/REPORT.md).

## Build

```bash
git clone --recursive git@github.com:MiguelMochizuki/snake-ai.git
cd snake-ai
cmake -S . -B build && cmake --build build -j
```

Os dois projetos entram como submódulos em `deps/`: `snake-game` na tag v2.0.0 e `nn-cpp` num commit fixo. Só a lógica do jogo é compilada, sem o código de terminal.

## Uso

```bash
./build/compare expert        # jogador do ciclo hamiltoniano
./build/compare imitation     # treina a rede por imitação, grava imitation.ckpt e avalia
./build/watch models/imitation.ckpt [semente] [ms] [N]   # assiste a rede jogar no terminal
./build/compare dqn           # DQN com as features do ciclo
./build/compare dqn-ego GENOME   # DQN com features invariantes e shaping
./build/compare ga GENOME     # avalia a heurística evoluída
./build/evolve GENOME [passos] [nocoil]   # algoritmo genético
```

`GENOME` é o arquivo de pesos que o `evolve` grava. `passos` é o limite de passos por partida durante a evolução (padrão 4000). `nocoil` zera o termo de enrolamento, para comparação.

Para ver saída durante o treino, use `stdbuf -oL`, porque o stdout fica em buffer quando redirecionado.

## Código

| Arquivo | O que faz |
|---|---|
| `include/snake_env.hpp`, `src/snake_env.cpp` | Ambiente sem terminal sobre o `GameState`, com ações relativas e as duas famílias de features |
| `include/cycle.hpp`, `src/cycle.cpp` | Ciclo hamiltoniano 20x20 |
| `include/expert.hpp`, `src/expert.cpp` | Jogador do ciclo com atalhos seguros, o professor da imitação |
| `include/heuristic.hpp`, `src/heuristic.cpp` | Heurística invariante com pesos dependentes do tamanho |
| `apps/compare.cpp` | Avalia e treina expert, imitação, DQN e heurística |
| `apps/evolve.cpp` | Algoritmo genético |
| `apps/watch.cpp` | Modo de assistir: carrega um checkpoint e joga no terminal |
| `include/policy_net.hpp` | Arquitetura da rede e escolha gulosa, compartilhadas |
| `models/imitation.ckpt` | Rede treinada por imitação |
