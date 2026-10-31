# Relatório: IA para o jogo da cobrinha

## Resumo

**Atualização:** o esquema de ladrilhos (seção abaixo) deu um jogador mais rápido que o ciclo e igualmente seguro: o A* com garantia faz 397 pontos em cerca de 15 800 passos, contra 39 600 do ciclo. A rede treinada ainda imita o ciclo; imitar o A* seguro é o próximo passo.

Comparamos quatro jogadores no `snake-game` v2.0.0, tabuleiro 20x20. A **imitação** foi escolhida: uma rede 13→32→32→3 treinada para copiar o jogador do ciclo hamiltoniano chega a 393,8 pontos em média, de um máximo de 397 (tabuleiro cheio). Os dois caminhos que aprendem sem professor ficaram bem atrás.

## Como medimos

- 30 partidas novas por jogador, com as mesmas sementes de comida (100000 a 100029), nunca usadas no treino.
- Limite de 60 000 passos por partida. Uma partida também termina por falta de comida: 600 passos sem comer contam como morte.
- Pontuação máxima: 397, que é o tabuleiro de 400 células com a cobra inicial de 3. Encher o tabuleiro conta como vitória.
- Cada número vem de **um** treino e de **um** conjunto de 30 partidas. O treino não é determinístico: ao retreinar a imitação para salvar o modelo, a média foi 397,0 (0 mortes) em vez de 393,8. Não repetimos com várias sementes.

## Resultados

| Jogador | Média | Mín | Máx | Mortes |
|---|---|---|---|---|
| Especialista (ciclo hamiltoniano com atalhos seguros) | 397,0 | 397 | 397 | 0/30 |
| **Imitação** (13 features, MLP 13→32→32→3, Adam) | **393,8** (outro treino: 397,0) | 301 | 397 | 1/30 (outro treino: 0/30) |
| Heurística evoluída por GA, com termo de enrolamento, 20 mil passos na evolução | 263,5 | 173 | 369 | 30/30 |
| Heurística evoluída por GA, sem o termo, 20 mil passos na evolução | 251,7 | 122 | 385 | 30/30 |
| DQN com features invariantes e heurística como shaping | 38,5 | 14 | 74 | 30/30 |
| DQN com as features do ciclo | 7,3 | 0 | 24 | 30/30 |

Primeira versão, com outro especialista e outro conjunto de teste (100 partidas): o especialista BFS fazia 64,1 em média e a rede, com 11 features locais e SGD, 27,5.

## Decisões e justificativas

### 1. Trocar o especialista BFS pelo ciclo hamiltoniano

O BFS leva a cobra até a comida pelo caminho mais curto, mas ela acaba se prendendo (média 64). O ciclo hamiltoniano percorre todas as células numa ordem fixa. Com atalhos que só avançam e nunca passam da cauda, o corpo continua em ordem de ciclo e a cobra nunca se prende. Venceu as 30 partidas.
**Consequência:** o professor passou a ser perfeito, então o teto da imitação subiu de 64 para 397.

### 2. Dar à rede features relativas ao ciclo

Por ação (reto, direita, esquerda), a rede recebe: se está bloqueada, se o ciclo permite o movimento, e a distância restante até a comida pelo ciclo (dividida por 400). Mais a direção atual em one-hot, no total de 13 valores. A decisão do especialista (menor distância entre os movimentos permitidos) fica uma função simples dessas features, fácil de aprender: a loss chega a 0,0001.
**Consequência:** com 11 features locais (sem o ciclo) a imitação ficava em 27,5.

### 3. Escolher a imitação

- Atinge 98% da pontuação do especialista (393,8 de 397), contra 66% do melhor caminho sem professor (heurística com GA, 263,5) e 10% do melhor DQN (38,5).
- Treina em cerca de 1 minuto. O DQN gasta 300 mil passos e não passa de 40 pontos.
- É supervisionada, então usa o `nn-cpp` sem precisar de componentes de aprendizado por reforço que ele não tem.

### 4. O que não adotamos, e por quê

- **DQN:** aprende a não morrer de imediato, mas não a planejar. Com gamma 0,95 o horizonte é de uns 20 passos, e a comida costuma estar a mais de 100 passos no ciclo. Features invariantes, shaping pela heurística, penalidade de morte proporcional ao tamanho e penalidade de laços elevaram de 7,3 para 38,5, ainda longe. Gamma maior, mais passos e Double DQN não foram testados.
- **Algoritmo genético sobre a heurística:** chegou a cerca de 260 pontos, mas toda partida termina em morte. A heurística é invariante a rotação e a escala e tem pesos que dependem do tamanho da cobra, o que era o objetivo, mas não dá a garantia que o ciclo dá.
- **Termo de enrolamento** (pontos do tabuleiro com número de enrolamento igual a 1 em relação à curva fechada formada pelo corpo): melhorou a média em 3% a 5%, na mesma direção nas duas rodadas, mas com uma evolução por configuração e notas que variam de 173 a 369, não dá para afirmar que o efeito é real. O cálculo foi verificado contra uma soma de ângulos por força bruta, sem divergências em 2200 estados.

### 5. Dependências como submódulos git

`snake-game` (tag v2.0.0) e `nn-cpp` (commit fixo) entram em `deps/`. Fixam as versões, e o projeto compila sem ter os outros dois repositórios ao lado. Tentamos antes o FetchContent e abandonamos a pedido.

## Limitações

- As features do ciclo valem para o tabuleiro 20x20 e para o ciclo escolhido. A imitação **não é invariante** a rotação nem a escala, ao contrário da heurística. Para outro tamanho é preciso gerar outro ciclo e treinar de novo.
- A rede aprende a regra do especialista, não descobre uma estratégia. O que ela faz de melhor é o que o especialista já faz.
- A única morte da imitação (301 pontos) aconteceu numa partida longa. Não investigamos a causa.
- Um treino e 30 partidas por jogador. As diferenças pequenas (por exemplo, entre as heurísticas) não são conclusivas.
- Não há testes automatizados no projeto. O app imprime a comparação.

## Tentativa: caminhos mais curtos até a comida

Pergunta: a rede pode, depois de imitar o ciclo, andar em zigue-zague e minimizar a distância até a comida? Primeiro medimos o ritmo do professor (média de 20 partidas):

| Trecho | Passos por comida |
|---|---|
| até 25 pontos | 26 |
| 25 a 100 | 36 |
| 100 a 200 | 74 |
| 200 a 300 | 120 |
| 300 a 397 | 146 |

Uma partida completa leva cerca de 37 mil passos. Testamos duas formas de encurtar:

1. **Menor rota segura dentro da regra do ciclo.** Programação dinâmica sobre os índices do ciclo, aceitando só rotas que avançam no ciclo e ficam antes da cauda. Segura (0 mortes em 30 partidas), mas **sem ganho**: 37 434 passos por partida contra 37 500. Só o começo melhora (26 para 23 passos por comida até 25 pontos). Com a cobra longa o trecho livre à frente é curto, e a comida cai com frequência numa célula pulada por um atalho anterior, que fica atrás da cauda no ciclo; a cobra espera a cauda andar. O limite vem da própria garantia do ciclo. Houve também um erro nosso no caminho: sem tratar a comida atrás da cauda, a rota atravessava o corpo e a cobra morria (média 35,5). Revertemos esse código, porque não trazia ganho.
2. **Menor rota por BFS com teste de segurança** (`experiments/bfs_safe.cpp`). Segue o BFS até a comida se uma simulação mostrar que a cauda continua alcançável depois de comer; senão, persegue a cauda. Em 10 partidas: **392,4 pontos em média com 23 204 passos** (38% menos que o ciclo), mas as 10 terminam sem encher o tabuleiro: ficam entre 387 e 395 e morrem de fome (600 passos sem comer) quando restam poucas células, porque o teste recusa sempre o caminho até a comida. Relaxar o teste piora muito: aceitar o caminho com mais de 360 de comprimento deu 362,7 pontos e colisões; aceitar após 150 passos sem comer deu 194,4 e colisões.

**Conclusão.** Há um custo claro: o ciclo garante 397 pontos em 37 mil passos; o BFS com teste de cauda faz cerca de 392 em 23 mil, sem garantia e com um final travado. Não treinamos a rede para imitar o BFS: ela precisaria de features que codifiquem a rota e o resultado da simulação (como as features do ciclo codificam o professor atual), e o final travado teria que ser resolvido antes. A abordagem escolhida continua sendo a imitação do ciclo.

## Esquema de ladrilhos: ciclos hamiltonianos dinâmicos

O A* sozinho não é seguro: ele quebra a ordem de ciclo do corpo e o final fica travado (ver acima). Um esquema de ladrilhos 2x2 (setas por célula, como nas figuras que você trouxe) resolve isso. O tabuleiro vira uma grade de 10x10 ladrilhos. Qualquer árvore geradora dessa grade define um ciclo hamiltoniano das 400 células: ao longo de uma aresta da árvore o ciclo atravessa a borda do ladrilho nas duas faixas; dentro de um ladrilho ele usa o par de células de cada lado que não tem aresta da árvore.

Cada passo do corpo impõe uma condição à árvore: passar entre ladrilhos obriga a aresta a estar na árvore; andar dentro de um ladrilho obriga a aresta do lado correspondente a ficar fora. O corpo está sobre **algum** ciclo exatamente quando existe uma árvore geradora que respeita todas as condições. Isso se decide com union-find: as arestas obrigatórias não podem fechar laço, e as arestas não proibidas precisam conectar a grade (`src/tiles.cpp`). Se um movimento deixa o corpo sobre algum ciclo, a cobra sempre pode seguir esse ciclo até a cauda, e os modos podem ser trocados a qualquer passo.

Validação: 50 árvores aleatórias dão ciclo único com grau 2 em todas as células, e todo prefixo do ciclo é aceito como corpo (`experiments/safe_modes.cpp`).

Modos (`src/modes.cpp`), todos restritos aos movimentos que mantêm essa propriedade:

- **A\*:** o movimento seguro com menor caminho real até a comida.
- **Atalhos:** o movimento seguro mais próximo da comida ao longo do ciclo atual.
- **Ciclo:** segue o ciclo.

12 partidas, mesmas sementes, 397 = tabuleiro cheio:

| Estratégia | Pontos | Passos por partida | Mortes |
|---|---|---|---|
| Ciclo puro | 397 | 39 574 | 0/12 |
| Atalhos | 397 | 15 943 | 0/12 |
| A\* | 397 | 15 792 | 0/12 |
| A\* até 100, depois atalhos | 397 | 15 110 | 0/12 |
| Atalhos até 100, depois A\* | 397 | 17 022 | 0/12 |
| A\* até 100, depois ciclo | 397 | 26 023 | 0/12 |
| A\* até 300, depois ciclo | 397 | 16 504 | 0/12 |

Com a garantia, o A\* enche o tabuleiro em 60% menos passos que o ciclo e sem nenhuma morte, o que o BFS com teste de cauda não conseguia (392 pontos e final travado). A tentativa anterior de atalhos "clássicos" (pular células do ciclo fixo) quebra a propriedade, porque o corpo deixa de ser um trecho contínuo do ciclo; dentro do esquema, atalhos viram um critério de escolha entre os movimentos seguros.

### Seletor de modo aprendido

`apps/selector.cpp`: 9748 estados amostrados em 48 partidas com troca aleatória de modo. Em cada estado, cada modo é mantido por 100 passos e seguido por 150 passos de A\*; o rótulo é o modo que comeu mais, só trocando do A\* se ganhar por pelo menos uma comida (A\* 6251, atalhos 2597, ciclo 900). Rede 8→32→32→3 sobre 8 features sem escala (tamanho, distâncias até a comida e à cauda, passos sem comer, comida antes da cauda, movimentos seguros).

O seletor aprendido escolhe A\* em 98% dos passos e termina em 15 792 passos, igual ao A\* puro. A loss fica em 0,83, ou seja, os rótulos são ruidosos: a diferença entre modos em 250 passos é de poucas comidas. A melhor troca fixa (A\* até 100, depois atalhos) ganha 4% sobre o A\*, com 12 partidas e sem repetir, então não dá para afirmar que é real.

**Conclusão.** Escolher entre os três modos rende pouco; o ganho veio da garantia de segurança, não da escolha. O A\* seguro é o melhor jogador em velocidade e dá 397 sempre.

## Rede salva e modo de assistir

`compare imitation` grava o checkpoint (`imitation.ckpt`) e avalia a cópia recarregada, para provar que o arquivo guarda a rede treinada. A rede de `models/imitation.ckpt` enche o tabuleiro (397) nas sementes 1 e 3, em cerca de 36 mil passos. Para assistir:

```bash
./build/watch models/imitation.ckpt [semente=1] [ms por quadro=50] [desenhar a cada N passos=1]
```

Partidas completas passam de 35 mil passos; use, por exemplo, `watch models/imitation.ckpt 1 20 50`. Ctrl+C interrompe e devolve o cursor.

## Próximos passos

1. Treinar a rede para imitar o A* seguro. As features por ação seriam: se o movimento mantém o corpo sobre algum ciclo (teste de árvore), distância até a comida depois do movimento e bloqueio. O professor passa a ser 2,5 vezes mais rápido.
2. Investigar a partida que morreu e, se for erro de generalização, treinar com mais jogos ou com amostragem das partes longas.
3. Se invariância importar, tentar imitar o especialista com features egocêntricas e verificar quanto se perde dos 393,8 a 397.
