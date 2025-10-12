import random
import gymnasium as gym
import numpy as np

ambiente = gym.make('Taxi-v3')
alfa = 0.9                             # taxa de aprendizado que varia entre 0 e 1 para ditar o quanto de informação sera sobrescrita
gamma = 0.95                      # fator de desconto que varia entre 0 e 1 para ditar o quanto de importancia sera dada as recompensas futuras
epsilon = 1.0                        # taxa de exploracao que varia entre 0 e 1 para ditar o quanto o agente ira explorar o ambiente, usa-se 1 quando não
#                                      se tem conhecimento do ambiente e 0 para quando se tem conhecimento do ambiente
epsilon_decay = 0.9995          # fator de decaimento que varia entre 0 e 1 para ditar o quanto a taxa de exploracao ira diminuir a cada episodio, minimizando o valor de epsilon
epsilon_min = 0.1                   # taxa de exploracao minima que varia entre 0 e 1 para ditar o valor minimo que a taxa de exploracao pode atingir, para se ainda ter aleatoriedade
num_episodes = 10000          # numero de episodios que o agente ira treinar
max_steps = 100            # numero maximo de etapas que o agente pode executar em cada episodio
q_table = np.zeros((ambiente.observation_space.n, ambiente.action_space.n))  # inicializa a tabela Q com zeros
# a tabela sera uma matriz 5x5, ou seja, 25 posições, 5 posições do cliente, 4 posições do hotel, resultando em (25*5*4) = 500 estados possiveis

def escolher_acao(estado):
    if random.uniform(0, 1) < epsilon:
        # Explora: escolhe uma ação aleatória
        return ambiente.action_space.sample()
    else:
        # Exploita: escolhe a melhor ação baseada na tabela Q
        return np.argmax(q_table[estado, :])

for episodio in range(num_episodes):
   estado, _ = ambiente.reset() #reinicia o ambiente

   done = False

   for step in range(max_steps):
       acao = escolher_acao(estado)

       next_state, reward, done, truncated, _ = ambiente.step(acao) #realiza a ação no ambiente

       old_value = q_table[estado,acao] #armazena o valor de estado atual para a tabela antes dele ser atualizado para o próximo
       next_max = np.max(q_table[next_state,:])

       q_table[estado,acao] = (1-alfa) * old_value + alfa * (reward+gamma*next_max)

       estado = next_state      
    
       epsilon = max(epsilon_min, epsilon * epsilon_decay)

ambiente = gym.make('Taxi-v3', render_mode='human')

for episodio in range(5):
    estado, _ = ambiente.reset()
    done = False
    print('Episodio:',episodio)

    for step in range(max_steps):
        ambiente.render()
        acao = np.argmax(q_table[estado, :])
        next_state, reward, done, truncated, _ = ambiente.step(acao)
        estado = next_state
        if done or truncated:
            print('Episodio finalizado em {} passos'.format(step+1))
            break