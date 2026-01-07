import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from mpl_toolkits.mplot3d import Axes3D


ACTIONS = ["stick", "hit"]
COLOR_PROBAS = [1 / 3, 2 / 3]
STATE_DIM = (22, 10)
STATE_ACTION_DIM = (22, 10, 2)
M = len(ACTIONS)


class Player:
    def __init__(self):
        self.cards = []
        self.value = 0
        self.bust = False

    def draw_card(self):
        card_value = random.randint(1, 10)
        card_color = np.random.choice((-1, 1), p=COLOR_PROBAS)
        card = (card_value, card_color)
        self.cards.append(card)
        self.update_value(card)
        self.update_bust()

    def first_draw(self):
        card_value = random.randint(1, 10)
        card_color = 1
        card = (card_value, card_color)
        self.cards.append(card)
        self.update_value(card)

    def update_value(self, card):
        self.value += card[0] * card[1]

    def update_bust(self):
        if self.value > 21 or self.value < 1:
            self.bust = True
        return self.bust


class Dealer(Player):
    def __init__(self):
        super().__init__()
        self.cards = []
        self.value = 0
        self.bust = False

    def get_visible_card(self):
        return self.cards[0][0]

    def run_dealer_strategy(self):
        while self.value < 17 and not self.bust:
            self.draw_card()


class Game:
    def __init__(self):
        self.player = Player()
        self.dealer = Dealer()
        self.player.first_draw()
        self.dealer.first_draw()
        self.terminal = False

    def step(self, action):
        reward = 0
        if action == 0:
            self.dealer.run_dealer_strategy()
            if self.dealer.bust:
                reward = 1
            elif self.player.value > self.dealer.value:
                reward = 1
            elif self.player.value < self.dealer.value:
                reward = -1
            self.terminal = True

        elif action == 1:
            self.player.draw_card()
            if self.player.bust:
                reward = -1
                self.terminal = True

        return reward

    def manual_game(self):
        self.reset_game()
        while not self.terminal:
            print(f"Player value: {self.player.value}")
            print(f"Dealer visible black card: {self.dealer.cards[0]}")
            input_action = int(input("Enter your action: 0 for stick, 1 for hit"))
            reward = self.step(input_action)
        print(f"Player have : {self.player.value} | Dealer have : {self.dealer.value}")
        if reward == 1:
            print("Player wins!")
        elif reward == -1:
            print("Player loses!")
        else:
            print("It's a draw!")

    def run_one_episode(self, policy):
        self.reset_game()
        episode = []
        while not self.terminal:
            state = (self.player.value, self.dealer.get_visible_card())
            action = policy.epsilon_greedy(state)
            policy.increment_N_state_action(state, action)
            policy.increment_N_state(state)
            reward = self.step(action)
            episode.append((state, action, reward))
        return episode

    def get_state(self):
        return (self.player.value, self.dealer.get_visible_card())

    def reset_game(self):
        self.player = Player()
        self.dealer = Dealer()
        self.player.first_draw()
        self.dealer.first_draw()
        self.terminal = False

    def is_terminal(self):
        return self.terminal


class Policy:
    def __init__(self, gamma, N0, nb_episode):
        self.Q_function = np.zeros(STATE_ACTION_DIM)
        self.N_state_action = np.zeros(STATE_ACTION_DIM)
        self.N_state = np.zeros(STATE_DIM)
        self.N0 = N0
        self.gamma = gamma
        self.nb_episode = nb_episode

    def compute_alpha(self, state, action):
        indexes_state = self.get_state_indexes_action(state, action)
        return 1 / self.N_state_action[indexes_state]

    def compute_epsilon(self, state):
        state_indexes = self.get_state_indexes(state)
        N_count = self.N_state[state_indexes]
        return self.N0 / (self.N0 + N_count)

    def get_state_indexes(self, state):
        return (state[0], state[1] - 1)

    def get_state_indexes_action(self, state, action):
        return (*self.get_state_indexes(state), action)

    def epsilon_greedy(self, state):
        epsilon = self.compute_epsilon(state)
        actions_reward = self.Q_function[self.get_state_indexes(state)]
        best_action = np.argmax(actions_reward)
        actions = (best_action, 1 - best_action)
        p = (epsilon / M) + 1 - epsilon
        ip = epsilon / M
        return np.random.choice(actions, p=(p, ip))

    def increment_N_state(self, state):
        self.N_state[self.get_state_indexes(state)] += 1

    def increment_N_state_action(self, state, action):
        self.N_state_action[self.get_state_indexes_action(state, action)] += 1


class MonteCarlo(Policy):
    def __init__(self, gamma, N0, nb_episode):
        super().__init__(gamma, N0, nb_episode)

    def compute_discounted_reward(self, episodes):
        discounted_reward = 0
        for i in range(len(episodes)):
            discounted_reward += self.gamma**i * episodes[i][2]
        return discounted_reward

    def update_Q_function(self, episode):
        for i in range(len(episode)):
            state = episode[i][0]
            action = episode[i][1]
            indexes_state = self.get_state_indexes_action(state, action)
            alpha = self.compute_alpha(state, action)
            discounted_reward = self.compute_discounted_reward(episode[i:])
            self.Q_function[indexes_state] += alpha * (
                discounted_reward - self.Q_function[indexes_state]
            )
        return self.Q_function

    def policy_optimization(self, game):
        for i in range(self.nb_episode):
            episode = game.run_one_episode(self)
            self.update_Q_function(episode)
        return self.Q_function


class SARSA(Policy):
    def __init__(self, gamma, lambda_, N0, nb_episode):
        super().__init__(gamma, N0, nb_episode)
        self.E_traces = np.zeros(STATE_ACTION_DIM)
        self.lambda_ = lambda_

    def increment_E_traces(self, state, action):
        self.E_traces[self.get_state_indexes_action(state, action)] += 1

    def reset_E_traces(self):
        self.E_traces = np.zeros(STATE_ACTION_DIM)

    def compute_decay(self, terminal, reward, state, action, next_state, next_action):
        if terminal:
            return (
                reward - self.Q_function[self.get_state_indexes_action(state, action)]
            )
        else:
            return (
                reward
                + self.gamma
                * self.Q_function[
                    self.get_state_indexes_action(next_state, next_action)
                ]
                - self.Q_function[self.get_state_indexes_action(state, action)]
            )

    def update_Q_E(self, state, action, decay):
        self.Q_function += self.compute_alpha(state, action) * decay * self.E_traces
        self.E_traces *= self.gamma * self.lambda_

    def policy_optimization(self, game):
        for episode in range(self.nb_episode):
            self.reset_E_traces()
            game.reset_game()
            state = game.get_state()
            action = self.epsilon_greedy(state)

            while not game.is_terminal():
                reward = game.step(action)
                self.increment_N_state(state)
                self.increment_N_state_action(state, action)
                self.increment_E_traces(state, action)

                if game.is_terminal():
                    decay = self.compute_decay(True, reward, state, action, None, None)
                    self.update_Q_E(state, action, decay)
                    break

                next_state = game.get_state()
                next_action = self.epsilon_greedy(next_state)
                decay = self.compute_decay(
                    game.is_terminal(), reward, state, action, next_state, next_action
                )
                self.update_Q_E(state, action, decay)
                state, action = next_state, next_action
        return self.Q_function


class LinearSarsa(Policy):
    def __init__(self, gamma, lambda_, N0, nb_episode):
        super().__init__(gamma, N0, nb_episode)
        self.w = np.zeros((3, 6, 2))  # 3x6x2 matrix of weights
        self.E_traces = np.zeros((3, 6, 2))
        self.lambda_ = lambda_
        self.alpha = 0.01
        self.epsilon = 0.05

    def compute_feature_vector(self, state, action):
        dealers_interval = [range(1, 6), range(4, 8), range(7, 11)]
        player_interval = [
            range(1, 7),
            range(4, 10),
            range(7, 13),
            range(10, 16),
            range(13, 19),
            range(16, 22),
        ]

        actions_interval = np.array([0, 1])

        player_value = state[0]
        dealer_card = state[1]

        dealer_vector = np.array(
            [dealer_card in interval for interval in dealers_interval]
        )
        player_vector = np.array(
            [player_value in interval for interval in player_interval]
        )
        action_interval = np.array(
            [action == action_type for action_type in actions_interval]
        )

        feature_vector = (
            dealer_vector[:, None, None]
            * player_vector[None, :, None]
            * action_interval[None, None, :]
        )

        return feature_vector

    def compute_q_value(self, state, action):
        feature_vector = self.compute_feature_vector(state, action)
        return np.sum(self.w * feature_vector)

    def compute_decay(self, terminal, reward, state, action, next_state, next_action):
        if terminal:
            return reward - self.compute_q_value(state, action)
        else:
            return (
                reward
                + self.gamma * self.compute_q_value(next_state, next_action)
                - self.compute_q_value(state, action)
            )

    def update_E(self, state, action):
        q_derivative = self.compute_feature_vector(state, action)
        self.E_traces = self.gamma * self.lambda_ * self.E_traces + q_derivative

    def update_w(self, decay):
        self.w += self.alpha * decay * self.E_traces

    def reset_E_traces(self):
        self.E_traces = np.zeros((3, 6, 2))

    def policy_optimization(self, game):
        for episode in range(self.nb_episode):
            self.reset_E_traces()
            game.reset_game()
            state = game.get_state()
            action = self.epsilon_greedy(state)

            while not game.is_terminal():
                reward = game.step(action)

                if game.is_terminal():
                    decay = self.compute_decay(True, reward, state, action, None, None)
                    self.update_E(state, action)
                    self.update_w(decay)
                    break

                next_state = game.get_state()
                next_action = self.epsilon_greedy(next_state)
                decay = self.compute_decay(
                    game.is_terminal(), reward, state, action, next_state, next_action
                )
                self.update_E(state, action)
                self.update_w(decay)
                state, action = next_state, next_action
        return self.w

    def compute_Q_table(self):
        Q_table = np.zeros(STATE_ACTION_DIM)
        for player_value, dealer_card, action in itertools.product(
            range(22), range(10), range(2)
        ):
            Q_table[player_value, dealer_card, action] = np.sum(
                self.w
                * self.compute_feature_vector((player_value, dealer_card + 1), action)
            )
        return Q_table

    def epsilon_greedy(self, state):
        reward_0 = self.compute_q_value(state, 0)
        reward_1 = self.compute_q_value(state, 1)

        best_action = 0 if reward_0 > reward_1 else 1
        actions = (best_action, 1 - best_action)

        p = (self.epsilon / M) + 1 - self.epsilon
        ip = self.epsilon / M

        return np.random.choice(actions, p=(p, ip))


def plot_3d(V_function):
    # Create coordinate arrays for the states
    n1, n2 = V_function.shape
    s1 = np.arange(n1)  # first component of state
    s2 = np.arange(n2)  # second component of state

    # Create a meshgrid for 3D plotting
    S1, S2 = np.meshgrid(s1, s2, indexing="ij")  # indexing='ij' matches V[i,j]

    # Create the figure and 3D axes
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    surf = ax.plot_surface(S1, S2, V_function, cmap="viridis", edgecolor="k")

    # Add a color bar to indicate value magnitude
    fig.colorbar(surf, shrink=0.5, aspect=10, label="V(s)")

    # Set axis labels and title
    ax.set_xlabel("Player Value")
    ax.set_ylabel("Dealer Card")
    ax.set_zlabel("Value V(s)")
    ax.set_title("3D Surface of Value Function")

    # Optional: adjust view angle
    ax.view_init(elev=30, azim=45)  # elevation, azimuth

    # Show the plot
    plt.show()


def sarsa_lambda_mse():
    monte_carlo = MonteCarlo(gamma=1, N0=100, nb_episode=100000)
    game_environment = Game()
    Q_star = monte_carlo.policy_optimization(game_environment)

    Sarsa_Q = []
    for lambda_ in np.linspace(0, 1, 10):
        sarsa = SARSA(gamma=1, N0=100, lambda_=lambda_, nb_episode=1000)
        game_environment = Game()
        Q_function = sarsa.policy_optimization(game_environment)
        Sarsa_Q.append(Q_function)

    mse = [(1 / Q_star.size) * np.sum((Q_star - Q) ** 2) for Q in Sarsa_Q]
    sns.lineplot(x=np.linspace(0, 1, 10), y=mse)
    plt.show()


def sarsa_episode_mse():
    monte_carlo = MonteCarlo(gamma=1, N0=100, nb_episode=100000)
    game_environment = Game()
    Q_star = monte_carlo.policy_optimization(game_environment)
    Sarsa_Q_0 = []
    for nb_episode in np.linspace(1000, 100000, 10).astype(int):
        sarsa = SARSA(gamma=1, N0=100, lambda_=0, nb_episode=nb_episode)
        game_environment = Game()
        Q_function = sarsa.policy_optimization(game_environment)
        Sarsa_Q_0.append(Q_function)

    mse_0 = [(1 / Q_star.size) * np.sum((Q_star - Q) ** 2) for Q in Sarsa_Q_0]

    sarsa_Q_1 = []
    for nb_episode in np.linspace(1000, 100000, 10).astype(int):
        sarsa = SARSA(gamma=1, N0=100, lambda_=1, nb_episode=nb_episode)
        game_environment = Game()
        Q_function = sarsa.policy_optimization(game_environment)
        sarsa_Q_1.append(Q_function)

    mse_1 = [(1 / Q_star.size) * np.sum((Q_star - Q) ** 2) for Q in sarsa_Q_1]

    sns.lineplot(x=np.linspace(1000, 100000, 10).astype(int), y=mse_0)
    sns.lineplot(x=np.linspace(1000, 100000, 10).astype(int), y=mse_1)
    plt.show()


def linear_sarsa_lambda_mse():
    monte_carlo = MonteCarlo(gamma=1, N0=100, nb_episode=100000)
    game_environment = Game()
    Q_star = monte_carlo.policy_optimization(game_environment)

    linear_Sarsa_Q_mse = []
    for lambda_ in np.linspace(0, 1, 10):
        mean_mse = []
        for i in range(100):
            sarsa = LinearSarsa(gamma=1, N0=100, lambda_=lambda_, nb_episode=1000)
            game_environment = Game()
            sarsa.policy_optimization(game_environment)
            Q_function = sarsa.compute_Q_table()
            mse = (1 / Q_star.size) * np.sum((Q_star - Q_function) ** 2)
            mean_mse.append(mse)
        linear_Sarsa_Q_mse.append(np.mean(mean_mse))
    sns.lineplot(x=np.linspace(0, 1, 10), y=linear_Sarsa_Q_mse)
    plt.show()


def linear_sarsa_episode_mse():
    monte_carlo = MonteCarlo(gamma=1, N0=100, nb_episode=100000)
    game_environment = Game()
    Q_star = monte_carlo.policy_optimization(game_environment)
    linear_Sarsa_Q_0 = []
    for nb_episode in np.linspace(100, 10000, 10).astype(int):
        sarsa = LinearSarsa(gamma=1, N0=100, lambda_=0, nb_episode=nb_episode)
        game_environment = Game()
        sarsa.policy_optimization(game_environment)
        Q_function = sarsa.compute_Q_table()
        linear_Sarsa_Q_0.append(Q_function)

    mse_0 = [(1 / Q_star.size) * np.sum((Q_star - Q) ** 2) for Q in linear_Sarsa_Q_0]

    linear_Sarsa_Q_1 = []

    for nb_episode in np.linspace(100, 10000, 10).astype(int):
        sarsa = LinearSarsa(gamma=1, N0=100, lambda_=1, nb_episode=nb_episode)
        game_environment = Game()
        sarsa.policy_optimization(game_environment)
        Q_function = sarsa.compute_Q_table()
        linear_Sarsa_Q_1.append(Q_function)

    mse_1 = [(1 / Q_star.size) * np.sum((Q_star - Q) ** 2) for Q in linear_Sarsa_Q_1]

    sns.lineplot(x=np.linspace(100, 10000, 10).astype(int), y=mse_0, color="blue")
    sns.lineplot(x=np.linspace(100, 10000, 10).astype(int), y=mse_1, color="red")
    plt.show()


def plot_monte_carlo():
    monte_carlo = MonteCarlo(gamma=1, N0=100, nb_episode=100000)
    game_environment = Game()
    Q_star = monte_carlo.policy_optimization(game_environment)
    V_star = np.max(Q_star, axis=2)
    plot_3d(V_star)


def plot_sarsa():
    sarsa = SARSA(gamma=1, N0=100, lambda_=0.3, nb_episode=100000)
    game_environment = Game()
    Q_star = sarsa.policy_optimization(game_environment)
    V_star = np.max(Q_star, axis=2)
    plot_3d(V_star)


def plot_linear_sarsa():
    linear_sarsa = LinearSarsa(gamma=1, lambda_=0.9, N0=100, nb_episode=100000)
    game_environment = Game()
    linear_sarsa.policy_optimization(game_environment)
    Q = linear_sarsa.compute_Q_table()
    V = np.max(Q, axis=2)
    plot_3d(V)


def main():
    # sarsa_episode_mse()
    # plot_monte_carlo()
    # plot_sarsa()
    # plot_linear_sarsa()
    # linear_sarsa_lambda_mse()
    linear_sarsa_episode_mse()

if __name__ == "__main__":
    main()
