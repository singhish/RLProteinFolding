from utils import fetch_protein
from protein import ProteinState
from ddpg import Agent, ReplayBuffer


EPISODES = 10000
STEPS = 500

if __name__ == "__main__":
    goal_state = fetch_protein("2jof")
    state_dim = goal_state.n_residues() * 2
    action_dim = goal_state.n_residues() * 2
    buffer = ReplayBuffer(10000)
    agent = Agent(state_dim, action_dim, (0, 360))

    for _ in range(EPISODES):
        data = {"state": ProteinState(n_residues=goal_state.n_residues())}
        for _ in range(STEPS):
            action = agent.get_action(data["state"])
            next_state = data["state"].do_action(action)
            reward = data["state"].eval_state() - next_state.eval_state()

            buffer.append(data["state"], action, reward, next_state)

            agent.update(buffer)

            print(data["state"].l2_norm(goal_state))
            data["state"] = ProteinState(angles=next_state.angles())
