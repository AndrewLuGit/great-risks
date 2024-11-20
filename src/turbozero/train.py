from game_def import *
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.evaluators.alphazero import AlphaZero
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import EpisodeReplayBuffer
from functools import partial
from core.testing.two_player_tester import TwoPlayerTester
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
import optax
import orbax.checkpoint as ocp

resnet = AZResnet(AZResnetConfig(
    policy_head_out_size=6,
    num_blocks=4,
    num_channels=32,
))

az_evaluator = AlphaZero(MCTS)(
    eval_fn = make_nn_eval_fn(resnet, observe),
    num_iterations = 32,
    max_nodes = 40,
    branching_factor = 6,
    action_selector = PUCTSelector(),
    temperature = 1.0
)

az_evaluator_test = AlphaZero(MCTS)(
    eval_fn = make_nn_eval_fn(resnet, observe),
    num_iterations = 64,
    max_nodes = 80,
    branching_factor = 6,
    action_selector = PUCTSelector(),
    temperature = 0.0
)

replay_memory = EpisodeReplayBuffer(capacity=1000)

trainer = Trainer(
    batch_size = 1024,
    train_batch_size = 4096,
    warmup_steps = 0,
    collection_steps_per_epoch = 256,
    train_steps_per_epoch = 64,
    nn = resnet,
    loss_fn = partial(az_default_loss_fn, l2_reg_lambda = 0.0),
    optimizer = optax.adam(1e-3),
    evaluator = az_evaluator,
    memory_buffer = replay_memory,
    max_episode_steps = 80,
    env_step_fn = step,
    env_init_fn = init,
    state_to_nn_input_fn=observe,
    testers = [TwoPlayerTester(num_episodes=64, render_fn=render_text, render_dir=".")],
    evaluator_test = az_evaluator_test,
    # wandb_project_name = 'turbozero-othello' 
)

output = trainer.train_loop(seed=0, num_epochs=100, eval_every=5)
checkpointer = ocp.StandardCheckpointer()
checkpointer.save("ckpt", output.train_state.params)
