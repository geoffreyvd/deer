""" Launcher for mountain car environment with continuous action space.
Same principles as run_toy_env. See the wiki for more details.

"""

import sys
import logging
import numpy as np

import deer.experiment.base_controllers as bc
from deer.default_parser import process_args
from deer.agent import NeuralAgent
from deer.learning_algos.AC_net_keras import MyACNetwork
from mountain_car_continuous_env import MyEnv as mountain_car_continuous_env
from deer.policies import LongerExplorationPolicy


class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 200
    EPOCHS = 200
    STEPS_PER_TEST = 200
    PERIOD_BTW_SUMMARY_PERFS = 10

    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 1

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.002
    LEARNING_RATE_DECAY = 0.99
    DISCOUNT = 0.9
    DISCOUNT_INC = 0.99
    DISCOUNT_MAX = 0.95
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_NORM = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = 0.2
    EPSILON_DECAY = 10000
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    FREEZE_INTERVAL = 100
    DETERMINISTIC = True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    if parameters.deterministic:
        rng = np.random.RandomState(12345)
    else:
        rng = np.random.RandomState()
    
    # --- Instantiate environment ---
    env = mountain_car_continuous_env(rng)

    # --- Instantiate qnetwork ---
    qnetwork = MyACNetwork(
        env,
        parameters.rms_decay,
        parameters.rms_epsilon,
        parameters.momentum,
        parameters.clip_norm,
        parameters.freeze_interval,
        parameters.batch_size,
        parameters.update_rule,
        rng)
    
    train_policy=LongerExplorationPolicy(qnetwork, env.nActions(), rng, 1.,10)
    
    # --- Instantiate agent ---
    agent = NeuralAgent(
        env,
        qnetwork,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng,
        exp_priority=1.,
        train_policy=train_policy)

    # --- Bind controllers to the agent ---
    # For comments, please refer to run_toy_env.py
    agent.attach(bc.VerboseController(
        evaluate_on='epoch', 
        periodicity=1))

    agent.attach(bc.TrainerController(
        evaluate_on='action', 
        periodicity=parameters.update_frequency, 
        show_episode_avg_V_value=True, 
        show_avg_Bellman_residual=True))

    agent.attach(bc.LearningRateController(
        initial_learning_rate=parameters.learning_rate,
        learning_rate_decay=parameters.learning_rate_decay,
        periodicity=1))

    agent.attach(bc.DiscountFactorController(
        initial_discount_factor=parameters.discount,
        discount_factor_growth=parameters.discount_inc,
        discount_factor_max=parameters.discount_max,
        periodicity=1))

    agent.attach(bc.EpsilonController(
        initial_e=parameters.epsilon_start, 
        e_decays=parameters.epsilon_decay, 
        e_min=parameters.epsilon_min,
        evaluate_on='action', 
        periodicity=1, 
        reset_every='none'))

    agent.attach(bc.InterleavedTestEpochController(
        id=0, 
        epoch_length=parameters.steps_per_test, 
        periodicity=1, 
        show_score=True,
        summarize_every=parameters.period_btw_summary_perfs))
    
    # --- Run the experiment ---
    agent.run(parameters.epochs, parameters.steps_per_epoch)
