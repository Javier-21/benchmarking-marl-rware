# This file is create by Javier-21 in May 2026.
# It contains utility functions for rendering episodes of trained policies.

import os
import jax
import jax.numpy as jnp

from jax import tree

def render_one_episode(config, params, actor_network, env, max_steps=500) -> None:
    """Rollout episodes of a trained MAPPO policy."""
    # Create actor networks (We only care about the policy during rendering)
    apply_fn = actor_network.apply
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    key = jax.random.PRNGKey(config.system.seed)
    key, reset_key = jax.random.split(key)

    state, timestep = reset_fn(reset_key)
    params = jax.tree.map(lambda x: jnp.squeeze(x), params)
    
    states = [state.env_state]
    episode_return = 0
    episode_length = 0
    while not timestep.last().all():
        key, action_key = jax.random.split(key)
        timestep.observation = tree.map(lambda x: x[jnp.newaxis, ...], timestep.observation)
        pi = apply_fn(params, timestep.observation)
        if config["arch"]["evaluation_greedy"]:
            action = pi.mode()
        else:
            action = pi.sample(seed=action_key)
        state, timestep = step_fn(state, action[0])
        states.append(state.env_state)
        episode_return += jnp.mean(timestep.reward)
        episode_length += 1

    # Limit the number of steps to record to the maximum number of steps
    steps = min([max_steps, len(states) - 1])
    states = states[:steps]

    # Render the episode
    save_path = f"./render/{config.logger.system_name}/rware_{config.env.scenario.task_name}.gif"
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    env.animate(states=states, interval=100, save_path=save_path)



def render_one_episode_rec_sable(config, params, sable_execution_fn, env, hstates, max_steps=500) -> None:
    """Rollout episodes of a trained MAPPO policy."""
    # Create actor networks (We only care about the policy during rendering)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    key = jax.random.PRNGKey(config.system.seed)
    key, reset_key = jax.random.split(key)

    state, timestep = reset_fn(reset_key)
    #params = jax.tree.map(lambda x: jnp.squeeze(x), params)
    params = jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, params)
    
    states = [state.env_state]
    episode_return = 0
    episode_length = 0
    while not timestep.last().all():
        key, action_key = jax.random.split(key)
        timestep.observation = tree.map(lambda x: x[jnp.newaxis, ...], timestep.observation)

        action, _, _, hstates = sable_execution_fn(  # type: ignore
                params,
                timestep.observation,
                hstates,
                action_key,
            )
        
        #Execute the action in the environment
        state, timestep = step_fn(state, action[0])
        states.append(state.env_state)
        episode_return += jnp.mean(timestep.reward)
        episode_length += 1

    # Limit the number of steps to record to the maximum number of steps
    steps = min([max_steps, len(states) - 1])
    states = states[:steps]

    # Render the episode
    save_path = f"./render/{config.logger.system_name}/rware_{config.env.scenario.task_name}.gif"
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    env.animate(states=states, interval=100, save_path=save_path)



def render_one_episode_sable(config, params, sable_execution_fn, env, max_steps=500) -> None:
    """Rollout episodes of a trained MAPPO policy."""
    # Create actor networks (We only care about the policy during rendering)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    key = jax.random.PRNGKey(config.system.seed)
    key, reset_key = jax.random.split(key)

    state, timestep = reset_fn(reset_key)
    #params = jax.tree.map(lambda x: jnp.squeeze(x), params)
    params = jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, params)
    
    states = [state.env_state]
    episode_return = 0
    episode_length = 0
    while not timestep.last().all():
        key, action_key = jax.random.split(key)
        timestep.observation = tree.map(lambda x: x[jnp.newaxis, ...], timestep.observation)
        
        action, _, _, _ = sable_execution_fn(  # type: ignore
                params,
                observation=timestep.observation,
                key=action_key,
            )
        
        #Execute the action in the environment
        state, timestep = step_fn(state, action[0])
        states.append(state.env_state)
        episode_return += jnp.mean(timestep.reward)
        episode_length += 1

    # Limit the number of steps to record to the maximum number of steps
    steps = min([max_steps, len(states) - 1])
    states = states[:steps]

    # Render the episode
    save_path = f"./render/{config.logger.system_name}/rware_{config.env.scenario.task_name}.gif"
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    env.animate(states=states, interval=100, save_path=save_path)