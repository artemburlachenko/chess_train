#!/usr/bin/env python3
# PGX Chess – fixed pmap keys & minor bugs
import datetime, os, pickle, time
from functools import partial
from typing import NamedTuple

import haiku as hk, jax, jax.numpy as jnp, mctx, optax, pgx, wandb
from pgx.experimental import auto_reset
from omegaconf import OmegaConf
from pydantic import BaseModel

# ---------- config ----------
class Config(BaseModel):
    env_id: str = "chess"
    seed: int = 42
    max_num_iters: int = 300
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    selfplay_batch_size: int = 2048
    num_simulations: int = 32
    max_num_steps: int = 256
    training_batch_size: int = 4096
    learning_rate: float = 1e-3
    eval_interval: int = 5

    class Config: extra = "forbid"

cfg = Config(**OmegaConf.from_cli())
print(cfg)

# ---------- env & net ----------
env = pgx.make(cfg.env_id)
dummy_state = env.init(jax.random.PRNGKey(0))

class AZNet(hk.Module):
    def __init__(self, actions, blocks, ch, v2=True): super().__init__(); self.a=actions; self.b=blocks; self.c=ch; self.v2=v2
    def __call__(self, x, *, train: bool):
        x = hk.Conv2D(self.c, 3, padding="SAME")(x)
        if not self.v2: x = hk.BatchNorm(True, True, .9)(x, train); x = jax.nn.relu(x)
        for _ in range(self.b):
            res = x
            if self.v2: x = hk.BatchNorm(True, True, .9)(x, train); x = jax.nn.relu(x)
            x = hk.Conv2D(self.c, 3, padding="SAME")(x)
            x = hk.BatchNorm(True, True, .9)(x, train); x = jax.nn.relu(x)
            x = hk.Conv2D(self.c, 3, padding="SAME")(x)
            x = x + res if self.v2 else jax.nn.relu(x + res)
        if self.v2: x = hk.BatchNorm(True, True, .9)(x, train); x = jax.nn.relu(x)

        p = hk.Conv2D(32, 1)(x); p = hk.BatchNorm(True, True, .9)(p, train); p = jax.nn.relu(p)
        p = hk.Flatten()(p); p = hk.Linear(self.a)(p)

        v = hk.Conv2D(8, 1)(x);  v = hk.BatchNorm(True, True, .9)(v, train); v = jax.nn.relu(v)
        v = hk.Flatten()(v);     v = hk.Linear(self.c)(v);       v = jax.nn.relu(v)
        v = hk.Linear(1)(v);     v = jnp.tanh(v).reshape((-1,))
        return p, v

def fwd(obs, *, eval=False):
    net = AZNet(env.num_actions, cfg.num_layers, cfg.num_channels, cfg.resnet_v2)
    return net(obs, train=not eval)

forward = hk.without_apply_rng(hk.transform_with_state(fwd))
optimizer = optax.adam(cfg.learning_rate)

# ---------- utility NamedTuples ----------
class SPOut(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray
class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray

# ---------- recurrent fn (bug-fixed) ----------
def recurrent(model, key, action, st: pgx.State):
    params, st_vars = model
    cur = st.current_player
    st  = jax.vmap(env.step)(st, action)
    (logits, value), _ = forward.apply(params, st_vars, st.observation, eval=True)
    logits = jnp.where(st.legal_action_mask, logits, -jnp.inf)
    reward = st.rewards[jnp.arange(st.rewards.shape[0]), cur]  # ← fixed line
    value  = jnp.where(st.terminated, 0.0, value)
    disc   = jnp.where(st.terminated, 0.0, -jnp.ones_like(value))
    return mctx.RecurrentFnOutput(reward, disc, logits, value), st

# ---------- self-play (expects key per device) ----------
@partial(jax.pmap, axis_name="dev")
def selfplay(model, keys) -> Sample:
    params, st_vars = model
    batch = cfg.selfplay_batch_size // jax.device_count()
    st = jax.vmap(env.init)(keys)

    def step(state, k):
        k1, k2 = jax.random.split(k)
        (lg, val), _ = forward.apply(params, st_vars, state.observation, eval=True)
        root = mctx.RootFnOutput(lg, val, state)
        po = mctx.gumbel_muzero_policy(model, k1, root, recurrent,
                                       cfg.num_simulations,
                                       invalid_actions=~state.legal_action_mask,
                                       qtransform=mctx.qtransform_completed_by_mix_value)
        actor = state.current_player
        state = jax.vmap(auto_reset(env.step, env.init))(state, po.action,
                                                         jax.random.split(k2, batch))
        disc = jnp.where(state.terminated, 0.0, -jnp.ones_like(val))
        return state, SPOut(state.observation,
                            state.rewards[jnp.arange(batch), actor],
                            state.terminated, po.action_weights, disc)

    _, data = jax.lax.scan(step, st, jax.random.split(keys[0], cfg.max_num_steps))
    mask = jnp.cumsum(data.terminated[::-1], 0)[::-1] >= 1
    def back(c, i):
        ix = cfg.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * c
        return v, v
    _, v_tgt = jax.lax.scan(back, jnp.zeros(batch), jnp.arange(cfg.max_num_steps))
    v_tgt = v_tgt[::-1]
    return Sample(data.obs, data.action_weights, v_tgt, mask)

# ---------- loss & train ----------
def loss_fn(p, s, smp: Sample):
    (lg, v), s = forward.apply(p, s, smp.obs, eval=False)
    pl = optax.softmax_cross_entropy(lg, smp.policy_tgt).mean()
    vl = (optax.l2_loss(v, smp.value_tgt) * smp.mask).mean()
    return pl + vl, (s, pl, vl)

@partial(jax.pmap, axis_name="dev", donate_argnums=(0,1))
def train_step(model, opt_state, smp: Sample):
    p, s = model
    grads, (s, pl, vl) = jax.grad(loss_fn, has_aux=True)(p, s, smp)
    grads = jax.lax.pmean(grads, "dev")
    upds, opt_state = optimizer.update(grads, opt_state, p)
    p = optax.apply_updates(p, upds)
    return (p, s), opt_state, pl, vl

# ---------- one loop iteration ----------
def loop(carry, _):
    model, opt_state, key = carry
    key, k_sp, k_mb = jax.random.split(key, 3)
    # replicate RNG key – now shape (n_dev, 2)
    k_sp_dev = jax.random.split(k_sp, jax.device_count())
    sample = selfplay(model, k_sp_dev)

    # simple per-device shuffle
    def _shuf(s, k): perm = jax.random.permutation(k, s.obs.shape[0]); return jax.tree_map(lambda x: x[perm], s)
    sample = jax.pmap(_shuf)(sample, jax.random.split(k_mb, jax.device_count()))

    steps = sample.obs.shape[0] // cfg.training_batch_size
    def body(c, i):
        m, o = c
        one = jax.tree_map(lambda x: x[i*cfg.training_batch_size:(i+1)*cfg.training_batch_size], sample)
        m, o, pl, vl = train_step(m, o, one)
        return (m, o), (pl, vl)
    (model, opt_state), (ploss, vloss) = jax.lax.scan(body, (model, opt_state), jnp.arange(steps))
    return (model, opt_state, key), (ploss.mean(), vloss.mean())

jit_loop = jax.jit(loop, donate_argnums=(0,))

# ---------- init ----------
params, st_vars = forward.init(jax.random.PRNGKey(0), dummy_state.observation)
opt_state = optimizer.init(params)
model, opt_state = jax.device_put_replicated((params, st_vars), jax.local_devices()), jax.device_put_replicated(opt_state, jax.local_devices())
wandb.init(project="pgx-chess-az-fixed", config=cfg.model_dump())

# ---------- training ----------
rng = jax.random.PRNGKey(cfg.seed)
for it in range(cfg.max_num_iters):
    t0 = time.time()
    (model, opt_state, rng), (pl, vl) = jit_loop((model, opt_state, rng), None)
    wandb.log({"iter": it, "policy_loss": pl, "value_loss": vl,
               "sec_per_iter": time.time()-t0})
    if it % cfg.eval_interval == 0:
        with open(f"ckpt_{it:05d}.pkl", "wb") as f:
            pickle.dump(jax.device_get(model[0]), f)

print("✅ done")
