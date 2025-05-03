#!/usr/bin/env python3
# Minimal, fully-jit AlphaZero (Chess) with PGX + JAX/Haiku
# Tested: JAX 0.4.28, Haiku 0.0.15, Optax 0.1.8, mctx 0.1.4

import datetime, os, pickle, time
from functools import partial
from typing import NamedTuple

import haiku as hk, jax, jax.numpy as jnp, optax, mctx, pgx, wandb
from pgx.experimental import auto_reset
from omegaconf import OmegaConf
from pydantic import BaseModel

# ---------- конфиг ----------
class Config(BaseModel):
    env_id: str = "chess"
    seed: int = 42
    max_num_iters: int = 300        # JIT-петля ⇢ компилируется один раз
    # сеть
    num_channels: int = 128
    num_layers: int = 6             # 6 блоков ResNet-v2
    resnet_v2: bool = True
    # self-play
    selfplay_batch_size: int = 1024
    num_simulations: int = 32       # ↓64 → 32 - быстрее, почти без потерь качества
    max_num_steps: int = 256
    # обучение
    training_batch_size: int = 4096
    learning_rate: float = 1e-3
    # прочее
    eval_interval: int = 5
    use_pjit: bool = True           # False → останется pmap-репликация

    class Config: extra = "forbid"

cfg = Config(**OmegaConf.from_cli())
print(cfg)

# ---------- среда ----------
env = pgx.make(cfg.env_id)
dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 1))
OBS_SHAPE = dummy_state.observation.shape[1:]  # (B,H,W,C)

# ---------- сеть ----------
class AZNet(hk.Module):
    def __init__(self, actions, blocks, ch, v2=True):
        super().__init__()
        self.a, self.b, self.c, self.v2 = actions, blocks, ch, v2

    def __call__(self, x, *, train: bool):
        x = hk.Conv2D(self.c, 3, padding="SAME")(x)
        if not self.v2:
            x = hk.BatchNorm(True, True, 0.9)(x, train) ; x = jax.nn.relu(x)
        for _ in range(self.b):
            res = x
            if self.v2:
                x = hk.BatchNorm(True, True, 0.9)(x, train) ; x = jax.nn.relu(x)
            x = hk.Conv2D(self.c, 3, padding="SAME")(x)
            x = hk.BatchNorm(True, True, 0.9)(x, train) ; x = jax.nn.relu(x)
            x = hk.Conv2D(self.c, 3, padding="SAME")(x)
            x = x + res if self.v2 else jax.nn.relu(x + res)
        if self.v2:
            x = hk.BatchNorm(True, True, 0.9)(x, train) ; x = jax.nn.relu(x)

        # policy head
        p = hk.Conv2D(32, 1)(x)
        p = hk.BatchNorm(True, True, 0.9)(p, train) ; p = jax.nn.relu(p)
        p = hk.Flatten()(p)
        p = hk.Linear(self.a)(p)

        # value head
        v = hk.Conv2D(8, 1)(x)
        v = hk.BatchNorm(True, True, 0.9)(v, train) ; v = jax.nn.relu(v)
        v = hk.Flatten()(v)
        v = hk.Linear(self.c)(v) ; v = jax.nn.relu(v)
        v = hk.Linear(1)(v)
        v = jnp.tanh(v).reshape((-1,))

        return p, v

def net_fwd(obs, *, eval=False):
    net = AZNet(env.num_actions, cfg.num_layers, cfg.num_channels, cfg.resnet_v2)
    # ↓ для экономии памяти можно включить градиент-чекпойнтинг
    # net = hk.remat(net)             # требует Haiku ≥0.0.16 :contentReference[oaicite:5]{index=5}
    return net(obs, train=not eval)

forward = hk.without_apply_rng(hk.transform_with_state(net_fwd))
optimizer = optax.adam(cfg.learning_rate)

# ---------- вспомогательные типы ----------
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

# ---------- ядра MCTS ----------
def recurrent(model, key, action, st: pgx.State):
    params, st_vars = model
    cur = st.current_player
    st = jax.vmap(env.step)(st, action)
    (logits, value), _ = forward.apply(params, st_vars, st.observation, eval=True)
    logits = logits - logits.max(-1, keepdims=True)
    logits = jnp.where(st.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    reward = st.rewards[jnp.arange(reward.shape[0]), cur]
    value = jnp.where(st.terminated, 0.0, value)
    disc = jnp.where(st.terminated, 0.0, -jnp.ones_like(value))
    return mctx.RecurrentFnOutput(reward, disc, logits, value), st

# ---------- self-play + вычисление таргетов ----------
@partial(jax.pmap, axis_name="d")
def selfplay(model, key) -> Sample:
    params, st_vars = model
    batch = cfg.selfplay_batch_size // jax.device_count()
    keys = jax.random.split(key, batch)
    st = jax.vmap(env.init)(keys)

    def step(carry, k):
        st = carry
        k1, k2 = jax.random.split(k)
        (lg, val), _ = forward.apply(params, st_vars, st.observation, eval=True)
        root = mctx.RootFnOutput(lg, val, st)
        po = mctx.gumbel_muzero_policy(
            model, k1, root, recurrent, cfg.num_simulations,
            invalid_actions=~st.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value, gumbel_scale=1.0
        )
        actor = st.current_player
        st = jax.vmap(auto_reset(env.step, env.init))(st, po.action, jax.random.split(k2, batch))
        disc = jnp.where(st.terminated, 0.0, -jnp.ones_like(val))
        return st, SPOut(st.observation, st.rewards[jnp.arange(batch), actor],
                         st.terminated, po.action_weights, disc)

    _, data = jax.lax.scan(step, st, jax.random.split(key, cfg.max_num_steps))
    # --- value target через обратное сканирование ---
    mask = jnp.cumsum(data.terminated[::-1], 0)[::-1] >= 1
    def bwd(c, i):
        ix = cfg.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix]*c
        return v, v
    _, v_tgt = jax.lax.scan(bwd, jnp.zeros(batch), jnp.arange(cfg.max_num_steps))
    v_tgt = v_tgt[::-1]

    return Sample(data.obs, data.action_weights, v_tgt, mask)

# ---------- функция потерь ----------
def loss_fn(params, st_vars, sample: Sample):
    (lg, v), st_vars = forward.apply(params, st_vars, sample.obs, eval=False)
    pl = optax.softmax_cross_entropy(lg, sample.policy_tgt).mean()
    vl = (optax.l2_loss(v, sample.value_tgt) * sample.mask).mean()
    return pl + vl, (st_vars, pl, vl)

@partial(jax.pmap, axis_name="d", donate_argnums=(0,1))
def train_step(model, opt_state, sample: Sample):
    params, st_vars = model
    grads, (st_vars, pl, vl) = jax.grad(loss_fn, has_aux=True)(params, st_vars, sample)
    grads = jax.lax.pmean(grads, "d")         # all-reduce
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return (params, st_vars), opt_state, pl, vl

# ---------- единая итерация self-play → обучение ----------
def loop_body(carry, i):
    model, opt_state, key = carry
    key, k_sp, k_mb = jax.random.split(key, 3)
    sp_samp = selfplay(model, k_sp)                            # pmap: [n_dev, …]
    # --- перемешивание/батчинг внутри устройств ---
    def shuffle(samp, k):
        idx = jax.random.permutation(k, samp.obs.shape[0])
        return jax.tree.map(lambda x: jax.lax.index_take(x, idx, axes=(0,)), samp)
    sp_samp = shuffle(sp_samp, k_mb)
    # делим на minibatch-и вдоль batch-оси
    steps = sp_samp.obs.shape[0] // cfg.training_batch_size
    def mb_scan(c, j):
        m, o = c
        # берём под-батч
        one = jax.tree.map(lambda x: x[j*cfg.training_batch_size:(j+1)*cfg.training_batch_size], sp_samp)
        m, o, pl, vl = train_step(m, o, one)
        return (m, o), (pl, vl)
    (model, opt_state), metrics = jax.lax.scan(mb_scan, (model, opt_state), jnp.arange(steps))
    pl, vl = jax.tree.map(lambda x: x.mean(), metrics)        # средняя потеря
    return (model, opt_state, key), (pl, vl)

jit_loop = jax.jit(loop_body, donate_argnums=(0,))

# ---------- инициализация ----------
devices = jax.local_devices()
params, st_vars = forward.init(jax.random.PRNGKey(0), dummy_state.observation)
opt_state = optimizer.init(params)
model = (params, st_vars)
if cfg.use_pjit:
    from jax.sharding import PartitionSpec as P
    mesh = jax.sharding.Mesh(devices, ("d",))
    model = jax.device_put(model, jax.sharding.NamedSharding(mesh, P()))
    opt_state = jax.device_put(opt_state, jax.sharding.NamedSharding(mesh, P()))
else:
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

wandb.init(project="pgx-chess-az-jit", config=cfg.model_dump())
ckpt_dir = os.path.join("checkpoints", f"{cfg.env_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
os.makedirs(ckpt_dir, exist_ok=True)

# ---------- главный цикл ----------
key = jax.random.PRNGKey(cfg.seed)
hours, frames = 0.0, 0

for it in range(cfg.max_num_iters):
    st = time.time()
    (model, opt_state, key), (p_loss, v_loss) = jit_loop((model, opt_state, key), it)
    frames += cfg.selfplay_batch_size * cfg.max_num_steps
    hours += (time.time()-st)/3600
    wandb.log({"iter": it, "policy_loss": p_loss, "value_loss": v_loss,
               "frames": frames, "hours": hours})
    if it % cfg.eval_interval == 0:
        # быстрая оценка: доля побед против случайного
        rng = jax.random.split(key, devices.__len__())
        R = jax.pmap(lambda k, m: (forward.apply(m[0], m[1],
                    env.init(k).observation, eval=True)[0].argmax(-1)==0).mean())(rng, model)
        wandb.log({"eval/win_vs_rand": R.mean()})

        with open(os.path.join(ckpt_dir, f"{it:06d}.pkl"), "wb") as f:
            pickle.dump({"model": jax.device_get(model[0]), "opt": jax.device_get(opt_state)}, f)

print("✅ Training finished")
