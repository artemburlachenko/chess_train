#!/usr/bin/env python3
import os, sys, time, runpy

# Чистим переменные, которые мешают PJRT на TPU-VM
for v in ("TPU_DRIVER_ADDRESS", "PJRT_DEVICE", "TPU_NAME", "XLA_FLAGS"):
    os.environ.pop(v, None)

# Безбуферный вывод и платформа по умолчанию
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("JAX_PLATFORMS", "tpu")

# 1) СНАЧАЛА: инициализация JAX Distributed — до любых импортов, трогающих JAX
import jax
t0 = time.time()
jax.distributed.initialize()
print(
    f"proc {jax.process_index()} of {jax.process_count()} "
    f"local {len(jax.local_devices())} global {len(jax.devices())} "
    f"init={time.time()-t0:.2f}s",
    flush=True,
)

# 2) W&B выключаем на всех, кроме лидера, либо если нет ключа
if jax.process_index() != 0 or not os.getenv("WANDB_API_KEY"):
    os.environ["WANDB_MODE"] = "disabled"

# 3) Запускаем тренер
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.argv = ["chess_train.py", *sys.argv[1:]]  # пробрасываем CLI-арги в тренер

try:
    runpy.run_path(os.path.join(here, "chess_train.py"), run_name="__main__")
finally:
    # Аккуратное завершение, чтобы не словить барьеры «different incarnation»
    try:
        jax.distributed.shutdown()
    except Exception:
        pass
