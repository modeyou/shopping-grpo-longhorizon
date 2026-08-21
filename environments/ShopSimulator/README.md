# ShopSimulator

This is the embedded ShopSimulator environment used by the tutorial. The
snapshot includes the product archive, Environment v2.1, default Reward v3,
the opt-in Reward v4 candidate and the
structured `/api/shop_agent` service. Its upstream source commit is recorded in
[`EMBEDDED_SOURCE.json`](EMBEDDED_SOURCE.json).

Users should not install or start this directory manually. From the repository
root, run:

```bash
bash scripts/setup.sh
bash scripts/start_environment.sh
```

Generated product JSON, search indexes, virtual environments and logs are not
committed.

Reward v4 is opt-in and does not change the default configuration:

```bash
export SHOP_ENV_CONFIG="$PWD/environments/ShopSimulator/shop_env/configs/environment-v4.json"
bash scripts/start_environment.sh
```
