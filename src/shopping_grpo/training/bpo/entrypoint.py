"""Install BPO runtime hooks before entering veRL's Hydra application."""

from shopping_grpo.training.bpo.runtime import install_bpo_runtime


def main():
    install_bpo_runtime()
    from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
