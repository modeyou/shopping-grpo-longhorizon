'''Frozen contracts for the single formal native Reward-v4 GRPO run.'''

from argparse import Namespace
from pathlib import Path

from scripts.run_formal_grpo import TOTAL_UPDATES, build_command


def arguments(tmp_path):
    return Namespace(
        model=tmp_path / 'model',
        output=tmp_path / 'output',
        shopper_model='shopper',
        shopper_base_url='https://shopper.example/v1',
        env_url='http://127.0.0.1:5700',
        seed=20260823,
        experiment_name='formal-test',
        resume_from_checkpoint=None,
        dry_run=True,
        preflight_only=False,
    )


def test_formal_launcher_freezes_one_500_update_native_run(tmp_path):
    command = build_command(arguments(tmp_path))
    assert TOTAL_UPDATES == 500
    expected = {
        '--require-clean-git', '--logger', 'swanlab', '--reward-profile', 'none',
        'trainer.n_gpus_per_node=4', 'trainer.total_training_steps=500',
        'trainer.save_freq=25', 'trainer.test_freq=50',
        'trainer.val_before_train=true',
        'actor_rollout_ref.model.use_remove_padding=true',
        'actor_rollout_ref.model.use_fused_kernels=true',
        'actor_rollout_ref.model.use_liger=true',
        'actor_rollout_ref.model.fused_kernel_options.impl_backend=torch',
        'actor_rollout_ref.actor.optim.lr_warmup_steps=-1',
        'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03',
        'actor_rollout_ref.actor.optim.lr_scheduler_type=constant',
    }
    assert expected.issubset(set(command))
    rendered = ' '.join(command)
    assert 'bounded-v1' not in rendered
    assert 'shopping_scheduler' not in rendered


def test_formal_resume_keeps_500_step_target(tmp_path):
    args = arguments(tmp_path)
    args.dry_run = False
    args.resume_from_checkpoint = Path('global_step_250')
    command = build_command(args)
    assert '--resume-from-checkpoint' in command
    assert 'trainer.total_training_steps=500' in command
