#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "deliverables" / "project_groupID.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.9",
        },
    }

    nb.cells = [
        md(
            """
            # DURCL Course Project Notebook

            This notebook is the single-file deliverable for the Adaptive Environment Selection course project.
            It is organized into three parts:

            1. **Training pipeline**: dataset access, project-specific training code, scheduler implementation, and the exact heavy-training command path.
            2. **Evaluation pipeline**: checked-in result aggregation by default, with an optional checkpoint-based evaluation entrypoint.
            3. **Result interpretation**: tables, figures, toy sanity checks, and the written analysis used in the report.

            The notebook is designed to be **self-bootstrapping**:

            - If it is run inside a cloned copy of the repository, it uses the local files directly.
            - If it is run as a standalone notebook, it can download the repository snapshot from GitHub.
            - Large artifacts such as model checkpoints are intentionally loaded by link rather than embedded in the notebook.

            Heavy RL training is **not executed by default** because it requires a compatible multi-GPU environment, but the code path and command construction are included here so the full workflow is documented in one place.
            """
        ),
        code(
            """
            from __future__ import annotations

            import importlib
            import json
            import math
            import os
            import shutil
            import subprocess
            import sys
            import textwrap
            import urllib.request
            import zipfile
            from pathlib import Path


            def ensure_packages(packages: list[str]) -> None:
                missing = []
                for pkg in packages:
                    module = pkg.split("==", 1)[0].split(">=", 1)[0]
                    module = {"pillow": "PIL", "pyarrow": "pyarrow"}.get(module, module)
                    try:
                        importlib.import_module(module)
                    except Exception:
                        missing.append(pkg)
                if missing:
                    print("Installing missing packages:", missing)
                    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


            ensure_packages([
                "pandas",
                "matplotlib",
                "pillow",
                "pyarrow",
                "reportlab",
                "pytest",
            ])

            import pandas as pd
            import matplotlib.pyplot as plt
            from IPython.display import Code, Image, Markdown, display


            plt.style.use("seaborn-v0_8-whitegrid")
            pd.set_option("display.max_colwidth", 120)
            """
        ),
        code(
            """
            RUN_UNIT_TESTS = True
            RUN_HEAVY_TRAINING = False
            RUN_CHECKPOINT_EVAL = False
            REGENERATE_FIGURES = False
            RERUN_TOY_SIMULATION = False

            REPO_URL = "https://github.com/kkkzzz0101/Adaptive-Env-Selection.git"
            REPO_ZIP_URL = "https://github.com/kkkzzz0101/Adaptive-Env-Selection/archive/refs/heads/main.zip"
            RAW_BASE = "https://raw.githubusercontent.com/kkkzzz0101/Adaptive-Env-Selection/main"

            HF_MODEL_ID = os.environ.get(
                "AES_HF_MODEL_ID",
                "zkkk452/adaptive-env-selection-checkpoint",
            ).strip()
            CHECKPOINT_ARCHIVE_URL = os.environ.get("AES_CHECKPOINT_ARCHIVE_URL", "").strip()
            LOCAL_MODEL_PATH = os.environ.get("AES_LOCAL_MODEL_PATH", "").strip()
            HF_TOKEN = (
                os.environ.get("HF_TOKEN", "").strip()
                or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
            )

            print({
                "RUN_UNIT_TESTS": RUN_UNIT_TESTS,
                "RUN_HEAVY_TRAINING": RUN_HEAVY_TRAINING,
                "RUN_CHECKPOINT_EVAL": RUN_CHECKPOINT_EVAL,
                "REGENERATE_FIGURES": REGENERATE_FIGURES,
                "RERUN_TOY_SIMULATION": RERUN_TOY_SIMULATION,
            })
            """
        ),
        code(
            """
            NOTEBOOK_CWD = Path.cwd()


            def github_raw_url(path: str) -> str:
                return f"{RAW_BASE}/{path}"


            def find_repo_root(start: Path) -> Path | None:
                start = start.resolve()
                for candidate in [start, *start.parents]:
                    if (candidate / "README.md").exists() and (candidate / "paper" / "deliverables").exists():
                        return candidate
                return None


            def download_repo_snapshot(dest_dir: Path) -> Path:
                dest_dir.mkdir(parents=True, exist_ok=True)
                zip_path = dest_dir / "Adaptive-Env-Selection-main.zip"
                extract_dir = dest_dir / "Adaptive-Env-Selection-main"
                if not extract_dir.exists():
                    print(f"Downloading repository snapshot from {REPO_ZIP_URL}")
                    urllib.request.urlretrieve(REPO_ZIP_URL, zip_path)
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(dest_dir)
                return extract_dir


            ROOT = find_repo_root(NOTEBOOK_CWD)
            if ROOT is None:
                ROOT = download_repo_snapshot(NOTEBOOK_CWD / "aes_notebook_runtime")
                print(f"Using downloaded repository at: {ROOT}")
            else:
                print(f"Using local repository at: {ROOT}")

            sys.path.insert(0, str(ROOT / "src"))
            sys.path.insert(0, str(ROOT / "scripts"))
            """
        ),
        code(
            """
            asset_rows = [
                {"artifact": "Repository root", "path": str(ROOT), "exists": ROOT.exists()},
                {"artifact": "Notebook", "path": str(ROOT / "paper/deliverables/project_groupID.ipynb"), "exists": (ROOT / "paper/deliverables/project_groupID.ipynb").exists()},
                {"artifact": "Training data", "path": str(ROOT / "references/sec/data"), "exists": (ROOT / "references/sec/data").exists()},
                {"artifact": "Scheduler source", "path": str(ROOT / "src/scheduler/adaptive_curriculum_scheduler.py"), "exists": (ROOT / "src/scheduler/adaptive_curriculum_scheduler.py").exists()},
                {"artifact": "Eval probe", "path": str(ROOT / "scripts/sec_inference_probe.py"), "exists": (ROOT / "scripts/sec_inference_probe.py").exists()},
                {"artifact": "Result report", "path": str(ROOT / "docs/result_report.md"), "exists": (ROOT / "docs/result_report.md").exists()},
                {"artifact": "Paper figures", "path": str(ROOT / "paper/figures"), "exists": (ROOT / "paper/figures").exists()},
            ]
            display(pd.DataFrame(asset_rows))
            """
        ),
        md(
            """
            ## Part I. Training Pipeline

            The heavy RL runs in this project use the DUMP/verl path together with the project scheduler and SEC-format datasets.
            This section keeps all the project-specific training ingredients in one notebook:

            - the checked-in datasets,
            - the project scheduler implementation,
            - the data-preparation and training-entry scripts,
            - and a portable reconstruction of the heavy training command.

            The notebook documents the full training path, but does **not** launch RL training unless `RUN_HEAVY_TRAINING=True`.
            """
        ),
        code(
            """
            data_root = ROOT / "references" / "sec" / "data"
            data_files = sorted(data_root.rglob("*.parquet"))

            import pyarrow.parquet as pq

            data_rows = []
            for path in data_files:
                rel = path.relative_to(ROOT).as_posix()
                data_rows.append({
                    "file": rel,
                    "rows": pq.read_metadata(path).num_rows,
                    "download_url": github_raw_url(rel),
                })

            data_df = pd.DataFrame(data_rows).sort_values("file").reset_index(drop=True)
            display(data_df)
            print(f"Total parquet files: {len(data_df)}")
            """
        ),
        code(
            """
            def show_source(rel_path: str, start: int = 1, end: int | None = None) -> None:
                path = ROOT / rel_path
                lines = path.read_text(encoding="utf-8").splitlines()
                if end is None:
                    end = len(lines)
                snippet = "\\n".join(f"{idx:4d}: {line}" for idx, line in enumerate(lines[start - 1:end], start))
                print(f"--- {rel_path} ---")
                print(snippet)


            show_source("scripts/prepare_sec4_random_dataset.py", 1, 140)
            """
        ),
        code(
            """
            show_source("src/scheduler/adaptive_curriculum_scheduler.py", 1, 220)
            """
        ),
        code(
            """
            show_source("scripts/run_baseline_random_dump_2gpu.sh", 1, 120)
            """
        ),
        code(
            """
            if RUN_UNIT_TESTS:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "tests/test_scheduler_unit.py", "-q"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(result.stdout)
            else:
                print("Skipping unit tests. Set RUN_UNIT_TESTS=True to execute the scheduler sanity suite.")
            """
        ),
        code(
            """
            TRAINING_CONFIG = {
                "model_path": os.environ.get("AES_TRAIN_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
                "n_gpus_per_node": int(os.environ.get("N_GPUS_PER_NODE", "2")),
                "total_training_steps": int(os.environ.get("TOTAL_TRAINING_STEPS", "1000")),
                "save_freq": int(os.environ.get("SAVE_FREQ", "200")),
                "test_freq": int(os.environ.get("TEST_FREQ", "200")),
                "train_batch_size": int(os.environ.get("TRAIN_BATCH_SIZE", "32")),
                "rollout_n": int(os.environ.get("ROLLOUT_N", "4")),
                "max_prompt_length": int(os.environ.get("MAX_PROMPT_LENGTH", "1152")),
                "max_response_length": int(os.environ.get("MAX_RESPONSE_LENGTH", "512")),
                "exp_name": os.environ.get("EXP_NAME", "baseline_random_dump_2gpu_1k"),
            }


            def build_heavy_training_command(root: Path, cfg: dict[str, object]) -> tuple[list[str], dict[str, str]]:
                data_root = root / "experiments" / "baselines" / "data_sec4_2gpu_1k" / "mixed"
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_visible_devices"])
                env["PYTHONPATH"] = str(root / "references" / "DUMP") + os.pathsep + env.get("PYTHONPATH", "")
                env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

                cmd = [
                    sys.executable,
                    "-m",
                    "verl.trainer.main_ppo",
                    "algorithm.adv_estimator=grpo",
                    f"data.train_files={data_root / 'train.parquet'}",
                    f"data.val_files={data_root / 'val.parquet'}",
                    f"data.train_batch_size={cfg['train_batch_size']}",
                    "data.val_batch_size=64",
                    "data.enable_curriculum_learning=False",
                    "data.data_source_key=null",
                    f"data.max_prompt_length={cfg['max_prompt_length']}",
                    f"data.max_response_length={cfg['max_response_length']}",
                    f"actor_rollout_ref.model.path={cfg['model_path']}",
                    "actor_rollout_ref.model.use_remove_padding=False",
                    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
                    "actor_rollout_ref.actor.use_dynamic_bsz=True",
                    "actor_rollout_ref.actor.optim.lr=1e-6",
                    f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['train_batch_size']}",
                    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192",
                    "actor_rollout_ref.actor.use_kl_loss=True",
                    "actor_rollout_ref.actor.kl_loss_coef=0.001",
                    "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
                    "actor_rollout_ref.actor.fsdp_config.param_offload=False",
                    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
                    "actor_rollout_ref.ref.fsdp_config.param_offload=False",
                    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
                    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192",
                    "actor_rollout_ref.rollout.name=hf",
                    f"actor_rollout_ref.rollout.n={cfg['rollout_n']}",
                    "actor_rollout_ref.rollout.temperature=1.0",
                    "actor_rollout_ref.rollout.top_p=1.0",
                    "actor_rollout_ref.rollout.top_k=0",
                    "actor_rollout_ref.rollout.do_sample=True",
                    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
                    "+actor_rollout_ref.rollout.micro_batch_size=1",
                    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
                    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192",
                    f"critic.model.path={cfg['model_path']}",
                    f"critic.model.tokenizer_path={cfg['model_path']}",
                    "critic.model.use_remove_padding=False",
                    "critic.model.enable_gradient_checkpointing=False",
                    "critic.optim.lr=1e-5",
                    "critic.ppo_micro_batch_size_per_gpu=1",
                    "critic.forward_micro_batch_size_per_gpu=1",
                    "critic.ppo_max_token_len_per_gpu=12288",
                    "algorithm.kl_ctrl.kl_coef=0.001",
                    "trainer.critic_warmup=0",
                    "trainer.logger=['console']",
                    "trainer.project_name=aes_baseline_dump",
                    f"trainer.experiment_name={cfg['exp_name']}",
                    f"trainer.n_gpus_per_node={cfg['n_gpus_per_node']}",
                    "trainer.nnodes=1",
                    f"trainer.default_local_dir={root / 'checkpoints' / 'aes_baseline_dump' / str(cfg['exp_name'])}",
                    "trainer.default_hdfs_dir=null",
                    f"trainer.save_freq={cfg['save_freq']}",
                    f"trainer.test_freq={cfg['test_freq']}",
                    "trainer.total_epochs=1",
                    "+trainer.val_before_train=False",
                    f"trainer.total_training_steps={cfg['total_training_steps']}",
                ]
                return cmd, env


            cmd, env = build_heavy_training_command(ROOT, TRAINING_CONFIG)
            print("Portable heavy-training command preview:")
            print(" ".join(str(x) for x in cmd[:12]), "...")

            if RUN_HEAVY_TRAINING:
                print("Launching heavy RL training. This requires the original multi-GPU environment and full verl dependencies.")
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            else:
                print("Heavy RL training is skipped by default. Set RUN_HEAVY_TRAINING=True only in a compatible multi-GPU environment.")
            """
        ),
        md(
            """
            ## Part II. Evaluation Pipeline

            The notebook supports two evaluation modes:

            1. **Default**: load the checked-in CSV summaries already stored in the repository.
            2. **Optional checkpoint eval**: load a model checkpoint from Hugging Face, a local path, or an extracted archive and run the SEC inference probe.

            This design keeps the notebook lightweight for grading while still documenting the complete evaluation path.

            The default public checkpoint target is:
            - `zkkk452/adaptive-env-selection-checkpoint`

            At the time this notebook was finalized, the training instance that stores the exported checkpoint was affected by a platform login issue, so this Hugging Face repository is currently a public placeholder that will receive the model files once the instance becomes accessible again.
            """
        ),
        md(
            """
            ### Final Rebucket Run Used In The Submission

            The final notebook should center the **window-fitted linear rebucketing method** rather than the earlier rebucketing variants that still appear in some checked-in reports.

            The authoritative rebucketing result used in the course submission is:

            - `rebucket 80 -> 200`
            - step 100: `math = 0.440`, `zebra = 0.237`
            - step 150: `math = 0.520`, `zebra = 0.237`
            - step 200: `math = 0.580`, `zebra = 0.300`

            The final comparison shown below focuses on the **step-200** result.
            """
        ),
        code(
            """
            final_rebucket = pd.DataFrame([
                {"run": "rebucket_window_linear_80_200", "step": 100, "math": 0.440, "zebra": 0.237},
                {"run": "rebucket_window_linear_80_200", "step": 150, "math": 0.520, "zebra": 0.237},
                {"run": "rebucket_window_linear_80_200", "step": 200, "math": 0.580, "zebra": 0.300},
            ])
            display(final_rebucket)
            """
        ),
        code(
            """
            mz = pd.read_csv(ROOT / "experiments" / "results" / "math_zebra_2data" / "baseline_vs_norebucket_metrics.csv")
            display(mz)

            step200 = mz[mz["step"] == 200].copy()
            if len(step200) == 2:
                baseline = step200[step200["run"] == "baseline_random"].iloc[0]
                scheduler = step200[step200["run"] == "scheduler_no_rebucket"].iloc[0]
                rebucket = final_rebucket[final_rebucket["step"] == 200].iloc[0]

                final_comparison = pd.DataFrame([
                    {
                        "task": "Math",
                        "random_step200": baseline["math_train"],
                        "no_rebucket_step200": scheduler["math_train"],
                        "rebucket_80_200_step200": rebucket["math"],
                        "rebucket_vs_random": rebucket["math"] - baseline["math_train"],
                        "rebucket_vs_no_rebucket": rebucket["math"] - scheduler["math_train"],
                    },
                    {
                        "task": "Zebra",
                        "random_step200": baseline["zebra_train"],
                        "no_rebucket_step200": scheduler["zebra_train"],
                        "rebucket_80_200_step200": rebucket["zebra"],
                        "rebucket_vs_random": rebucket["zebra"] - baseline["zebra_train"],
                        "rebucket_vs_no_rebucket": rebucket["zebra"] - scheduler["zebra_train"],
                    },
                ])
                display(final_comparison)
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            for ax, task in zip(axes, ["math", "zebra"]):
                ax.plot(final_rebucket["step"], final_rebucket[task], marker="o", linewidth=2.2)
                ax.set_title(f"Rebucket 80->200 ({task.capitalize()})")
                ax.set_xlabel("Step")
                ax.set_ylabel("Validation score")
                ax.set_xticks(final_rebucket["step"].tolist())
                ax.set_ylim(0.20 if task == "zebra" else 0.40, 0.62)
                for _, row in final_rebucket.iterrows():
                    ax.text(row["step"], row[task] + 0.01, f"{row[task]:.3f}", ha="center", fontsize=9)

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            def weighted_pass_rate(path: Path) -> float:
                df = pd.read_csv(path)
                return float((df["n"] * df["task_pass_rate"]).sum() / df["n"].sum())


            eval_root = ROOT / "experiments" / "eval_ckpt400"
            rows = []
            for run in ["baseline400", "scheduler400"]:
                for dataset in ["countdown", "zebra", "arc", "math"]:
                    csv_path = eval_root / run / dataset / f"{dataset}_test_summary.csv"
                    rows.append({
                        "run": run,
                        "dataset": dataset,
                        "weighted_task_pass_rate": round(weighted_pass_rate(csv_path), 3),
                        "source_csv": csv_path.relative_to(ROOT).as_posix(),
                    })

            ckpt400_df = pd.DataFrame(rows)
            display(ckpt400_df)
            display(ckpt400_df.pivot(index="dataset", columns="run", values="weighted_task_pass_rate"))
            """
        ),
        code(
            """
            def ensure_eval_dependencies() -> None:
                ensure_packages(["transformers", "torch"])


            def resolve_model_reference(root: Path) -> str:
                if HF_MODEL_ID:
                    return HF_MODEL_ID
                if LOCAL_MODEL_PATH:
                    return LOCAL_MODEL_PATH
                if CHECKPOINT_ARCHIVE_URL:
                    archive_dir = root / "artifacts" / "downloaded_checkpoint"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    archive_path = archive_dir / "checkpoint_archive"
                    if not archive_path.exists():
                        print(f"Downloading checkpoint archive from {CHECKPOINT_ARCHIVE_URL}")
                        urllib.request.urlretrieve(CHECKPOINT_ARCHIVE_URL, archive_path)
                    raise ValueError("Downloaded checkpoint archives should be extracted manually or provided as AES_LOCAL_MODEL_PATH.")
                raise ValueError("No checkpoint source configured. Set AES_HF_MODEL_ID or AES_LOCAL_MODEL_PATH.")


            if RUN_CHECKPOINT_EVAL:
                ensure_eval_dependencies()
                model_ref = resolve_model_reference(ROOT)
                probe_out = ROOT / "experiments" / "notebook_eval"
                env = os.environ.copy()
                env["AES_ROOT"] = str(ROOT)
                env["AES_MODEL_PATH"] = model_ref
                if HF_TOKEN:
                    env["HF_TOKEN"] = HF_TOKEN
                    env["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

                datasets = ["countdown", "zebra", "arc", "math"]
                for dataset in datasets:
                    cmd = [
                        sys.executable,
                        str(ROOT / "scripts" / "sec_inference_probe.py"),
                        "--dataset",
                        dataset,
                        "--model-path",
                        model_ref,
                        "--sec-root",
                        str(ROOT / "references" / "sec" / "data"),
                        "--out-dir",
                        str(probe_out / dataset),
                        "--n-per-difficulty",
                        "10",
                        "--rollouts",
                        "1",
                    ]
                    print("Running:", " ".join(cmd))
                    subprocess.run(cmd, cwd=ROOT, env=env, check=True)

                summaries = sorted(probe_out.rglob("*_summary.csv"))
                display(pd.DataFrame({"summary_csv": [p.relative_to(ROOT).as_posix() for p in summaries]}))
            else:
                print(
                    "Checkpoint eval is disabled. To enable it, set RUN_CHECKPOINT_EVAL=True. "
                    f"The current default Hugging Face target is: {HF_MODEL_ID}"
                )
            """
        ),
        md(
            """
            ## Part III. Result Interpretation

            This section keeps only the report context that still matches the final submission story.
            Some checked-in documents were written before the final `rebucket 80 -> 200` result was available, so the notebook should not treat every older report number as authoritative.

            The current notebook should therefore:

            - use the final `rebucket 80 -> 200` step-200 result as the main rebucketing outcome,
            - use the checked-in random and no-rebucket CSV as the baseline comparison source,
            - and keep older rebucketing writeups only as background context.

            The guiding idea is that the notebook should serve as a compact, inspectable version of the repository for course submission.
            """
        ),
        code(
            """
            checked_in_report_note = pd.DataFrame([
                {
                    "file": "docs/result_report.md",
                    "status": "outdated for final notebook",
                    "reason": "still summarizes the older difficulty-init rebucketing numbers",
                },
                {
                    "file": "docs/course_project_report.md",
                    "status": "outdated for final notebook",
                    "reason": "still highlights the earlier 0.540 / 0.300 acc-init rebucketing result",
                },
                {
                    "file": "README.md",
                    "status": "partially outdated",
                    "reason": "contains older rebucketing highlights and should not override the final 80->200 submission result",
                },
            ])
            display(checked_in_report_note)
            """
        ),
        code(
            """
            if REGENERATE_FIGURES:
                subprocess.run([sys.executable, str(ROOT / "paper" / "scripts" / "make_figures.py")], cwd=ROOT, check=True)
            else:
                print("Using checked-in figures. Set REGENERATE_FIGURES=True to rebuild them.")

            figure_names = [
                "fig_method_pipeline",
                "fig_step200_math_zebra",
                "fig_ucb_score_drift",
                "fig_initial_accuracy_profile",
                "fig_rebucket_composition",
                "fig_inferred_transition_matrix",
                "fig_toy_rebucket_guardrails",
            ]

            for name in figure_names:
                png_path = ROOT / "paper" / "figures" / f"{name}.png"
                display(Markdown(f"### {name}"))
                display(Image(filename=str(png_path)))
            """
        ),
        code(
            """
            if RERUN_TOY_SIMULATION:
                subprocess.run([sys.executable, str(ROOT / "scripts" / "toy_simulation_rebucket.py")], cwd=ROOT, check=True)
            else:
                print("Using checked-in toy-simulation artifacts. Set RERUN_TOY_SIMULATION=True to rerun them.")

            toy_summary = json.loads((ROOT / "artifacts" / "toy_rebucket_summary.json").read_text(encoding="utf-8"))
            display(pd.DataFrame(toy_summary["scenarios"]))
            display(Image(filename=str(ROOT / "artifacts" / "toy_rebucket_simulation.png")))
            """
        ),
        md(
            """
            ## Final Takeaway

            This notebook intentionally separates **heavy training**, **replayable evaluation**, and **checked-in analysis**.
            That matches the reality of the project:

            - the repository already contains the important result tables, diagnostic traces, and paper figures;
            - the notebook can bootstrap the repo and inspect those assets from a single file;
            - checkpoint-based evaluation is supported through links;
            - and full RL retraining remains available as a documented code path for a compatible multi-GPU environment.

            In short, the notebook is meant to be both a readable submission artifact and a faithful index to the complete project workflow.
            """
        ),
    ]
    return nb


def main() -> None:
    nb = build_notebook()
    OUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
