"""
Run plan for the revision experiments.

Every run is trained by train.py into checkpoints/<sub>/<tag>/ and selected on the VALIDATION split
(best Rank@1 under the run's --eval_weights). The test split is only touched by run_revision.py --stage eval.

Scoring modes used at test time:
  - "val_selected": fusion weights chosen by grid search on the validation split of the same run
  - "global":       w = (1, 0, 0), only the T_full <-> V_video branch
  - "default":      w = (0.50, 0.35, 0.15), the a-priori weights of the previous submission
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

SEEDS = (42, 1234, 2025)
ABLATION_SUB = "TVPReid-PRID"
EPOCHS = {"TVPReid-PRID": 30, "TVPReid-iLIDs": 50, "TVPReid-Duke": 30}   # same as the seed-42 runs in reports/
COMMON_ARGS = ["--lr", "0.0001", "--batch_size", "16"]

# Training recipes. "v1" is the submitted model (no extra flags). "v2" fixes what limits the small subsets:
# no two captions of a video in one batch (false negatives), learnable temperature, stronger weight decay, and
# best.pth chosen on the mean of val R@1/R@5/R@10 instead of the noisy R@1 peak. The cross-batch queue (--queue_size)
# is left out: on PRID it collapsed training as soon as it was switched on (val R@1 fell to chance, 1/71). "v2s" trains every subset on its own with these flags;
# "v2" first trains one model on the train splits of all three subsets ("joint", sub_dataset all, validation =
# the three val splits) and fine-tunes it per subset. Nothing here looks at the test split.
V2_FLAGS = ["--unique_video_batches", "--learnable_tau", "--weight_decay", "0.05",
            "--select_metric", "mean_r1_5_10"]
# "v2j2": the v2 fine-tunes started from the continued joint model (joint2) instead of the 15-epoch one.
# "v2s3d": the v2 fine-tunes with a Kinetics-400 S3D clip feature added to the motion stream (MFGF's motion
# encoder family); started from the same 15-epoch joint model, so the only change against "v2" is S3D.
# "m2s": every subset trained on its own (as MFGF) with the MFGF backbones: Frozen-in-Time Space-Time ViT on 4
# discrete frames + S3D on 16 consecutive frames + BERT (--backbone mfgf), with the v2 training flags.
MFGF_FLAGS = V2_FLAGS + ["--backbone", "mfgf"]
# "m2f": m2s with the real Frozen-in-Time CC-3M+WebVid-2M checkpoint (m2s fell back to timm ImageNet-21k because the
# Oxford server timed out); the notebook runs it with PBL_REQUIRE_FROZEN=1 so a fallback aborts the run.
# "m3": m2s + fixes for what made the components lose in the m2s ablation (all flags off by default in train.py):
#   --clip_flip         flip whole clips (+ left/right swap in the caption), not every frame on its own
#   --text_pos          identical Q_app / Q_mot strings in a batch are positives, not negatives
#   --dsca_mode clause  whole clauses instead of keyword bags for Q_app / Q_mot
#   --temporal_head_lr  the zero-initialised temporal attention learns at the head lr (1e-4), not 2e-5
#   --mim_mode centered L_MIM on batch-centred V_app / V_mot
#   --ema 0.99          validation and best.pth use EMA weights (less noise when selecting on a small val split)
# Like m2s it runs with PBL_NO_FROZEN=1 (timm ImageNet-21k ViT); evaluate with --grid_step 0.1.
M3_FLAGS = MFGF_FLAGS + ["--clip_flip", "--text_pos", "--dsca_mode", "clause", "--temporal_head_lr",
                         "--mim_mode", "centered", "--ema", "0.99"]
RECIPES = {"v1": [], "v2s": V2_FLAGS, "v2": V2_FLAGS, "v2j2": V2_FLAGS, "v2s3d": V2_FLAGS + ["--s3d_motion"],
           "m2s": MFGF_FLAGS, "m2f": MFGF_FLAGS, "m3": M3_FLAGS, "m3j": M3_FLAGS, "m3c": M3_FLAGS}
# "m3c": m3j with one more stage in front: text-image person pre-training on the CUHK-PEDES-style set
# (experiments/pretrain_cuhk.py, sub "cuhk", tag m3c_cuhk_seed42) -> joint run on the three TVPReid train splits
# (init_from the CUHK model) -> per-subset fine-tune. External data: must be stated when reporting.
CUHK_SUB = "cuhk"
CUHK_ARGS = ["--epochs", "4", "--max_hours", "3.0", "--batch_size", "64"]
# "m3j": m3, but each subset is fine-tuned from ONE model trained on the train splits of all three subsets
# (sub_dataset all, seed 42, tag m3j_joint_seed42). The seeds only differ in the fine-tune -> say so when reporting.
M3J_JOINT_SEED = 42
M3J_JOINT_EPOCHS = 6
M3J_FINETUNE_EPOCHS = {"TVPReid-PRID": 10, "TVPReid-iLIDs": 10, "TVPReid-Duke": 6}
EPOCHS_M2S = {"TVPReid-PRID": 30, "TVPReid-iLIDs": 30, "TVPReid-Duke": 20}
JOINT_SUB = "all"
JOINT_EPOCHS = 15
# "joint2": the joint model trained further (init_from its best.pth, fresh optimizer, lower lr). The joint val curve
# was still rising at epoch 15; whether joint2 replaces joint is decided on the validation split.
JOINT2_EPOCHS = 10
JOINT2_ARGS = ["--lr", "3e-05"]
JOINT_CHAIN = {"joint2": "joint", "joint3": "joint2"}      # continued joint run -> the run it starts from
FINETUNE_EPOCHS = {"TVPReid-PRID": 15, "TVPReid-iLIDs": 20, "TVPReid-Duke": 10}

NO_TEXT_LOSSES = ["--no_dsca", "--lambda_app", "0", "--lambda_mot", "0", "--lambda_mim", "0"]
GLOBAL_SELECTION = ["--eval_weights", "1", "0", "0"]

# name -> (train.py flags, short description). "full" is the main model (no flags).
ABLATION_RUNS = {
    "cum1": (["--no_pretrained_vit", "--no_temporal_attn"] + NO_TEXT_LOSSES + GLOBAL_SELECTION,
             "baseline: ViT from scratch, no temporal attention, full-sentence BERT, global score only"),
    "cum2": (["--no_temporal_attn"] + NO_TEXT_LOSSES + GLOBAL_SELECTION, "+ ImageNet-1K spatial init"),
    "cum3": (NO_TEXT_LOSSES + GLOBAL_SELECTION, "+ divided space-time attention"),
    "no_mim": (["--lambda_mim", "0"], "full model without L_MIM (= cumulative step 4 when scored globally)"),
    "no_init": (["--no_pretrained_vit"], "full model without ImageNet-1K init"),
    "no_temporal": (["--no_temporal_attn"], "full model without temporal attention"),
    "no_dsca": (["--no_dsca"], "full model without DSCA decomposition (T_app = T_mot = T_full)"),
}

# Cumulative table (Table 6): row -> (run name, scoring mode)
CUMULATIVE_ROWS = [
    ("cum1", "global"),
    ("cum2", "global"),
    ("cum3", "global"),
    ("no_mim", "global"),       # + DSCA (Q_app, Q_mot; L_app, L_mot), no MIM
    ("full", "global"),         # + L_MIM
    ("full", "val_selected"),   # + multi-level scoring = full SpaceTime-DSCA
]

# Leave-one-out table (Table 6b): row -> (run name, scoring mode); first row is the reference
LEAVE_ONE_OUT_ROWS = [
    ("full", "val_selected", "full"),
    ("no_init", "val_selected", "no_init"),
    ("no_temporal", "val_selected", "no_temporal"),
    ("no_dsca", "val_selected", "no_dsca"),
    ("no_mim", "val_selected", "no_mim"),
    ("full", "global", "no_multilevel"),
]


@dataclass
class Run:
    sub: str
    name: str          # "full", "joint" or an ABLATION_RUNS key
    seed: int
    flags: List[str] = field(default_factory=list)
    recipe: str = "v1"

    @property
    def tag(self) -> str:
        base = {"full": f"seed{self.seed}", "joint": f"joint_seed{self.seed}", "pretrain": f"cuhk_seed{self.seed}",
                "joint2": f"joint2_seed{self.seed}", "joint3": f"joint3_seed{self.seed}"}.get(self.name, f"abl_{self.name}_seed{self.seed}")
        return base if self.recipe == "v1" else f"{self.recipe}_{base}"

    @property
    def init_run(self) -> Optional["Run"]:
        """The joint run this run is fine-tuned from (recipe v2), else None."""
        if self.recipe == "m3j":
            return None if self.name == "joint" else Run(JOINT_SUB, "joint", M3J_JOINT_SEED, recipe="m3j")
        if self.recipe == "m3c":
            if self.name == "pretrain":
                return None
            if self.name == "joint":
                return Run(CUHK_SUB, "pretrain", M3J_JOINT_SEED, recipe="m3c")
            return Run(JOINT_SUB, "joint", M3J_JOINT_SEED, recipe="m3c")
        if self.name in JOINT_CHAIN:
            return Run(JOINT_SUB, JOINT_CHAIN[self.name], self.seed, recipe="v2")
        if self.recipe in ("v2", "v2s3d") and self.name != "joint":
            return Run(JOINT_SUB, "joint", self.seed, recipe="v2")
        if self.recipe == "v2j2":
            return Run(JOINT_SUB, "joint2", self.seed, recipe="v2")
        return None

    @property
    def epochs(self) -> int:
        if self.recipe in ("m3j", "m3c"):
            if self.name == "pretrain":
                return 1                     # pretrain_cuhk.py writes {"epochs": [1]} when it is finished
            return M3J_JOINT_EPOCHS if self.name == "joint" else M3J_FINETUNE_EPOCHS[self.sub]
        if self.name == "joint":
            return JOINT_EPOCHS
        if self.name in JOINT_CHAIN:
            return JOINT2_EPOCHS
        if self.recipe in ("m2s", "m2f", "m3"):
            return EPOCHS_M2S[self.sub]
        return FINETUNE_EPOCHS[self.sub] if self.recipe in ("v2", "v2j2", "v2s3d") else EPOCHS[self.sub]

    def train_args(self) -> List[str]:
        return (["--sub_dataset", self.sub, "--epochs", str(self.epochs), "--seed", str(self.seed), "--tag", self.tag]
                + COMMON_ARGS + RECIPES[self.recipe] + list(self.flags)
                + (JOINT2_ARGS if self.name in JOINT_CHAIN else []))


LEAVE_ONE_OUT_RUNS = ("no_init", "no_temporal", "no_dsca", "no_mim")      # required by the review (Table 6b)
CUMULATIVE_ONLY_RUNS = ("cum1", "cum2", "cum3")                            # only needed to re-measure Table 6


def build_runs(subs, seeds, ablation_seeds, ablations: bool = True, ablation_names=None, recipe: str = "v1") -> List[Run]:
    runs = []
    if recipe == "v2":                       # the joint model of a seed is trained before its fine-tunes
        runs += [Run(JOINT_SUB, "joint", s, recipe=recipe) for s in seeds]
    if recipe == "m3j":                      # one joint model (seed 42) shared by every fine-tune seed
        runs.append(Run(JOINT_SUB, "joint", M3J_JOINT_SEED, recipe=recipe))
    if recipe == "m3c":                      # CUHK pre-training -> joint -> fine-tunes
        runs.append(Run(CUHK_SUB, "pretrain", M3J_JOINT_SEED, recipe=recipe))
        runs.append(Run(JOINT_SUB, "joint", M3J_JOINT_SEED, recipe=recipe))
    runs += [Run(sub, "full", s, recipe=recipe) for sub in subs for s in seeds]
    if ablations and ABLATION_SUB in subs:
        names = list(ablation_names or ABLATION_RUNS)
        for s in ablation_seeds:
            if s not in seeds:
                if recipe == "v2":
                    runs.append(Run(JOINT_SUB, "joint", s, recipe=recipe))
                runs.append(Run(ABLATION_SUB, "full", s, recipe=recipe))
            for name in names:
                runs.append(Run(ABLATION_SUB, name, s, ABLATION_RUNS[name][0], recipe=recipe))
    return runs
