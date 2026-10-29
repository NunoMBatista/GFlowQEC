"""The networks table of the paper: its measured cells.

Parameter counts are measured by building each network exactly as the runs built it and counting its
trainable parameters. Update counts, the reward exponent beta and the log Z warm start are read from the
stamps of the shipped checkpoints. The same beta values are restated in the text ("beta = 64, with none
qualifying, at n = 300 ... beta = 8 at gamma = +-1"); they are printed once, here.

Where the paper prints one value for a group of files (every beta, every parameter count, 315,280, 252,224,
116,972), the files must agree with each other: a disagreement stops the script with an assertion instead of
printing an average. Ranges are min-max over the files of that cell.

Reads only data/models, data/n300 and data/n180 (through gflowqec.paths.DATA). No oracle call; about 4 s on a
laptop. Run from the repo root:
    python -m readouts.nets
"""
import json

import numpy as np
import torch

from gflowqec.codes import RelTables, conj_table, group_for, identity_index
from gflowqec.oracle import key_beta
from gflowqec.paths import DATA
from gflowqec.policy import OnlinePolicy, TupleTransformer, derive_beta
from gflowqec.rivals import MittenPPO, MittenSoftQ
from gflowqec.screens import ScreenTransformer

MODELS = DATA / "models"
G180, G300 = "C6xS3", "C10xS3"      # the groups of the two arenas: |G| = 36 gives n = 180, |G| = 60 gives n = 300
SEEDS3, SEEDS6, SEEDS10 = (0, 1, 2), range(6), range(10)


def load(path):
    """A checkpoint as saved by the runs (a plain dict of tensors and stamps), read on the CPU."""
    return torch.load(path, weights_only=False, map_location="cpu")


def count(module):
    """Trainable parameters of a module (buffers such as the transformer's causal mask are not parameters)."""
    return sum(p.numel() for p in module.parameters())


def one(values, what):
    """The single value a group of files must share; stops if they disagree."""
    values = list(values)
    assert len(set(values)) == 1, f"{what}: the files disagree: {sorted(set(values))}"
    return values[0]


def span(values):
    """'min-max' of a list of integers, with thousands separators."""
    return f"{min(values):,}-{max(values):,}"


# ----------------------------------------------------------------------------- the networks, built as run
# The online policy, the soft Q / Munchausen DQN learner and the PPO actor share one trunk: a perceptron with
# two hidden layers of 64 over the state encoding (one 0/1 vector over G per sum, four sums, plus a one-hot of
# the slot: 4|G| + 10 inputs) with |G| outputs per slot. The online policy adds one scalar log Z. Soft Q keeps a
# target copy of its network (refreshed, never trained); PPO adds a critic of the same trunk shape with one
# output. The seed does not change any count.
size = {}
for g in (G180, G300):
    n = len(group_for(g)[1])
    ident = identity_index(g)
    online = OnlinePolicy(n, hidden=64, seed=0, identity=ident, conj=conj_table(g))
    softq = MittenSoftQ(n, hidden=64, seed=0, identity=ident, conj=conj_table(g), alpha_m=0.0)
    ppo = MittenPPO(n, hidden=64, seed=0, identity=ident)
    size[g] = {"enc": online.enc_dim(), "online": count(online.net), "logZ": online.logZ.numel(),
               "q": count(softq.net), "target": count(softq.tnet),
               "actor": count(ppo.net), "critic": count(ppo.vnet)}
    # the three learners read the same encoding with the same trunk
    assert softq.enc_dim() == ppo.enc_dim() == size[g]["enc"]
s180, s300 = size[G180], size[G300]

# GFlowQEC-offline: the decoder-only transformer over the ten slots, conditioned on the bias (n = 180 only).
n180 = len(group_for(G180)[1])
torch.manual_seed(0)
transformer = TupleTransformer(n180)
offline_params = count(transformer)

# The supervised screen: the same transformer body without the causal mask, which also reads the relational
# counts of the tuple (a 504-entry vector, codes.RelTables) through one linear map.
T = RelTables(G180)
torch.manual_seed(0)
screen_params = count(ScreenTransformer(n180, rel_dim=T.dim))

# ----------------------------------------------------------------------------- the shipped checkpoints
# GFlowQEC-offline, ten seeds: trained for a fixed number of updates on the stored forms.
offline = [load(MODELS / f"r177_transformer_w18_b256_x20_s{s}.pt") for s in SEEDS10]
for c in offline:
    # the saved state holds the trained parameters plus the 100-entry causal mask, which is a buffer
    assert sum(v.numel() for k, v in c["state"].items() if k != "causal") == offline_params
offline_steps = one((c["steps"] for c in offline), "offline steps")
passes = one((round(c["visits"] / c["rows"]) for c in offline), "offline passes")    # stored forms visited / stored forms
offline_rows = one((c["rows"] for c in offline), "offline training rows")

# The supervised screen, ten seeds; its update count was chosen on held-out stored forms (the choice file).
screens = [load(MODELS / f"r177_mscreen_T1tfr_st252224_s{s}.pt") for s in SEEDS10]
choice = json.load(open(DATA / "n180" / "screen180match_choice.json"))
assert one((c["params"] for c in screens), "screen params stamp") == screen_params
screen_steps = one([c["steps"] for c in screens] + [choice["step"]], "screen steps")
screen_rows = one((c["rows"] for c in screens), "screen training rows")
training_rows = one((offline_rows, screen_rows), "offline and screen training rows")

# GFlowQEC-online at n = 300, depth 127: the 12 runs of the two deployments (_d127 and its twin _c127).
# One update is one saved loss, so the update count is the length of the loss list.
online300 = ([f"mitten_lersearch_lsgfn_C10xS3_b0.8_t0.05-0.67_w_k1i127det_s{s}_d127" for s in SEEDS6]
             + [f"mitten_lersearch_lsgfn_C10xS3_b0.8_t0.05-0.67_w_k1i127detC_s{s}_c127" for s in SEEDS6])
online300_ckpt = [load(MODELS / f"{stem}.pt") for stem in online300]
online300_updates = [len(c["losses"]) for c in online300_ckpt]

# GFlowQEC-online at n = 180 (depth 7), one bias per run: three seeds per bias.
online180_ckpt = [load(MODELS / f"r174_lsgfn_C6xS3_e10{b}_b3000_k1i7det_cert_s{s}.pt") for b in "XZ" for s in SEEDS3]

# Soft Q-learning at n = 300 (six seeds); soft Q and Munchausen DQN at n = 180 (three learners, two biases).
q300 = [load(MODELS / f"mitten_lersearch_sdqn_C10xS3_b0.8_t0.05-0.67_w_s{s}.pt") for s in SEEDS6]
q180 = [load(MODELS / f"{arm}_C6xS3_e10{b}_b3000_{tail}_s{s}.pt")
        for arm, tail in (("r209_sdqn", "cert"), ("r174_mdqn", "cert"), ("r209_mdqnls", "k1i7det_cert"))
        for b in "XZ" for s in SEEDS3]

# PPO at n = 300 (entropy weight 0.01, six seeds) and at n = 180 (two entropy weights, two biases, three seeds).
# A PPO update is one batch of 8 trajectories, trained 4 epochs; one saved loss per batch.
ppo300 = [load(MODELS / f"mitten_lersearch_ppo_C10xS3_b0.8_t0.05-0.67_w_s{s}.pt") for s in SEEDS6]
ppo180 = [load(MODELS / f"r209_ppo_C6xS3_e10{b}_b3000_ent{e}_cert_s{s}.pt")
          for b in "XZ" for e in ("0.01", "0.1") for s in SEEDS3]

# ----------------------------------------------------------------------------- beta
# The reward exponent rule (policy.derive_beta): the smallest beta on the grid 1 ... 64 that puts at
# least 0.8 of the reward s^beta on the top tenth of the calibration scores, or 64 if none does.
#
# n = 300: the rule ran on each run's own 200 calibration draws (uniform, the first 200 rows of the run dump,
# marked src = 2). The score of a row is the oracle's -log(p_L + 1 / (k * shots)), with p_L the per-qubit
# logical error rate. "None qualifies" means that even beta = 64 leaves less than 0.8 of the reward on the top
# tenth, so 64 is the fallback, not a grid value that passed.
none_qualifies = 0
for stem, ckpt in zip(online300, online300_ckpt):
    d = json.load(open(DATA / "n300" / f"{stem}.json"))
    calib = [i for i in range(200) if d["src"][i] == 2]
    assert len(calib) == 200, f"{stem}: the first 200 rows are not all calibration draws"
    vals = [-np.log(d["ler_pq"][i] + 1.0 / (max(d["k"][i], 1) * d["shots"])) for i in calib]
    assert derive_beta(vals) == 64.0 == ckpt["beta"], f"{stem}: rule and stamp disagree"
    # the rule's own test, at beta = 64
    v = np.maximum(np.asarray(vals, float), 1e-2)
    v = v / v.max()
    cut = np.quantile(v, 0.9)
    w = v ** 64
    if w[v >= cut].sum() / w.sum() < 0.8:
        none_qualifies += 1

# n = 180: one beta per bias for every learner, from the rule applied to the uniform reference sample and its
# mirror images (oracle.key_beta); the offline policy stores it per bias as {+1: beta, -1: beta}.
beta180 = one([key_beta(1.0), key_beta(-1.0)], "reference-sample beta")
assert one((c["beta"] for c in online180_ckpt), "online 180 beta") == beta180
offline_beta = one((b for c in offline for b in (c["beta"][1.0], c["beta"][-1.0])), "offline beta")
assert offline_beta == beta180
beta_online300 = one((c["beta"] for c in online300_ckpt), "online 300 beta")
beta_q300 = one((c["beta"] for c in q300), "Q 300 beta")
beta_q180 = one((c["beta"] for c in q180), "Q 180 beta")
beta_ppo300 = one((c["beta"] for c in ppo300), "PPO 300 beta")
beta_ppo180 = one((c["beta"] for c in ppo180), "PPO 180 beta")

# The offline policy's log Z warm start (the policy appendix's warm-start equation on the stored forms),
# stamped per bias; the table prints the mean of the two.
logz0 = one((f"{(c['logz0'][1.0] + c['logz0'][-1.0]) / 2:.3f}" for c in offline), "offline log Z_0")


def b(x):
    """A beta as the table prints it (they are whole numbers)."""
    return f"{x:.0f}"


# ----------------------------------------------------------------------------- the table
print(f"Table B.1 input encoding: online, Q and PPO 180: {s180['enc']}; 300: {s300['enc']}; "
      f"screen relational counts {T.dim}")
print(f"Table B.1 parameters, online: 180: {s180['online']:,}; 300: {s300['online']:,}; "
      f"+{one((s180['logZ'], s300['logZ']), 'log Z size')} for log Z")
print(f"Table B.1 parameters, GFlowQEC-offline: {offline_params:,}")
print(f"Table B.1 parameters, supervised screen: {screen_params:,}")
print(f"Table B.1 parameters, soft Q and Munchausen DQN: 180: {s180['q']:,}; 300: {s300['q']:,}; "
      f"target copy {s180['target']:,} / {s300['target']:,}, not trained")
print(f"Table B.1 parameters, PPO: actor 180: {s180['actor']:,}; 300: {s300['actor']:,}; "
      f"critic 180: {s180['critic']:,}; 300: {s300['critic']:,}")
print(f"Table B.1 updates, online 300 depth 127: {span(online300_updates)}")
print(f"Table B.1 updates, offline: {offline_steps:,}; about {passes} passes")
print(f"Table B.1 updates, supervised screen: {screen_steps:,}, chosen on held-out stored forms")
print(f"Table B.1 updates, Q 300: {span([len(c['losses']) for c in q300])}")
print(f"Table B.1 updates, PPO 300: {span([len(c['losses']) for c in ppo300])} batches (entropy 0.01); "
      f"180: {span([len(c['losses']) for c in ppo180])}")
print(f"Table B.1 training data, offline and screen: {training_rows:,} stored forms")
print(f"Table B.1 beta, online: 180: {b(beta180)}; 300: {b(beta_online300)}, "
      f"none qualifies in {none_qualifies} of {len(online300)} runs")
print(f"Table B.1 beta, offline: {b(offline_beta)} at gamma = +1 and -1")
print(f"Table B.1 beta, Q: 300: {b(beta_q300)}; 180: {b(beta_q180)}; PPO: 300: {b(beta_ppo300)}; 180: {b(beta_ppo180)}")
print(f"Table B.1 log Z_0, offline: {logz0}")
print(f"Table B.1 seeds and models, GFlowQEC-offline: {SEEDS10[0]}-{SEEDS10[-1]}, {len(offline)} models; "
      f"supervised screen: {SEEDS10[0]}-{SEEDS10[-1]}, {len(screens)} models")
