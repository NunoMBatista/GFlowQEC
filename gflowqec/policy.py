"""The samplers of the paper and the losses they train with.

GFlowQEC-offline (n = 180). `TupleTransformer` is a small decoder-only transformer that writes a code one group
element at a time (the ten slots of codes.ENTRIES) and is trained offline, by trajectory balance, on the stored
forms of earlier runs. Its MaxEnt-RL rival, offline soft Q-learning, trains the same network on the same data
with `soft_bellman_loss` instead. `augment` is the symmetry augmentation both see, and `logp_of_trajs` lets the
trained policy score candidates it did not draw itself.

GFlowQEC-online (n = 180 and 300). `OnlinePolicy` is a two-layer perceptron over the same ten slots, trained by
trajectory balance while it pays the oracle. `arm_gflowqec_online` wraps it in the local search: every drawn
code is refined by masking one slot and re-emitting it, and the change is kept only when the score improves.
`arm_gfn` is the same sampler without the local search (the plain GFlowNet row).

The n = 36 sampler. `GFN` builds a two-block code over Z_6 x Z_3 one grid cell at a time and is graded against
the exact answer key.

Random streams. Every network draws its initial weights from torch's global generator, so the order in which
things are built matters. OnlinePolicy re-seeds torch at the top of its constructor, and the offline training
script builds an OnlinePolicy twin (for its closed-form log Z warm start) BEFORE the transformer: the
transformer's initial weights depend on that twin having been built first, with its relational inputs.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from gflowqec.codes import (ENTRIES, HAS_IDENTITY, canonicalise, conj_table, draw_gen, generates_g, group_for,
                            identity_index, mirror, rel_features)
from gflowqec.oracle import cond_vec

# ============================================================================================ the ten-slot object
# A = [g1+g2+g3, e+g4+g5], B = [h1+h2+h3, e+h4+h5]: slots 0-4 hold A's terms, slots 5-9 hold B's.
BLOCKS = ENTRIES                                   # the slots of each unordered entry, emitted in this order
BLOCK_OF = {p: bi for bi, blk in enumerate(BLOCKS) for p in blk}
FORCED = (0, 5)                                    # gauge order: g1 = h1 = e, so slots 0 and 5 hold the identity

# GFlowQEC-offline
LAMS = (1.0, -1.0)                                 # the two biases it is conditioned on: Z-biased, then X-biased
BASE_STEPS = 15764                                 # the update count of the first training run (capped by wall clock)
# the training budget as a gradient-step count, keyed by the token the model file names carry
# (x20 = twenty times BASE_STEPS, the policy the paper reports)
SCALE_STEPS = {"x20": 20 * BASE_STEPS}
TARGET_EVERY = 100                                 # soft Q-learning: hard refresh of the target copy every 100 steps

# GFlowQEC-online's local search
LS_FREE_POS = (1, 2, 3, 4, 6, 7, 8, 9)             # the slots a local move may re-emit (all but the forced two)
# a run stops when this many consecutive emissions find no new orbit: a search that has collapsed onto codes it
# already paid for no longer spends budget, and would otherwise loop forever
LS_COLLAPSE = 2000


# ============================================================================================ GFlowQEC-offline
def augment(traj, lam, rng, conj, T):
    """The symmetry augmentation of the offline training stream. Three steps, each preserving the code (up to
    equivalence) and its canonical form: a random reordering of the free slots inside each entry, a conjugation
    of each side by a random group element, then, half the time, the mirror (h^-1, g^-1), which swaps the X and
    Z sectors and so flips the bias. Returns (trajectory, lam)."""
    out = list(traj)
    for blk in BLOCKS:
        ps = [p for p in blk if p not in FORCED]
        vals = [traj[p] for p in ps]
        rng.shuffle(vals)
        for p, v in zip(ps, vals):
            out[p] = v
    gA, gB = rng.integers(len(conj)), rng.integers(len(conj))
    out = [int(conj[gA, v]) for v in out[:5]] + [int(conj[gB, v]) for v in out[5:]]
    if rng.random() < 0.5:
        return mirror(out, T), -lam
    return out, lam


def legal_mask(tokens, n, ident):
    """(B, 10, n) legal actions in gauge order: a forced slot takes the identity only; a free slot bars the
    identity and every value already placed earlier in its own entry (a repeat would cancel over GF(2))."""
    B = tokens.shape[0]
    mask = np.ones((B, 10, n), dtype=bool)
    for pos in range(10):
        if pos in FORCED:
            mask[:, pos, :] = False
            mask[:, pos, ident] = True
            continue
        mask[:, pos, ident] = False
        for p in BLOCKS[BLOCK_OF[pos]]:
            if p < pos:
                mask[np.arange(B), pos, tokens[:, p]] = False
    return torch.from_numpy(mask)


class TupleTransformer(nn.Module):
    """Decoder-only transformer over the ten tokens. The condition (the bias, as cond_vec) rides on the start
    token; output t predicts token t from the tokens before it. log Z(cond) = logz0 + zc(cond): logz0 is the
    closed-form warm start, set by the training script, and zc a learned residual that starts at zero."""

    def __init__(self, n, d=128, layers=3, heads=4):
        super().__init__()
        self.n = n
        self.tok = nn.Embedding(n, d)
        self.start = nn.Parameter(torch.zeros(d))
        self.cond = nn.Linear(2, d)
        self.pos = nn.Embedding(10, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        # enable_nested_tensor=False only silences torch's warning: with norm_first=True the
        # nested-tensor fast path is switched off anyway, so the computation is unchanged.
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, n)
        self.zc = nn.Linear(2, 1)                        # log Z residual per condition
        nn.init.zeros_(self.zc.weight)
        nn.init.zeros_(self.zc.bias)
        self.logz0 = 0.0
        self.register_buffer("causal", nn.Transformer.generate_square_subsequent_mask(10))

    def logits(self, tokens, cond):
        """tokens (B, 10) long, cond (B, 2) -> (B, 10, n): output t predicts token t from tokens < t."""
        x = torch.cat([(self.start + self.cond(cond)).unsqueeze(1), self.tok(tokens[:, :9])], dim=1)
        x = x + self.pos(torch.arange(10, device=tokens.device)).unsqueeze(0)
        return self.head(self.body(x, mask=self.causal))

    def logp(self, tokens, cond, mask):
        """Teacher-forced log-probability of whole trajectories (B,), illegal actions masked out."""
        lg = self.logits(tokens, cond).masked_fill(~mask, -1e9)
        lp = torch.log_softmax(lg, dim=-1)
        return lp.gather(2, tokens.unsqueeze(2)).squeeze(2).sum(1)

    def logZ(self, cond):
        return self.logz0 + self.zc(cond).squeeze(-1)

    @torch.no_grad()
    def sample(self, B, cond, ident):
        """B trajectories at one condition, on-policy; returns (tokens (B, 10) numpy, logp (B,))."""
        tokens = torch.zeros(B, 10, dtype=torch.long)
        logp = torch.zeros(B)
        for pos in range(10):
            mask = legal_mask(tokens.numpy(), self.n, ident)[:, pos, :]
            lg = self.logits(tokens, cond)[:, pos, :].masked_fill(~mask, -1e9)
            probs = torch.softmax(lg, dim=-1)
            a = torch.multinomial(probs, 1).squeeze(1)
            logp += torch.log(probs.gather(1, a.unsqueeze(1)).squeeze(1) + 1e-30)
            tokens[:, pos] = a
        return tokens.numpy(), logp.numpy()


def logp_of_trajs(net, trajs, lam, ident):
    """Teacher-forced log-probability of already-canonical trajectories: exactly the number
    `TupleTransformer.sample` stamps on its own draws (same masks, same order), evaluated on trajectories the
    model did NOT draw. This is the policy used as a ranker of outside candidates.

    Every canonical tuple is legal under the policy's own mask (slots 0 and 5 hold the identity, the other
    values are non-identity and distinct within their entry), so a masked token here would mean a malformed
    candidate, not a rare one: assert instead of silently returning -1e9."""
    tokens = torch.tensor(trajs, dtype=torch.long)
    cond = torch.tensor([cond_vec(lam)] * len(trajs), dtype=torch.float32)
    mask = legal_mask(tokens.numpy(), net.n, ident)
    assert bool(mask.gather(2, tokens.unsqueeze(2)).all()), "a candidate is illegal under the policy's own mask"
    with torch.no_grad():
        return [float(v) for v in net.logp(tokens, cond, mask)]


def soft_bellman_loss(q, q_next, mask, actions, logr):
    """Offline soft Q-learning's loss: the mean squared soft Bellman residual over every (trajectory, slot).

    On a tree with a terminal-only reward and temperature 1, soft Q-learning and trajectory balance share one
    optimum: pi(a | s) = exp(Q(s, a) - V(s)) samples x in proportion to R(x), and V(s0) equals TB's log Z. So
    the two can be trained on the same data with the same network and differ only in this loss.

        Q(s_t, a_t)      is regressed onto  V(s_t+1) = logsumexp over the legal set of Q(s_t+1, .)
        Q(s_T-1, a_T-1)  is regressed onto  log R(x)   (the leaf; no bootstrap at the last slot)

    Every slot is in the mean, forced slots included: a forced slot has one legal action, and its Q carries the
    value chain to the slot before it.

    q       (B, T, A)  Q(s_t, a) at every slot, WITH gradient (the online network)
    q_next  (B, T, A)  the values the targets are read from (a target network), no gradient
    mask    (B, T, A)  bool, the legal set at each slot
    actions (B, T)     long, the action taken at each slot
    logr    (B,)       terminal log R(x)
    """
    # the Q of the action actually taken; masked entries never reach the gradient
    q_taken = q.gather(2, actions.unsqueeze(2)).squeeze(2)                    # (B, T)
    # V(s_t) at temperature 1, over the legal set only
    v = torch.logsumexp(q_next.masked_fill(~mask, float("-inf")), dim=2)     # (B, T)
    # the target of slot t is V(s_t+1); the last slot's target is the leaf's log R
    y = torch.cat([v[:, 1:], logr.unsqueeze(1)], dim=1).detach()
    return ((q_taken - y) ** 2).mean()


# ============================================================================================ GFlowQEC-online
class OnlinePolicy:
    """Trajectory balance over the ten-slot construction, in gauge order (slots 0 and 5 forced to the identity).

    The state is one indicator vector per entry over the |G| group elements, plus a one-hot of the slot being
    filled (4n + 10 inputs), optionally followed by a condition vector (`cond_dim`) and the relational features
    of the partial tuple (`rel`, a codes.RelTables). The entry bags are lossless (an entry is an unordered sum)
    and identical across every ordering of the same partial content. The trajectory IS the ordered tuple, so
    the construction is a tree and P_B = 1 exactly.

    objective "tb" is trajectory balance. objective "reg" is the supervised twin: the same network, sampler,
    rows and replay, with the loss replaced by a per-step regression of the taken action's raw logit (plus
    log Z) onto log R; `temp` divides the logits the sampler sees (never the ones the loss sees) and must stay
    1.0 under "tb"."""

    def __init__(self, n_elems, hidden=256, lr=1e-3, lr_z=1e-1, seed=0, identity=None, conj=None, cond_dim=0,
                 rel=None, objective="tb", temp=1.0):
        # the seed comes first: every network below draws its initial weights from torch's global generator
        torch.manual_seed(seed)
        self.n = n_elems
        self.n_pos = 10
        self.rel = rel
        self.rel_dim = rel.dim if rel is not None else 0
        self.identity = identity
        assert objective in ("tb", "reg")
        assert temp == 1.0 or objective == "reg", "the sampler temperature is a knob of the regression twin"
        self.objective, self.temp = objective, float(temp)
        self.cond_dim = cond_dim
        self._cond = None
        # in canonical coordinates one term of every entry is the identity: the two entries written e + . + .
        # already carry it, the other two have their FIRST slot forced to it, i.e. {0: e, 5: e}
        self.forced = {blk[0]: identity for bi, blk in enumerate(BLOCKS) if bi not in HAS_IDENTITY}
        # the conjugation table for the replay augmentation (codes.conj_table); conjugation fixes e, so it
        # preserves canonical form as well as the code
        self.conj = conj
        self.net = nn.Sequential(
            nn.Linear(self.enc_dim(), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_elems))
        self.logZ = nn.Parameter(torch.zeros(1))
        groups = [{"params": self.net.parameters(), "lr": lr},
                  {"params": [self.logZ], "lr": lr_z}]
        if cond_dim:
            # the log Z residual head of a conditional policy. Only the offline script builds one (its "twin", for
            # log_n_ordered); the head is never trained there but is still built, because its initial weights are
            # drawn from torch's generator and the transformer built after the twin depends on those draws
            self.zc = nn.Sequential(nn.Linear(cond_dim, 16), nn.ReLU(),
                                    nn.Linear(16, 1))
            nn.init.zeros_(self.zc[-1].weight)
            nn.init.zeros_(self.zc[-1].bias)
            groups.append({"params": self.zc.parameters(), "lr": 1e-2})
        self.opt = torch.optim.Adam(groups)
        self.rng = np.random.default_rng(seed)

    def enc_dim(self):
        """Width of the state encoding: one indicator vector per entry over the |G| elements, a one-hot of the
        slot, then the optional condition and relational features."""
        return len(BLOCKS) * self.n + self.n_pos + self.cond_dim + self.rel_dim

    def log_n_ordered(self):
        """log of the number of legal ordered ten-tuples, = -log P of any one under the uniform policy.
        Each entry has one forced or implicit identity; its free slots draw from the n - 1 non-identity
        elements minus those already used in the entry (59, 58 per entry at n = 60)."""
        tot = 0.0
        for blk in BLOCKS:
            free = len(blk) - sum(1 for p in blk if p in self.forced)
            for i in range(free):
                tot += np.log(self.n - 1 - i)
        return float(tot)

    def warm_start_logZ(self, rewards):
        """Set log Z to where trajectory balance wants it under the UNIFORM initial policy, in closed form.

        The untrained policy is uniform, so logp = -log_n_ordered for every trajectory and the TB residual
        vanishes at log Z = mean(log R) + log_n_ordered. Left at 0, the first few hundred gradient steps would
        go into climbing log Z while the policy gets almost no signal."""
        logr = np.log(np.maximum(np.asarray(rewards, float), 1e-9))
        with torch.no_grad():
            if self.objective == "reg":
                # the twin's residual is log Z + logit - log R with raw logits that start near 0, so it vanishes
                # at log Z = mean(log R): no log|X| term, nothing is normalised
                self.logZ.fill_(float(logr.mean()))
            else:
                self.logZ.fill_(float(logr.mean()) + self.log_n_ordered())

    def _encode(self, chosen, pos):
        base = len(BLOCKS) * self.n
        x = np.zeros(self.enc_dim(), dtype=np.float32)
        for i, a in enumerate(chosen):
            x[BLOCK_OF[i] * self.n + a] = 1.0
        x[base + pos] = 1.0
        if self.cond_dim:
            x[base + self.n_pos:base + self.n_pos + self.cond_dim] = self._cond
        if self.rel is not None:
            x[base + self.n_pos + self.cond_dim:] = rel_features(list(chosen), self.rel)
        return torch.from_numpy(x)

    def _mask(self, chosen, pos):
        """Legal actions at `pos` after the prefix `chosen` (legal_mask for one state)."""
        mask = np.zeros(self.n, dtype=bool)
        if pos in self.forced:
            mask[self.forced[pos]] = True
            return mask
        mask[:] = True
        mask[self.identity] = False                  # e is already in every entry
        for p in BLOCKS[BLOCK_OF[pos]]:              # no repeats within the entry
            if p < len(chosen):
                mask[chosen[p]] = False
        return mask

    def _step_logits(self, chosen, pos):
        mask = self._mask(chosen, pos)
        logits = self.net(self._encode(chosen, pos))
        if self.temp != 1.0:
            logits = logits / self.temp              # the regression twin's sampler temperature
        return logits.masked_fill(torch.from_numpy(~mask), -1e9), mask

    def sample(self, epsilon=0.0):
        """One trajectory. Returns (the ten indices, sum of log forward probabilities). With probability
        epsilon a step takes a uniform legal action; logp still records the policy's own probability."""
        chosen, logp = [], torch.zeros(1)
        for pos in range(self.n_pos):
            logits, mask = self._step_logits(chosen, pos)
            probs = torch.softmax(logits, dim=-1)
            if epsilon and self.rng.random() < epsilon:
                a = int(self.rng.choice(np.flatnonzero(mask)))
            else:
                a = int(torch.multinomial(probs.detach(), 1))
            logp = logp + torch.log(probs[a] + 1e-30)
            chosen.append(a)
        return chosen, logp

    def logp_of(self, chosen):
        """Log forward probability of a GIVEN trajectory under the current weights. Trajectory balance is
        off-policy, so this is what lets stored (trajectory, reward) pairs be replayed at no oracle cost."""
        logp = torch.zeros(1)
        for pos in range(self.n_pos):
            logits, _ = self._step_logits(list(chosen[:pos]), pos)
            logp = logp + torch.log(torch.softmax(logits, dim=-1)[chosen[pos]] + 1e-30)
        return logp

    def vals_of(self, chosen):
        """The regression twin's prediction: the taken action's RAW logit at every free slot of a GIVEN
        trajectory (no softmax, no sampler temperature); the two forced slots are skipped, so 8 values."""
        vals = []
        for pos in range(self.n_pos):
            if pos in self.forced:
                continue
            logits = self.net(self._encode(list(chosen[:pos]), pos))
            vals.append(logits[chosen[pos]])
        return torch.stack(vals)

    def fit_of(self, chosen):
        """What the objective fits for one stored trajectory: log P_F under tb, the raw logits under reg."""
        return self.vals_of(chosen) if self.objective == "reg" else self.logp_of(chosen)

    # ---------------------------------------------------------------- the local move
    def _ls_mask(self, cur, pos, masked):
        """Legal actions at `pos` when re-emitting the masked slots of a COMPLETE tuple. Unlike `_mask`, the
        slots after `pos` already hold values that must not be duplicated inside the entry: the bar set is the
        entry's KEPT values plus the values already RE-EMITTED in this pass."""
        mask = np.zeros(self.n, dtype=bool)
        if pos in self.forced:
            mask[self.forced[pos]] = True
            return mask
        mask[:] = True
        mask[self.identity] = False                  # e is already in every canonical entry
        for p in BLOCKS[BLOCK_OF[pos]]:
            if p == pos:
                continue
            if p not in masked or p < pos:           # kept, or already re-emitted
                mask[cur[p]] = False
        return mask

    def _ls_probs(self, cur, pos, masked, epsilon):
        """The proposal at one masked slot, as one vector: (1 - eps) * softmax(masked logits) + eps * uniform
        over the legal set. The network sees the prefix cur[:pos] only."""
        logits = self.net(self._encode(list(cur[:pos]), pos))
        if self.temp != 1.0:
            logits = logits / self.temp              # the regression twin's sampler temperature
        m = self._ls_mask(cur, pos, masked)
        probs = torch.softmax(logits.masked_fill(torch.from_numpy(~m), -1e9), dim=-1).detach()
        if epsilon:
            unif = torch.from_numpy(m.astype(np.float32) / m.sum())
            probs = (1.0 - epsilon) * probs + epsilon * unif
        return probs, m

    def reconstruct(self, chosen, masked, epsilon=0.0):
        """Re-emit the masked slots of `chosen` from the proposal, keeping the rest. Returns (new tuple,
        log q(new | chosen)). Masking random slots gives the single-slot neighbourhood that parallel tempering
        and MAP-Elites move in (a suffix backtrack could only ever touch the B side here)."""
        cur = list(chosen)
        logq = 0.0
        for pos in sorted(masked):
            probs, _ = self._ls_probs(cur, pos, masked, epsilon)
            a = int(torch.multinomial(probs, 1))
            logq += float(np.log(float(probs[a]) + 1e-30))
            cur[pos] = a
        return cur, logq

    # ---------------------------------------------------------------- training
    def _permute(self, chosen):
        """The replay augmentation: a random within-entry reordering of a stored trajectory, then a conjugation
        of the A side and of the B side by independent random elements. Both keep the code (hence the reward)
        and the canonical form, so training on the result is exact trajectory balance on another leaf of equal
        reward."""
        out = list(chosen)
        for blk in BLOCKS:
            ps = [p for p in blk if p not in self.forced]   # forced slots stay put
            vals = [chosen[p] for p in ps]
            self.rng.shuffle(vals)
            for p, v in zip(ps, vals):
                out[p] = v
        if self.conj is not None:
            gA, gB = self.rng.integers(self.n), self.rng.integers(self.n)
            out = [int(self.conj[gA, v]) for v in out[:5]] + \
                  [int(self.conj[gB, v]) for v in out[5:]]
        return out

    def replay_step(self, buffer, batch=8, augment=True):
        """One gradient step on a minibatch of already-scored (ordered tuple, reward) rows of `buffer`."""
        if len(buffer) < 2:
            return None
        n = min(batch, len(buffer))
        idx = self.rng.choice(len(buffer), n, replace=False)
        pairs = [(self.fit_of(self._permute(buffer[i][0]) if augment else buffer[i][0]),
                  buffer[i][1]) for i in idx]
        return self.train_step(pairs)

    def tb_loss(self, logp, reward):
        """(log Z + sum log P_F - log R)^2. P_B = 1 because the trajectory IS the ordered tuple. The log is
        taken in float64 before torch: rewards s^beta at beta 40-64 overflow float32, their logs do not."""
        logr = torch.tensor(np.log(max(float(reward), 1e-9)))
        return (self.logZ + logp - logr) ** 2

    def reg_loss(self, vals, reward):
        """The regression twin: mean over the free slots of (log Z + logit[a_t] - log R)^2. The same label
        trajectory balance fits, the same intercept parameter, no normalisation."""
        logr = torch.tensor(np.log(max(float(reward), 1e-9)))
        return ((self.logZ + vals - logr) ** 2).mean()

    def train_step(self, batch):
        """`batch` is a list of (logp, reward) under tb, of (raw logits, reward) under reg."""
        loss = torch.stack([self.reg_loss(lp, r) if self.objective == "reg" else self.tb_loss(lp, r)
                            for lp, r in batch]).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach())


def derive_beta(values, target_mass=0.8, floor=1e-2):
    """The reward exponent: the smallest beta on the grid that places >= target_mass of the reward s^beta on
    the top decile of `values`."""
    v = np.maximum(np.asarray(values, float), floor)
    v = v / v.max()
    cut = np.quantile(v, 0.9)
    for beta in (1, 2, 4, 8, 16, 27, 40, 64):
        w = v ** beta
        if w[v >= cut].sum() / w.sum() >= target_mass:
            return float(beta)
    return 64.0


def betas_at_masses(values, masses=(0.35, 0.8)):
    """The betas two target masses derive from the same values, printed beside the run's own beta so a reader
    can see whether the mass knob changed anything (a sweep over a knob that maps to one beta compares two
    identical arms)."""
    return {m: derive_beta(values, target_mass=m) for m in masses}


def arm_gfn(oracle, group, rng, seed, calib=200, batch=8, epsilon=0.05, replay=24,
            beta_mass=0.35, hidden=64, save_path=None):
    """The plain online GFlowNet: gauge-order emission, conjugation and reordering replay augmentation, the
    closed-form log Z warm start, no local search. The calibration draws come out of the same budget."""
    vals, calib_exps = [], []
    for _ in range(calib):
        e = draw_gen(rng, group)
        v = oracle.score(e)
        if v is None:
            break
        vals.append(v)
        calib_exps.append(e)
    betas = betas_at_masses(vals)
    beta = derive_beta(vals, target_mass=beta_mass)
    print(f"  derived beta {beta} from {len(vals)} calibration draws; "
          f"betas at target masses {betas}", flush=True)
    n = len(group_for(group)[1])
    ident = identity_index(group)
    gfn = OnlinePolicy(n, seed=seed, hidden=hidden, identity=ident, conj=conj_table(group))
    gfn.warm_start_logZ([max(float(v), 1e-2) ** beta for v in vals])
    print(f"  warm-started logZ {float(gfn.logZ):.3f} "
          f"(log|X| {gfn.log_n_ordered():.3f})", flush=True)
    # calibration draws are free gauge-coordinate training data with known rewards
    buffer = [(canonicalise(group, e), max(float(v), 1e-2) ** beta)
              for e, v in zip(calib_exps, vals)]
    # Non-generating emissions never reach the oracle: they train at the floor reward (the worst calibration
    # draw), a free negative signal only a learner can use
    r_floor = min(max(float(v), 1e-2) ** beta for v in vals)
    losses = []
    n_rej = 0
    while oracle.calls < oracle.budget:
        pairs = []
        attempts = 0
        while len(pairs) < batch and attempts < 50 * batch:
            attempts += 1
            exps, logp = gfn.sample(epsilon=epsilon)
            if not generates_g(group, exps):
                n_rej += 1
                pairs.append((logp, r_floor))
                buffer.append((list(exps), r_floor))
                continue
            v = oracle.score(exps)
            if v is None:
                break
            r = max(float(v), 1e-2) ** beta
            pairs.append((logp, r))
            buffer.append((list(exps), r))
        if not pairs or oracle.calls >= oracle.budget:
            if pairs:
                losses.append(gfn.train_step(pairs))
            break
        losses.append(gfn.train_step(pairs))
        for _ in range(replay):
            L = gfn.replay_step(buffer, batch=batch, augment=True)
            if L is not None:
                losses.append(L)
    print(f"  non-generating emissions trained at floor: {n_rej}", flush=True)
    if losses:
        print(f"  TB loss first/last: {losses[0]:.3f} -> {losses[-1]:.3f}"
              f"   gradient steps {len(losses)}   logZ {float(gfn.logZ):.3f}", flush=True)
    if save_path:
        torch.save({"net": gfn.net.state_dict(), "logZ": float(gfn.logZ),
                    "beta": beta, "losses": losses, "hidden": hidden}, save_path)


def arm_gflowqec_online(oracle, group, rng, seed, calib=200, batch=4, iters=7, k_mask=1,
                        epsilon=0.05, replay=24, beta_mass=0.8, hidden=64,
                        uniform=False, uniform_local=False, save_path=None, paired=False,
                        beta=None, objective="tb", temp=1.0):
    """GFlowQEC-online: a GFlowNet inside a local search (after Kim et al., ICLR 2024). The run files and old
    checkpoints call this arm "lsgfn".

    A round: draw `batch` fresh trajectories from the policy (the global move), refine each one `iters` times
    (the local move: mask `k_mask` random free slots and re-emit them from the policy, keep the child iff its
    score strictly improves), then train on every row, accepted and rejected alike. The depth of a cell is
    `iters` (7, 31, 127).

    Before the search, `calib` uniform generating draws are paid (they fix beta, the floor reward and the log Z
    warm start). `beta` given (n = 180, from the reference sample) replaces the beta derived from them.

    Gradient steps per emission are held at exactly (1 + replay) / 8 = 3.125 at every depth: a round emits a
    variable number of rows, so the replay count is set to hit 3.125 x rows minus the chunk steps already taken,
    carrying the fraction across rounds.

    Options that exist only to regenerate the reference cells and the ablations of the paper:
      uniform        no GFlowNet at all: uniform global draws and uniform local moves, nothing trained;
      uniform_local  the GFlowNet's global draws and training, but the local move re-emits uniformly;
      paired         the child is scored on the incumbent's noise stream, and an accepted child is then
                     re-scored on its own stream (oracle.rescore_own);
      objective "reg" with `temp`  the supervised twin (OnlinePolicy's regression objective).
    """
    assert not (uniform and uniform_local), "pick one ablation, not both"
    assert k_mask >= 1
    assert objective in ("tb", "reg")
    assert objective == "tb" or not (uniform or uniform_local or paired), \
        "the objective swap is the default configuration with the loss swapped, nothing else"
    assert temp == 1.0 or objective == "reg", "the sampler temperature is a knob of the regression twin"

    vals, calib_exps = [], []
    # src = 2: the uniform calibration draws are neither a global (0) nor a local (1) emission
    oracle.src = 2
    for _ in range(calib):
        e = draw_gen(rng, group)
        v = oracle.score(e)
        if v is None:
            break
        vals.append(v)
        calib_exps.append(e)
    beta = derive_beta(vals, target_mass=beta_mass) if beta is None else beta
    print(f"  {'given' if beta is not None else 'derived'} beta {beta} ({len(vals)} calibration draws); "
          f"betas at target masses {betas_at_masses(vals)}", flush=True)
    n = len(group_for(group)[1])
    ident = identity_index(group)
    gfn = OnlinePolicy(n, seed=seed, hidden=hidden, identity=ident, conj=conj_table(group),
                       objective=objective, temp=temp)
    free_pos = LS_FREE_POS
    assert 1 <= k_mask <= len(free_pos)
    gfn.warm_start_logZ([max(float(v), 1e-2) ** beta for v in vals])
    print(f"  warm-started logZ {float(gfn.logZ):.3f} "
          f"(log|X| {gfn.log_n_ordered():.3f})", flush=True)
    buffer = [(canonicalise(group, e), max(float(v), 1e-2) ** beta)
              for e, v in zip(calib_exps, vals)]
    r_floor = min(max(float(v), 1e-2) ** beta for v in vals)
    if objective == "reg":
        print(f"  objective swap: per-step regression of the taken action's logit + logZ onto log R "
              f"(same rows, replay, augmentation, epsilon); sampler temperature {temp}; "
              f"logZ warm start = mean log R", flush=True)
    reward = lambda v: max(float(v), 1e-2) ** beta

    losses, n_rej, n_prop, n_acc, stale, n_emis = [], 0, 0, 0, 0, 0
    n_self, carry = 0, 0.0        # self-moves (prop == cur); fractional step carry
    acc_series, next_mark = [], 5000   # acceptance every 5,000 calls: catches late decay
    while oracle.calls < oracle.budget:
        # (trajectory, reward) rows, NOT (log P_F, reward): the round is trained in chunks, and each chunk's
        # optimizer step invalidates any autograd graph built before it, so log P_F is recomputed under the
        # current weights at training time (TB is off-policy, so this is the same objective)
        rows = []
        for _ in range(batch):
            # ---- the global move: one fresh trajectory from the policy
            oracle.src = 0
            exps = (canonicalise(group, draw_gen(rng, group)) if uniform
                    else gfn.sample(epsilon=epsilon)[0])
            if not generates_g(group, exps):
                n_rej += 1
                stale += 1
                rows.append((list(exps), r_floor))
                buffer.append((list(exps), r_floor))
                continue
            before = oracle.calls
            v = oracle.score(exps)
            if v is None:
                break
            stale = 0 if oracle.calls > before else stale + 1
            rows.append((list(exps), reward(v)))
            buffer.append((list(exps), reward(v)))
            # ---- the local moves: refine that trajectory in place
            oracle.src = 1
            cur, cur_v = list(exps), v
            for _it in range(iters):
                masked = set(int(p) for p in
                             rng.choice(free_pos, size=k_mask, replace=False))
                prop, _ = gfn.reconstruct(cur, masked, epsilon=1.0 if (uniform or uniform_local) else epsilon)
                if not generates_g(group, prop):
                    # outside the arena: R = 0, so the move is rejected and the chain stays put
                    n_rej += 1
                    stale += 1
                    rows.append((list(prop), r_floor))
                    buffer.append((list(prop), r_floor))
                    continue
                before = oracle.calls
                v2 = oracle.score(prop, stream=cur) if paired else oracle.score(prop)
                if v2 is None:
                    break
                stale = 0 if oracle.calls > before else stale + 1
                rows.append((list(prop), reward(v2)))
                buffer.append((list(prop), reward(v2)))
                # the acceptance rate's denominator is the proposals that could move the chain: generating
                # AND different (a self-move re-emits the current value and is never accepted)
                n_prop += 1
                if list(prop) == list(cur):
                    n_self += 1
                acc = v2 > cur_v
                if acc:
                    n_acc += 1
                    if paired:
                        # the new incumbent's reference value must be its OWN-stream value
                        v_own = oracle.rescore_own(prop)
                        if v_own is None:
                            break
                        cur, cur_v = prop, v_own
                    else:
                        cur, cur_v = prop, v2
            if oracle.calls >= oracle.budget:
                break
        if not rows:
            break
        n_emis += len(rows)
        n_chunks = 0
        if not uniform:          # with no GFlowNet there is nothing to train
            for i in range(0, len(rows), 8):
                chunk = [(gfn.fit_of(t), r) for t, r in rows[i:i + 8]]
                losses.append(gfn.train_step(chunk))
                n_chunks += 1
        if oracle.calls >= oracle.budget:
            break
        # exact parity: (1 + replay) / 8 = 3.125 total steps per emission at every depth, carrying the fraction
        want = (1.0 + replay) / 8.0 * len(rows) - n_chunks + carry
        n_rep = int(want)
        carry = want - n_rep
        for _ in range(0 if uniform else max(0, n_rep)):
            L = gfn.replay_step(buffer, batch=8, augment=True)
            if L is not None:
                losses.append(L)
        if oracle.calls >= next_mark:
            acc_series.append((oracle.calls, n_prop - n_self, n_acc))
            next_mark += 5000
        if stale > LS_COLLAPSE:
            print(f"  COLLAPSE: {stale} consecutive emissions without a new orbit "
                  f"at {oracle.calls} calls; stopping", flush=True)
            break
    print(f"  non-generating emissions trained at floor: {n_rej}", flush=True)
    n_move = n_prop - n_self
    print(f"  local proposals {n_prop} (generating), self-moves {n_self} "
          f"({100.0 * n_self / max(n_prop, 1):.1f}%), moving {n_move}, "
          f"accepted {n_acc} ({100.0 * n_acc / max(n_move, 1):.1f}% of moving)", flush=True)
    if acc_series:
        prev = (0, 0, 0)
        marks = []
        for c, m, a in acc_series:
            dm, da = m - prev[1], a - prev[2]
            marks.append(f"{c // 1000}k:{100.0 * da / max(dm, 1):.0f}%")
            prev = (c, m, a)
        print(f"  acceptance per 5k calls (of moving proposals): {' '.join(marks)}", flush=True)
    # emissions exceed oracle calls because cache hits and non-generating tuples cost nothing
    print(f"  emissions {n_emis}, oracle calls {oracle.calls}, cache hits {oracle.hits}; "
          f"gradient steps per emission {len(losses) / max(n_emis, 1):.3f}, "
          f"per call {len(losses) / max(oracle.calls, 1):.3f}", flush=True)
    if losses:
        print(f"  {'REG' if objective == 'reg' else 'TB'} loss first/last: {losses[0]:.3f} -> {losses[-1]:.3f}"
              f"   gradient steps {len(losses)}   logZ {float(gfn.logZ):.3f}", flush=True)
    if save_path:
        torch.save({"net": gfn.net.state_dict(), "logZ": float(gfn.logZ), "beta": beta,
                    "k_mask": k_mask, "iters": iters, "losses": losses,
                    "accept": n_acc / max(n_prop, 1), "n_emissions": n_emis,
                    "n_proposals": n_prop, "n_nongenerating": n_rej,
                    "n_selfmoves": n_self, "n_moving": n_move, "acc_series": acc_series,
                    "batch": batch, "n_emissions_total": n_emis, "n_steps": len(losses),
                    "calls": oracle.calls, "hits": oracle.hits,
                    **({"objective": "reg", "temp": temp} if objective == "reg" else {}),
                    **({"paired": True, "n_rescored": oracle.n_rescored} if paired else {})},
                   save_path)


# ============================================================================================ the n = 36 sampler
class GFN:
    """A trajectory-balance GFlowNet over two-block codes on the l x m grid (Z_l x Z_m).

    A construction is built in 2 * weight steps: first the `weight` monomials of A (one grid cell per step, no
    repeats), then those of B. A small MLP reads the partial state (the A and B cell bitmaps) and gives one
    logit per cell; the finished (A, B) gets reward R = max(merit, floor)^beta from the oracle. At convergence
    the sampler draws each construction with probability proportional to its reward:

        loss = ( log Z + sum_t log P_F(action_t | state_t) - log R(A, B) )^2

    No backward-policy term: every terminal state has exactly (weight!)^2 trajectories (the monomials of each
    polynomial can arrive in any order), so a uniform backward policy adds the same constant to every
    trajectory, and that constant is absorbed into the learned log Z. The reward floor keeps log R finite on
    the 61 % of the space with merit 0. Oracle accounting is the baselines': a distinct code is paid the first
    time it is scored, and once the budget is spent, trajectories ending in unseen codes leave the loss."""

    def __init__(self, l, m, weight, beta=1.0, reward_floor=0.01, seed=0, hidden=64):
        self.l = l
        self.m = m
        self.weight = weight
        self.cells = l * m            # actions = grid cells; state = 2 cell-bitmaps
        self.beta = beta
        self.floor = reward_floor

        torch.manual_seed(seed)
        # Flat-bitmap MLP: sees the partial state (A bitmap | B bitmap) and returns one logit per grid cell
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * self.cells, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, self.cells),
        )

        # log Z: the learned normalizer of the TB loss (log of total reward mass). It gets a much larger
        # learning rate than the network, the standard TB trick: a wrong Z misattributes error to every action
        # and stalls early training.
        self.log_z = torch.nn.Parameter(torch.zeros(()))
        self.optimizer = torch.optim.Adam([
            {"params": self.net.parameters(), "lr": 1e-3},
            {"params": [self.log_z], "lr": 0.1},
        ])

    def _roll_out(self, batch, epsilon):
        """Sample `batch` trajectories from the current policy.

        Returns (a_list, b_list, log_pf): the finished polynomials of each trajectory, and the summed
        log-probability the policy assigned to the actions taken. With probability epsilon a step takes a
        uniformly random valid cell instead (exploration); log_pf still records the policy's own probability of
        that action, which is what the TB loss needs.
        """
        states = torch.zeros(batch, 2 * self.cells)
        log_pf = torch.zeros(batch)

        for step in range(2 * self.weight):
            # First `weight` steps fill A, the rest fill B. A cell already used in the polynomial being filled
            # is invalid (repeats cancel over GF(2)).
            filling_a = step < self.weight
            if filling_a:
                used = states[:, :self.cells]
            else:
                used = states[:, self.cells:]

            # Policy distribution over cells, with used cells masked out.
            logits = self.net(states)
            logits = logits.masked_fill(used.bool(), float("-inf"))
            policy = torch.distributions.Categorical(logits=logits)
            actions = policy.sample()

            # Exploration: some rows take a uniform valid cell instead.
            if epsilon > 0:
                uniform_logits = torch.zeros(batch, self.cells)
                uniform_logits = uniform_logits.masked_fill(used.bool(), float("-inf"))
                uniform_actions = torch.distributions.Categorical(logits=uniform_logits).sample()
                explore = torch.rand(batch) < epsilon
                actions = torch.where(explore, uniform_actions, actions)

            log_pf = log_pf + policy.log_prob(actions)

            # Write the chosen cell into the state. Clone first: the pre-step state was the network's input
            # and autograd still needs it intact.
            if filling_a:
                column = actions
            else:
                column = actions + self.cells
            states = states.clone()
            states[torch.arange(batch), column] = 1.0

        # Decode the bitmaps back into polynomial tuples for the oracle.
        a_list, b_list = [], []
        for t in range(batch):
            a_cells, b_cells = [], []
            for c in range(self.cells):
                if states[t, c] > 0:
                    a_cells.append(divmod(c, self.m))    # cell index -> (i, j)
                if states[t, self.cells + c] > 0:
                    b_cells.append(divmod(c, self.m))
            a_list.append(tuple(sorted(a_cells)))
            b_list.append(tuple(sorted(b_cells)))
        return a_list, b_list, log_pf

    def train_step(self, oracle, batch, epsilon):
        """One TB gradient step on a fresh on-policy batch. Returns the loss, or None if nothing was usable."""
        a_list, b_list, log_pf = self._roll_out(batch, epsilon)

        # Look up rewards. Cached codes are free; new ones cost budget; new ones after the budget is spent
        # cannot be scored, so they leave the batch.
        kept_rows = []
        kept_log_rewards = []
        for t in range(batch):
            scored = oracle.try_score(a_list[t], b_list[t])
            if scored is None:
                continue
            reward = max(scored.merit, self.floor) ** self.beta
            kept_rows.append(t)
            kept_log_rewards.append(math.log(reward))

        if not kept_rows:
            return None

        # Trajectory balance: log Z + log P_F(trajectory) should equal log R.
        prediction = self.log_z + log_pf[kept_rows]
        target = torch.tensor(kept_log_rewards)
        loss = ((prediction - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, oracle, steps, batch=64, epsilon=0.1):
        """The full training loop. Returns the per-step losses (for diagnostics)."""
        losses = []
        for _ in range(steps):
            loss = self.train_step(oracle, batch, epsilon)
            if loss is not None:
                losses.append(loss)
        return losses

    @torch.no_grad()
    def sample(self, count, batch=256):
        """Draw constructions from the trained sampler: no oracle, no exploration. Once trained, generating
        candidates costs network forward passes only."""
        pairs = []
        while len(pairs) < count:
            take = min(batch, count - len(pairs))
            a_list, b_list, _ = self._roll_out(take, epsilon=0.0)
            for t in range(take):
                pairs.append((a_list[t], b_list[t]))
        return pairs
