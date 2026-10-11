"""The rivals of the paper: every search method GFlowQEC is compared against.

Every arm talks to the same oracle object (oracle.py) through the same interface: oracle.score(exps) pays one
call for a new orbit and returns its score, returns the cached score of an orbit already paid (no call), and
returns None once the budget is spent. oracle.calls counts distinct orbits paid. So every arm here runs on the
same oracle, the same reachable space (the generating arena of codes.draw_gen) and the same budget as GFlowQEC.

    arm_random        uniform sampling of the generating arena
    arm_pt            parallel tempering over single-slot moves (tuned PT; `warm` gives the pool-warm variant)
    arm_me            MAP-Elites over the same single-slot move, on a descriptor grid (me_cell_of at n = 180)
    arm_maxent        soft Q-learning (alpha_m = 0) or Munchausen DQN (alpha_m = 0.15): the MaxEnt-RL rival,
                      which targets the same distribution as a GFlowNet (Tiapkin et al. 2024)
    arm_mdqnls        Munchausen DQN inside GFlowQEC-online's local search (n = 180)
    arm_ppo           PPO, the plain reward maximiser
    arm_bo            Bayesian optimisation: a Matern 5/2 GP on one-hot features, expected improvement
    warm_pool_states  the chain starts of pool-warm PT at n = 180, drawn from the stored runs' top codes

The learners (MittenSoftQ, MittenPPO) are subclasses of policy.OnlinePolicy: same ten-slot emission, state
encoding, legal-action mask and augmentation; only the learning rule differs. They stay subclasses on purpose:
OnlinePolicy's constructor seeds torch and builds the trunk before they re-initialise the head, so the order in
which the initial weights are drawn is part of every printed run.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from ldpc import mod2
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm

from gflowqec.codes import (ENTRIES, GROUP_180, HAS_IDENTITY, build, canonical_half, canonicalise, conj_table,
                            draw_gen, generates_g, group_for, identity_index, orbit_key, orbit_rep)
from gflowqec.counts import kind_of, load_dumps
from gflowqec.oracle import bias_score, key_top
from gflowqec.policy import LS_COLLAPSE, LS_FREE_POS, OnlinePolicy, derive_beta

# MAP-Elites descriptor grid at n = 300, fixed before any run: k bins {60, 62, 66, other} x 8 equal-mass bins of
# ler_x - ler_z (interior edges = the 1/8..7/8 quantiles on 26,281 uniform generating codes). Never recomputed.
ME_K_BIN = {60: 0, 62: 1, 66: 2}
ME_ASYM_EDGES = (-0.074, -0.025, -0.01, 0.0, 0.01, 0.024, 0.072)
# MAP-Elites descriptor at n = 180: k bins {36, 38, 40, other} and five bins of d_A - d_B (always even, so no value
# sits on an edge)
ME_K_BIN_N180 = {36: 0, 38: 1, 40: 2}
ME_ASYM_EDGES_N180 = (-3, -1, 1, 3)
# PPO stops when this many consecutive emissions find no new orbit: a maximiser that has concentrated onto codes
# it already paid for can no longer spend the distinct-orbit budget, and the loop would spin forever. The run is
# graded on the distinct orbits it found.
PPO_COLLAPSE = 2000


# ============================================================================================ the single-slot move
def pt_proposal(state, rng, n, ident):
    """Replace one element of the ten-tuple with a legal alternative: distinct within its entry, and the
    identity barred from the two entries written e + . + . (they already contain it). Repeats across entries
    are legal."""
    pos = int(rng.integers(len(state)))
    entry_i = next(i for i, e in enumerate(ENTRIES) if pos in e)
    entry = ENTRIES[entry_i]
    barred = {state[p] for p in entry if p != pos}
    if entry_i in HAS_IDENTITY:
        barred.add(ident)
    pool = [i for i in range(n) if i not in barred and i != state[pos]]
    prop = list(state)
    prop[pos] = pool[int(rng.integers(len(pool)))]
    return prop


def propose_gen(state, rng, n, ident, group):
    """A single-slot move inside the generating arena: redraw the proposal until it generates G.
    The check is free, so rejected proposals cost nothing."""
    while True:
        prop = pt_proposal(state, rng, n, ident)
        if generates_g(group, prop):
            return prop


# ============================================================================================ uniform, PT
def arm_random(oracle, group, rng):
    while oracle.calls < oracle.budget:
        oracle.score(draw_gen(rng, group))


def arm_pt(oracle, group, rng, tmin, tmax, n_chains=8, warm=None):
    """Parallel tempering: n_chains Metropolis chains on a geometric temperature ladder [tmin, tmax], each
    proposing a single-slot move per step, with neighbouring chains offered a swap every 10 steps. The ladder is
    an argument because the score's units differ between arenas; each arena's grid of ladders was tried in good
    faith and the best is reported.

    warm: a list of generating tuples whose first n_chains entries become the initial chain states, in order,
    instead of uniform draws (pool-warm PT); chains past the end of the list draw uniformly as always.
    warm=None is the plain arm.

    PT sets no stream tag, so every row it pays keeps the oracle's src 0."""
    n = len(group_for(group)[1])
    ident = identity_index(group)
    temps = np.geomspace(tmin, tmax, n_chains)
    warm = warm or []
    states, vals = [], []
    for c in range(n_chains):
        st = list(warm[c]) if c < len(warm) else list(draw_gen(rng, group))
        v = oracle.score(st)
        if v is None:
            return
        states.append(st)
        vals.append(v)
    step = 0
    while oracle.calls < oracle.budget:
        for c in range(n_chains):
            prop = propose_gen(states[c], rng, n, ident, group)
            v = oracle.score(prop)
            if v is None:
                return
            if v >= vals[c] or rng.random() < np.exp((v - vals[c]) / temps[c]):
                states[c], vals[c] = prop, v
        step += 1
        if step % 10 == 0:
            for c in range(n_chains - 1):
                d = (1 / temps[c] - 1 / temps[c + 1]) * (vals[c + 1] - vals[c])
                if d > 0 or rng.random() < np.exp(d):
                    states[c], states[c + 1] = states[c + 1], states[c]
                    vals[c], vals[c + 1] = vals[c + 1], vals[c]


def warm_pool_states(lam, sign, rng, n_chains=8):
    """The chain starts of pool-warm PT at n = 180: PT is handed the same certified computation the offline
    policy trained on, so the comparison controls for that free head start.

    The pool is the 42 stored runs (counts.load_dumps, the certifying-oracle runs). A row qualifies when it was
    certified (dX is not None) and its bias score reaches the key's top at this run's lam; a censored exact
    ladder is a lower bound on the distance, so a censored row that reaches the top IS a top. Survivors are
    folded into neighbourhoods by codes.canonical_half of the bottleneck half (rep[5:10] at a Z-biased condition,
    rep[:5] at an X-biased one), each neighbourhood keeping the first representative seen (file-name order),
    and n_chains neighbourhoods are drawn without replacement.

    Returns (states, info) with info the dump's header block."""
    top = key_top(lam)[0]
    half = slice(5, 10) if sign == "Z" else slice(0, 5)
    runs, _ = load_dumps()
    stored = [d for d in runs if kind_of(d) == "[cert]"]
    assert len(stored) == 42, f"expected the 42 stored runs, found {len(stored)}"
    rep_of_class, tops = {}, set()
    for d in stored:
        for i in range(len(d["dX"])):
            if d["dX"][i] is None:          # not certified
                continue
            if bias_score(d["dX"][i], d["dZ"][i], lam) < top:
                continue
            rep = [int(v) for v in d["rep"][i]]
            tops.add(orbit_key(GROUP_180, rep))          # distinct orbits (2,603 at gamma = +1)
            rep_of_class.setdefault(canonical_half(rep[half]), rep)
    classes = sorted(rep_of_class)
    chosen = [classes[int(j)] for j in rng.choice(len(classes), size=n_chains, replace=False)]
    states = [rep_of_class[k] for k in chosen]
    for st in states:
        assert generates_g(GROUP_180, st), "a pool representative does not generate G"
    print(f"  warm pool at lam {lam:+.2f}: {len(tops)} certified top orbits in {len(classes)} "
          f"gauge-canonical classes; {n_chains} chains seeded from classes "
          f"{[list(k) for k in chosen]}", flush=True)
    return states, {"n_tops": len(tops), "n_classes": len(classes), "chosen": [list(k) for k in chosen]}


# ============================================================================================ MAP-Elites
def arm_me(oracle, group, rng, init=200, cell_of=None):
    """MAP-Elites on a fixed descriptor grid.

    `init` uniform draws seed the archive (200, the learners' calibration count). Then each step picks a parent
    uniformly over the occupied cells, mutates it by the single-slot move, pays it, and the child replaces its
    cell's champion on a strict improvement of the score. Selection is rank-based within cells, so any monotone
    reshaping of the score changes nothing.

    cell_of: a callable exps -> cell (me_cell_of at n = 180). None = the n = 300 grid: k bin x bin of
    ler_x - ler_z, read from the oracle's meta row of the paid orbit."""
    n = len(group_for(group)[1])
    ident = identity_index(group)

    if cell_of is None:
        def cell_of(exps):
            m = oracle.meta[orbit_key(group, exps)]
            kbin = ME_K_BIN.get(m["k"], 3)
            return (kbin, int(np.searchsorted(ME_ASYM_EDGES, m["ler_x"] - m["ler_z"])))

    archive = {}                      # cell -> (score, exps)
    for _ in range(init):
        e = draw_gen(rng, group)
        v = oracle.score(e)
        if v is None:
            return
        c = cell_of(e)
        if c not in archive or v > archive[c][0]:
            archive[c] = (v, list(e))
    stale = 0                         # consecutive proposals that were cache hits
    while oracle.calls < oracle.budget:
        if stale > 200:
            # stagnation guard: every champion's single-move neighbourhood can end up fully paid, after which
            # proposals cost nothing and the loop would spin forever; one uniform immigrant re-opens it
            prop, stale = draw_gen(rng, group), 0
        else:
            cells = tuple(archive)
            parent = archive[cells[rng.integers(len(cells))]][1]
            prop = propose_gen(parent, rng, n, ident, group)
        before = oracle.calls
        v = oracle.score(prop)
        if v is None:
            return
        stale = 0 if oracle.calls > before else stale + 1
        c = cell_of(prop)
        if c not in archive or v > archive[c][0]:
            archive[c] = (v, list(prop))


def me_cell_of(oracle, grid):
    """The n = 180 MAP-Elites descriptor, a closure exps -> cell over the bound distances every paid row carries
    (a censored bound enters as its stored value):
        g5   the asymmetry bin alone: searchsorted(ME_ASYM_EDGES_N180, d_A - d_B), 5 cells
        g20  (k bin, asymmetry bin), k bins {36: 0, 38: 1, 40: 2, any other k: 3}, 20 cells
    k = n - rank(H_X) - rank(H_Z) is computed here once per orbit (a few ms, not a paid call)."""
    if grid not in ("g20", "g5"):
        raise ValueError(f"unknown ME grid {grid}")
    k_of = {}                  # orbit key -> k

    def cell_of(exps):
        key = orbit_key(GROUP_180, exps)
        m = oracle.meta[key]
        abin = int(np.searchsorted(ME_ASYM_EDGES_N180, m["dA"] - m["dB"]))
        if grid == "g5":
            return abin
        if key not in k_of:
            code = build(GROUP_180, np.array(m["rep"]))
            Hx = np.asarray(code.matrix_x) % 2
            Hz = np.asarray(code.matrix_z) % 2
            k_of[key] = int(Hx.shape[1] - mod2.rank(Hx) - mod2.rank(Hz))
        return (ME_K_BIN_N180.get(k_of[key], 3), abin)

    return cell_of


# ============================================================================================ MaxEnt RL
class MittenSoftQ(OnlinePolicy):
    """Soft Q-learning / Munchausen DQN over the ten-slot construction.

    Why this rival: GFlowNet training is entropy-regularised RL (Tiapkin et al. 2024). The construction here is
    a tree (the trajectory IS the ordered tuple), so the equivalent MaxEnt-RL problem has a terminal-only reward
    log R(x) and entropy coefficient 1, and its soft-optimal policy samples proportionally to R: the same target
    as the GFlowNet, learned by a bootstrapped soft Bellman update instead of trajectory balance.

    Two settings, following Tiapkin et al.'s baselines: soft DQN (alpha_m = 0) and Munchausen DQN (alpha_m = 0.15,
    clip l0 = -85 in log-policy units, which never binds on legal actions). The temperature tau = 1 / (1 - alpha_m)
    keeps the entropy-equivalent coefficient (1 - alpha_m) * tau at 1 for both. Munchausen target (gamma = 1):
        y_t = r_t + alpha_m * tau * clip(log pibar(a_t | s_t), l0, 0) + tau * logsumexp(Qbar(s_{t+1}, .) / tau)
    with r_t = 0 except at the last slot, where it is log R and there is no bootstrap term.

    The Q head is zero-initialised, so the initial acting policy softmax(Q / tau) is uniform over legal actions,
    the same starting measure as every other arm. The forced slots (0 and 5) keep their transitions: their Q is
    part of the value bootstrap chain."""

    def __init__(self, n_elems, hidden=64, lr=1e-3, seed=0, identity=None, conj=None,
                 alpha_m=0.0, l0=-85.0, target_every=100):
        super().__init__(n_elems, hidden=hidden, lr=lr, seed=seed, identity=identity, conj=conj)
        # zero-init the Q head: Q == 0 makes the initial policy uniform over legal actions
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.alpha_m = alpha_m
        self.tau = 1.0 / (1.0 - alpha_m)     # entropy-equivalent coefficient 1, see above
        self.l0 = l0
        self.target_every = target_every
        self.tnet = copy.deepcopy(self.net)  # the target network, hard-refreshed every target_every steps
        self._steps = 0

    def _q(self, net, chosen, pos):
        """Masked Q-values at a state, under `net` (online or target)."""
        mask = self._mask(chosen, pos)
        q = net(self._encode(chosen, pos))
        return q.masked_fill(torch.from_numpy(~mask), -1e9), mask

    def _step_logits(self, chosen, pos):
        """The acting policy is softmax(Q / tau), so sample() and logp_of() work unchanged."""
        q, mask = self._q(self.net, chosen, pos)
        return q / self.tau, mask

    def sql_step(self, trajs):
        """One gradient step on a list of (ordered ten-tuple, reward R) trajectories, one transition per slot,
        targets from the target network."""
        preds, targets = [], []
        for chosen, R in trajs:
            for t in range(self.n_pos):
                q, _ = self._q(self.net, list(chosen[:t]), t)
                preds.append(q[chosen[t]])
                with torch.no_grad():
                    if t == self.n_pos - 1:
                        # log R taken in float64 first: R = s^beta can reach 1e66
                        y = float(np.log(max(float(R), 1e-300)))
                    else:
                        qn, _ = self._q(self.tnet, list(chosen[:t + 1]), t + 1)
                        y = float(self.tau * torch.logsumexp(qn / self.tau, dim=-1))
                    if self.alpha_m:
                        qb, _ = self._q(self.tnet, list(chosen[:t]), t)
                        logpi = torch.log_softmax(qb / self.tau, dim=-1)[chosen[t]]
                        y += self.alpha_m * self.tau * float(logpi.clamp(self.l0, 0.0))
                targets.append(y)
        loss = torch.stack(
            [(p - y) ** 2 for p, y in zip(preds, targets)]).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self._steps += 1
        if self._steps % self.target_every == 0:
            self.tnet.load_state_dict(self.net.state_dict())
        return float(loss.detach())

    def sql_replay(self, buffer, batch=8):
        """One gradient step on stored trajectories, with the same reordering + conjugation augmentation the
        GFlowNet replay uses (it keeps the code, hence the reward)."""
        if len(buffer) < 2:
            return None
        idx = self.rng.choice(len(buffer), min(batch, len(buffer)), replace=False)
        trajs = [(self._permute(buffer[i][0]), buffer[i][1]) for i in idx]
        return self.sql_step(trajs)


def arm_maxent(oracle, group, rng, seed, alpha_m, calib=200, batch=8, epsilon=0.05,
               replay=24, beta_mass=0.8, hidden=64, save_path=None, beta=None):
    """The MaxEnt-RL arm: soft DQN at alpha_m = 0, Munchausen DQN at alpha_m = 0.15.

    Everything except the learning rule mirrors the plain GFlowNet arm (policy.arm_gfn): 200 uniform calibration
    draws out of the same budget, the reward R = max(s, 0.01)^beta with beta from derive_beta (mass 0.8, the same
    target distribution), the same batch / epsilon / replay counts on the same augmented buffer, and
    non-generating emissions trained at the floor reward. A given beta (n = 180: the key beta) replaces the
    derived one; the calibration draws are still paid."""
    vals, calib_exps = [], []
    for _ in range(calib):
        e = draw_gen(rng, group)
        v = oracle.score(e)
        if v is None:
            break
        vals.append(v)
        calib_exps.append(e)
    beta = derive_beta(vals, target_mass=beta_mass) if beta is None else beta
    print(f"  beta {beta} ({len(vals)} calibration draws) "
          f"(alpha_m {alpha_m}, tau {1.0 / (1.0 - alpha_m):.2f})", flush=True)
    n = len(group_for(group)[1])
    ident = identity_index(group)
    agent = MittenSoftQ(n, hidden=hidden, seed=seed, identity=ident,
                        conj=conj_table(group), alpha_m=alpha_m)
    # the calibration draws are free training trajectories
    buffer = [(canonicalise(group, e), max(float(v), 1e-2) ** beta)
              for e, v in zip(calib_exps, vals)]
    r_floor = min(r for _, r in buffer)
    losses = []
    n_rej = 0
    while oracle.calls < oracle.budget:
        trajs = []
        attempts = 0
        while len(trajs) < batch and attempts < 50 * batch:
            attempts += 1
            exps, _ = agent.sample(epsilon=epsilon)
            if not generates_g(group, exps):
                n_rej += 1
                trajs.append((list(exps), r_floor))
                buffer.append((list(exps), r_floor))
                continue
            v = oracle.score(exps)
            if v is None:
                break
            r = max(float(v), 1e-2) ** beta
            trajs.append((list(exps), r))
            buffer.append((list(exps), r))
        if not trajs or oracle.calls >= oracle.budget:
            if trajs:
                losses.append(agent.sql_step(trajs))
            break
        losses.append(agent.sql_step(trajs))
        for _ in range(replay):
            L = agent.sql_replay(buffer, batch=batch)
            if L is not None:
                losses.append(L)
    print(f"  non-generating emissions trained at floor: {n_rej}", flush=True)
    if losses:
        print(f"  Q loss first/last: {losses[0]:.3f} -> {losses[-1]:.3f}"
              f"   gradient steps {len(losses)}", flush=True)
    if save_path:
        torch.save({"net": agent.net.state_dict(), "alpha_m": alpha_m, "beta": beta,
                    "losses": losses}, save_path)


def arm_mdqnls(oracle, group, rng, seed, alpha_m=0.15, calib=200, batch=4, iters=7, epsilon=0.05, replay=24,
               beta=None, save_path=None):
    """Munchausen DQN inside GFlowQEC-online's local search (n = 180), so the two learners are compared given the
    SAME local move.

    Everything but the learner is policy.arm_gflowqec_online's depth-7 path: 200 uniform calibration draws paid
    in-run (src 2), then rounds of `batch` = 4 fresh draws from the acting policy (src 0), each refined by
    `iters` = 7 local moves (src 1) that mask ONE free slot and re-emit it, kept on a strict improvement;
    non-generating emissions train at the floor reward; the run stops after LS_COLLAPSE stale emissions; exactly
    3.125 gradient steps per emission, the fraction carried, the last round's replay skipped at the budget.

    The learner is arm_maxent's at alpha_m 0.15: MittenSoftQ, one sql_step per chunk of 8 rows, then sql_replay to
    parity. The local move re-emits the masked slot from the acting policy softmax(Q / tau): OnlinePolicy's
    _ls_probs divides the network output by self.temp, so temp is set to tau; sample(), logp_of() and sql_step()
    never read temp (MittenSoftQ overrides _step_logits and trains through _q)."""
    # ---- 200 uniform calibration draws, paid in-run (src 2)
    oracle.src = 2
    vals, calib_exps = [], []
    for _ in range(calib):
        e = draw_gen(rng, group)
        v = oracle.score(e)
        if v is None:
            break
        vals.append(v)
        calib_exps.append(e)
    given = beta is not None
    if not given:
        beta = derive_beta(vals, target_mass=0.8)
    n = len(group_for(group)[1])
    agent = MittenSoftQ(n, hidden=64, seed=seed, identity=identity_index(group), conj=conj_table(group),
                        alpha_m=alpha_m)
    agent.temp = agent.tau              # the local move samples softmax(Q / tau), the policy the arm acts with
    print(f"  {'given' if given else 'derived'} beta {beta} ({len(vals)} calibration draws); "
          f"alpha_m {alpha_m}, tau {agent.tau:.3f}; local search depth {iters}, batch {batch}, epsilon {epsilon}",
          flush=True)
    free_pos = LS_FREE_POS              # a local move may touch every slot but the forced identity slots
    reward = lambda v: max(float(v), 1e-2) ** beta
    # the calibration draws are free training trajectories
    buffer = [(canonicalise(group, e), reward(v)) for e, v in zip(calib_exps, vals)]
    r_floor = min(r for _, r in buffer)

    losses, n_rej, n_prop, n_acc, n_self, stale, n_emis, carry = [], 0, 0, 0, 0, 0, 0, 0.0
    while oracle.calls < oracle.budget:
        rows = []                       # (trajectory, reward R) rows of this round
        for _ in range(batch):
            # ---- the global move: one fresh trajectory from the acting policy
            oracle.src = 0
            exps = agent.sample(epsilon=epsilon)[0]
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
            # ---- the local moves: refine that trajectory in place (one masked slot, strict improvement)
            oracle.src = 1
            cur, cur_v = list(exps), v
            for _it in range(iters):
                masked = set(int(p) for p in rng.choice(free_pos, size=1, replace=False))
                prop, _ = agent.reconstruct(cur, masked, epsilon=epsilon)
                if not generates_g(group, prop):
                    # outside the arena: R = 0, rejected, trained at the floor
                    n_rej += 1
                    stale += 1
                    rows.append((list(prop), r_floor))
                    buffer.append((list(prop), r_floor))
                    continue
                before = oracle.calls
                v2 = oracle.score(prop)
                if v2 is None:
                    break
                stale = 0 if oracle.calls > before else stale + 1
                rows.append((list(prop), reward(v2)))
                buffer.append((list(prop), reward(v2)))
                n_prop += 1
                if list(prop) == list(cur):
                    n_self += 1          # a self-move re-emits the current value: never an acceptance
                if v2 > cur_v:
                    n_acc += 1
                    cur, cur_v = prop, v2
            if oracle.calls >= oracle.budget:
                break
        if not rows:
            break
        n_emis += len(rows)
        # one soft-Q step per chunk of 8 rows of this round
        n_chunks = 0
        for i in range(0, len(rows), 8):
            losses.append(agent.sql_step(rows[i:i + 8]))
            n_chunks += 1
        if oracle.calls >= oracle.budget:
            break
        # exact parity with the other learners: (1 + replay) / 8 = 3.125 steps per emission, the fraction carried
        want = (1.0 + replay) / 8.0 * len(rows) - n_chunks + carry
        n_rep = int(want)
        carry = want - n_rep
        for _ in range(max(0, n_rep)):
            L = agent.sql_replay(buffer, batch=8)
            if L is not None:
                losses.append(L)
        if stale > LS_COLLAPSE:
            print(f"  COLLAPSE: {stale} consecutive emissions without a new orbit at {oracle.calls} calls; "
                  f"stopping", flush=True)
            break
    n_move = n_prop - n_self
    print(f"  non-generating emissions trained at floor: {n_rej}", flush=True)
    print(f"  local proposals {n_prop} (generating), self-moves {n_self}, moving {n_move}, "
          f"accepted {n_acc} ({100.0 * n_acc / max(n_move, 1):.1f}% of moving)", flush=True)
    print(f"  emissions {n_emis}, oracle calls {oracle.calls}, cache hits {oracle.hits}; "
          f"gradient steps per emission {len(losses) / max(n_emis, 1):.3f}", flush=True)
    if losses:
        print(f"  Q loss first/last: {losses[0]:.3f} -> {losses[-1]:.3f}   gradient steps {len(losses)}",
              flush=True)
    if save_path:
        torch.save({"net": agent.net.state_dict(), "alpha_m": alpha_m, "tau": agent.tau, "beta": beta,
                    "losses": losses, "n_acc": n_acc, "n_prop": n_prop,
                    "n_self": n_self, "n_nongenerating": n_rej, "n_emis": n_emis, "n_steps": len(losses),
                    "calls": oracle.calls, "hits": oracle.hits}, save_path)


# ============================================================================================ PPO
class MittenPPO(OnlinePolicy):
    """Clipped-surrogate PPO over the ten-slot construction: the plain reward maximiser. Where GFlowQEC and the
    MaxEnt arms sample proportionally to R, PPO maximises the expected reward, so it is expected to concentrate.

    Same emission, encoding, mask and 64-64 trunk as the other learners, and a zero-initialised policy head (a
    uniform initial policy). Two asymmetries come with the method family: no replay buffer (PPO is on-policy)
    and no epsilon exploration (an epsilon mixture would corrupt the clipped importance ratios; PPO explores
    through its stochastic policy and the entropy bonus).

    Literature-default constants: clip 0.2, GAE lambda 0.95 (gamma = 1 on the depth-10 tree), advantages
    normalised per update, 4 epochs per update, entropy coefficient 0.01, value coefficient 0.5, one Adam(3e-4)
    over both networks. Because advantages are normalised, the update is invariant to beta."""

    def __init__(self, n_elems, hidden=64, lr=3e-4, seed=0, identity=None,
                 clip=0.2, lam=0.95, ent_coef=0.01, vf_coef=0.5):
        super().__init__(n_elems, hidden=hidden, lr=lr, seed=seed, identity=identity)
        # zero-init the policy head: uniform initial policy over legal actions
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        # a separate value net, same trunk shape; zero-init head so V = 0 at init
        self.vnet = nn.Sequential(
            nn.Linear(self.enc_dim(), hidden), nn.ReLU(),      # 4n + 10 inputs
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        torch.nn.init.zeros_(self.vnet[-1].weight)
        torch.nn.init.zeros_(self.vnet[-1].bias)
        self.clip, self.lam = clip, lam
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        # replaces the trajectory-balance optimizer of the base class (log Z is unused here)
        self.opt = torch.optim.Adam(
            list(self.net.parameters()) + list(self.vnet.parameters()), lr=lr)

    def collect(self):
        """One on-policy trajectory. Returns (ten indices, per-step log-probs DETACHED at collection time: the
        'old' probabilities of the ratio)."""
        chosen, old = [], []
        for pos in range(self.n_pos):
            logits, _ = self._step_logits(chosen, pos)
            logp = torch.log_softmax(logits, dim=-1).detach()
            a = int(torch.multinomial(torch.softmax(logits.detach(), -1), 1))
            old.append(float(logp[a]))
            chosen.append(a)
        return chosen, old

    def _gae(self, trajs):
        """Advantages and value targets under the CURRENT value net, no grad. trajs is a list of
        (chosen, old_logps, terminal_reward); gamma = 1, and every reward is zero except the terminal one."""
        adv, ret = [], []
        with torch.no_grad():
            for chosen, _, r in trajs:
                V = [float(self.vnet(self._encode(list(chosen[:t]), t)))
                     for t in range(self.n_pos)] + [0.0]
                A, gae = [0.0] * self.n_pos, 0.0
                for t in reversed(range(self.n_pos)):
                    delta = (r if t == self.n_pos - 1 else 0.0) + V[t + 1] - V[t]
                    gae = delta + self.lam * gae
                    A[t] = gae
                adv.append(A)
                ret.append([a + v for a, v in zip(A, V[:self.n_pos])])
        return adv, ret

    def ppo_update(self, trajs, epochs=4):
        """One PPO update: fixed advantages, `epochs` full-batch gradient steps."""
        adv, ret = self._gae(trajs)
        flat = np.concatenate(adv)
        mu, sd = float(flat.mean()), float(flat.std()) + 1e-8
        losses = []
        for _ in range(epochs):
            pol, val, ent = [], [], []
            for (chosen, old, _), A, R in zip(trajs, adv, ret):
                for t in range(self.n_pos):
                    logits, _ = self._step_logits(list(chosen[:t]), t)
                    logp_all = torch.log_softmax(logits, dim=-1)
                    ratio = torch.exp(logp_all[chosen[t]] - old[t])
                    a = (A[t] - mu) / sd
                    pol.append(-torch.min(
                        ratio * a,
                        torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a))
                    p = torch.softmax(logits, dim=-1)
                    ent.append(-(p * logp_all).sum())
                    v = self.vnet(self._encode(list(chosen[:t]), t)).squeeze(-1)
                    val.append((v - R[t]) ** 2)
            loss = (torch.stack(pol).mean()
                    + self.vf_coef * torch.stack(val).mean()
                    - self.ent_coef * torch.stack(ent).mean())
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses))


def arm_ppo(oracle, group, rng, seed, calib=200, batch=8, epochs=4, beta_mass=0.8,
            save_path=None, ent_coef=0.01, beta=None):
    """PPO, the plain reward-maximising rival.

    The 200 calibration draws only set beta and the reward centering (on-policy PPO cannot train on them).
    Terminal reward beta * log(max(s, 0.01)) - c, with c the mean over the calibration draws (a constant shift
    that keeps value targets small); non-generating emissions train at the floor reward like every learner.
    A given beta (n = 180: the key beta) replaces the derived one; the calibration draws are still paid and
    still set c and the floor."""
    vals = []
    for _ in range(calib):
        v = oracle.score(draw_gen(rng, group))
        if v is None:
            break
        vals.append(v)
    given = beta is not None
    if not given:
        beta = derive_beta(vals, target_mass=beta_mass)
    # log R in float64 throughout; R = s^beta itself is never materialised
    logr = lambda v: beta * float(np.log(max(float(v), 1e-2)))
    c = float(np.mean([logr(v) for v in vals]))
    r_floor = min(logr(v) for v in vals) - c
    print(f"  {'given' if given else 'derived'} beta {beta} from {len(vals)} calibration draws; "
          f"centering c {c:.1f}, floor {r_floor:.1f}", flush=True)
    agent = MittenPPO(len(group_for(group)[1]), hidden=64, seed=seed,
                      identity=identity_index(group), ent_coef=ent_coef)
    losses = []
    n_rej = 0
    stale = 0                 # consecutive emissions that scored no NEW orbit
    while oracle.calls < oracle.budget:
        trajs = []
        attempts = 0
        while len(trajs) < batch and attempts < 50 * batch:
            attempts += 1
            chosen, old = agent.collect()
            before = oracle.calls
            if not generates_g(group, chosen):
                n_rej += 1
                trajs.append((chosen, old, r_floor))
            else:
                v = oracle.score(chosen)
                if v is None:
                    break
                trajs.append((chosen, old, logr(v) - c))
            stale = 0 if oracle.calls > before else stale + 1
        if not trajs or oracle.calls >= oracle.budget:
            if trajs:
                losses.append(agent.ppo_update(trajs, epochs=epochs))
            break
        losses.append(agent.ppo_update(trajs, epochs=epochs))
        if stale > PPO_COLLAPSE:
            print(f"  COLLAPSE: {stale} consecutive emissions without a new orbit "
                  f"at {oracle.calls} calls; stopping", flush=True)
            break
    print(f"  non-generating emissions trained at floor: {n_rej}", flush=True)
    if losses:
        print(f"  PPO loss first/last: {losses[0]:.3f} -> {losses[-1]:.3f}"
              f"   updates {len(losses)}", flush=True)
    if save_path:
        torch.save({"net": agent.net.state_dict(), "vnet": agent.vnet.state_dict(),
                    "beta": beta, "center": c, "losses": losses}, save_path)


# ============================================================================================ Bayesian optimisation
def _matern52(d2, lengthscale):
    """Matern-5/2 kernel from SQUARED distances."""
    r = np.sqrt(np.maximum(d2, 0.0)) / lengthscale
    s = np.sqrt(5.0) * r
    return (1.0 + s + s * s / 3.0) * np.exp(-s)


def _sqdist(a, b):
    return (np.square(a).sum(1)[:, None] + np.square(b).sum(1)[None, :]
            - 2.0 * a @ b.T)


def expected_improvement(mean, sd, best, xi=0.01):
    """EI for MAXIMISING. Non-negative, and exactly zero where sd collapses."""
    sd = np.maximum(sd, 1e-12)
    z = (mean - best - xi) / sd
    return (mean - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


class MaternGP:
    """Matern 5/2 GP; (log length-scale, log signal variance, log noise) by marginal likelihood (L-BFGS-B),
    posterior through one Cholesky factor. Hand-rolled: scikit-learn is not a dependency.

    The one-hot features are deliberately NOT standardised per column: a column constant across the paid set
    would be divided by ~0 and explode on the first candidate that differs from it. y is centred; the signal
    variance carries the scale."""

    JITTER = 1e-8      # keeps the Cholesky well posed when the fitted noise runs to zero

    def fit(self, x, y, theta=None):
        """theta = (log length-scale, log signal variance, log noise) to REUSE instead of optimising; None
        optimises them. Reuse serves the windowed run's refit cadence: at 2,200 observations the hyper-parameter
        search costs about 50 times the posterior itself."""
        self.x = np.asarray(x, dtype=float)
        self.y_mean = float(np.mean(y))
        self.y = np.asarray(y, dtype=float) - self.y_mean
        d2 = _sqdist(self.x, self.x)
        eye = np.eye(len(self.x))

        def nll(theta):
            """Negative log marginal likelihood, constant term dropped."""
            ls, sv, nz = np.exp(theta)
            a = sv * _matern52(d2, ls) + (nz + self.JITTER) * eye
            c = cho_factor(a, lower=True)
            return 0.5 * float(self.y @ cho_solve(c, self.y)) + np.log(np.diag(c[0])).sum()

        # the usual weakly-informative start: the median pairwise distance, the sample variance, and a noise a
        # hundredth of it; bounds are a few decades either side
        if theta is None:
            ls0 = max(float(np.sqrt(np.median(d2))), 1e-3)
            sv0 = max(float(np.var(self.y)), 1e-6)
            theta0 = np.log([ls0, sv0, 0.01 * sv0])
            bounds = [(theta0[0] - 4.0, theta0[0] + 4.0), (theta0[1] - 6.0, theta0[1] + 6.0),
                      (theta0[2] - 6.0, theta0[2] + 6.0)]
            theta = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds).x
        self.theta = np.asarray(theta, dtype=float)
        self.ls, self.sv, self.nz = (float(v) for v in np.exp(self.theta))
        self._chol = cho_factor(self.sv * _matern52(d2, self.ls)
                                + (self.nz + self.JITTER) * eye, lower=True)
        self._alpha = cho_solve(self._chol, self.y)
        return self

    def predict(self, x):
        """Posterior mean and sd of the LATENT function (the noise term is not added)."""
        k = self.sv * _matern52(_sqdist(np.asarray(x, dtype=float), self.x), self.ls)
        var = self.sv - np.einsum("ij,ji->i", k, cho_solve(self._chol, k.T))
        return k @ self._alpha + self.y_mean, np.sqrt(np.maximum(var, 1e-12))


def bo_rows(paid, window, elite):
    """The observations the GP conditions on: the `window` most recent, plus the `elite` best-scoring ones
    whatever their age, returned in call order.

    window 0 = every observation (the exact GP). The exact GP is cubic in the observation count and cannot run
    40,000 calls, so the matched-budget run fixes the conditioning set at W = 2,000 recent plus E = 200 elite.
    The elite half keeps the incumbent region inside the window: EI needs the neighbourhood of the best
    observations, and pure recency would drop it. Ties in the elite sort break by call order (sorted is stable),
    so the selection is a deterministic function of the paid sequence."""
    items = list(paid.values())                 # dict insertion order = call order
    if not window or len(items) <= window:
        return items
    keep = set(range(len(items) - window, len(items)))                  # the W most recent
    by_score = sorted(range(len(items)), key=lambda i: -items[i][1])    # best score first
    keep.update(by_score[:elite])                                       # the E best so far
    return [items[i] for i in sorted(keep)]


def arm_bo(oracle, group, rng, init=20, slate=2000, xi=0.01, window=0, elite=0, refit=1):
    """Bayesian optimisation, the sample-efficiency baseline (arXiv:2601.18562 is the incumbent). This is a
    one-hot Matern GP, NOT a reimplementation of that paper's graph-network surrogate.

    `init` uniform paid calls, drawn exactly as arm_random draws them, then one GP step per remaining call: refit
    the GP on the conditioning set (bo_rows: every paid pair at window 0, the recent + elite subset otherwise) and
    pay the EI argmax over a FRESH slate of `slate` uniform generating candidates not paid yet. Candidates are
    encoded from orbit_rep, the gauge-invariant representative, so two gauge copies of one code share one feature
    vector and the slate's de-duplication is the same equivalence the oracle's cache uses.

    The EI incumbent `best` is the best over EVERY paid observation, not just the window.

    `refit` is how often the GP's hyper-parameters are re-optimised: 1 = every step, K > 1 every K-th step,
    reusing that theta in between; the posterior is refactored on the current window every step either way."""
    n_els = len(group_for(group)[1])
    n_pos = sum(len(entry) for entry in ENTRIES)      # ten positions
    paid = {}       # canonical rep -> (features, s), in call order; len == oracle.calls

    def rep_of(exps):
        return tuple(int(v) for v in orbit_rep(group, exps))

    def feats(rep):
        # one-hot of the canonical exponents: n_els group elements x 10 positions
        x = np.zeros(n_pos * n_els, dtype=np.float32)
        x[np.arange(n_pos) * n_els + np.array(rep)] = 1.0
        return x

    def pay(exps):
        rep = rep_of(exps)
        paid[rep] = (feats(rep), oracle.score(exps))

    while oracle.calls < init:
        pay(draw_gen(rng, group))
    theta, step = None, 0        # theta: the hyper-parameters carried between refits
    while oracle.calls < oracle.budget:
        rows = bo_rows(paid, window, elite)
        if step % refit == 0:
            theta = None         # re-optimise; at refit 1 this is every step
        gp = MaternGP().fit(np.array([f for f, _ in rows]),
                            np.array([s for _, s in rows]), theta=theta)
        theta, step = gp.theta, step + 1
        cand, seen = [], set()
        while len(cand) < slate:
            e = draw_gen(rng, group)
            rep = rep_of(e)
            if rep in paid or rep in seen:
                continue
            seen.add(rep)
            cand.append((e, feats(rep)))
        mean, sd = gp.predict(np.array([f for _, f in cand]))
        best = max(s for _, s in paid.values())
        pay(cand[int(np.argmax(expected_improvement(mean, sd, best, xi)))][0])
