# FastSLAM 1.0

**Tool:** FastSLAM Visualizer (https://robust-autonomous-systems-laboratory.github.io/fastslam-viz/)  
**References:** Thrun, Burgard & Fox — *Probabilistic Robotics*, Ch. 3 (EKF), Ch. 4 (Particle Filter), Ch. 10 (SLAM)

---

## Background

All of the localization work you have done so far — the particle filter, AMCL — assumed a known map. SLAM removes that assumption. The map is unknown at the start, and the robot must build it while simultaneously determining its own position within it. This is the central estimation problem in mobile robotics: localization requires a map, and consistent mapping requires knowing the robot's trajectory. Neither is available at the start.

FastSLAM 1.0 (Montemerlo et al., 2002; Thrun §10.2) resolves this by factoring the joint posterior over the robot's trajectory and the map into a form that can be maintained efficiently. The quantity it maintains is:

$$p(x_{1:t},\, m \mid z_{1:t},\, u_{1:t})$$

where the map $m = \{\ell_1, \ldots, \ell_N\}$ is a set of $N$ landmark positions. The joint state space has dimension $3t + 2N$, which grows without bound — a direct particle filter over this full space is intractable.

### Rao-Blackwellization

The factoring that makes FastSLAM tractable is called Rao-Blackwellization. The general principle: if you are estimating two unknowns jointly via Monte Carlo, but one can be computed exactly given the other, there is no reason to sample over both. Sample only over the intractable part; compute the tractable part analytically for each sample.

In SLAM, the trajectory $x_{1:t}$ is intractable — it is non-Gaussian and nonlinear, requiring Monte Carlo. The map, given the trajectory, is tractable: if you know exactly where the robot was at every timestep, each landmark's position can be estimated by an EKF, since the range-bearing observation model produces a Gaussian posterior under linearization.

The conditional independence that enables this: given the full trajectory $x_{1:t}$, the landmarks are mutually independent. Landmark $\ell_j$ was observed from specific robot positions; its estimate depends only on those observations and positions, not on any other landmark. The map posterior therefore factors:

$$p(x_{1:t},\, m \mid z_{1:t},\, u_{1:t}) = p(x_{1:t} \mid z_{1:t},\, u_{1:t}) \prod_{k=1}^{N} p(\ell_k \mid x_{1:t},\, z_{1:t})$$

A particle filter handles the left factor. An independent EKF **per** landmark **per** particle handles each right factor.

### What Each Particle Carries

Each particle represents one trajectory hypothesis. Attached to it is a complete map: $N$ independent EKFs, one per landmark. At each timestep, for each particle:

1. **Sample a new pose** from the motion model.
2. **Update the EKF** for the observed landmark using the range-bearing measurement and linearized observation model.
3. **Compute the particle weight** as the measurement likelihood under the updated landmark distribution.
4. **Resample** in proportion to weights using low-variance resampling.

In the visualizer: 100 particles × 8 landmarks = 800 EKF updates per step.

---
### Epistemic and Aleatoric Uncertainty

The total uncertainty in the aggregate landmark estimate — the size of the displayed ellipse — has two distinct sources, and the visualizer can decompose them.

The **aleatoric** component is the mean within-particle EKF covariance, $\mathbb{E}[\Sigma_k^{[i]}]$. It represents measurement noise: the irreducible uncertainty that remains because the range-bearing sensor is noisy. Even if every particle agreed exactly on the robot's trajectory, this uncertainty would persist, because each individual range and bearing observation contains sensor error. It shrinks as more observations of the landmark are accumulated and fused into each particle's EKF, but converges to a nonzero floor set by the sensor noise parameters $\sigma_\text{range}$ and $\sigma_\text{bearing}$.

The **epistemic** component is the variance of the per-particle landmark means across particles, $\text{Var}[\mu_k^{[i]}]$. It represents trajectory uncertainty: disagreement among particles about where the landmark is, arising from disagreement about where the robot has been. Because each particle placed the landmark from its own trajectory hypothesis, particles with different trajectory estimates will have landmark means in slightly different locations. This component shrinks as the particle cloud converges — as resampling eliminates trajectory hypotheses that are inconsistent with the observations, the surviving particles agree more closely on landmark positions.

The total variance follows directly from the law of total variance:

$$\text{Total} = \underbrace{\mathbb{E}\left[\Sigma_k^{[i]}\right]}_{\text{aleatoric}} + \underbrace{\text{Var}\left[\mu_k^{[i]}\right]}_{\text{epistemic}}$$

The decomposition matters for diagnosis. Aleatoric uncertainty is reduced by improving the sensor or accumulating more observations. Epistemic uncertainty is reduced by improving localization — more particles, better motion model, or loop closure. A system showing high total landmark uncertainty needs this decomposition to know which investment will actually help.

---
### Principal Limitations of FastSLAM 1.0

FastSLAM 1.0 has three well-known limitations that define the directions the research literature took after it.

**The proposal distribution problem.** At each step, FastSLAM 1.0 samples new particle poses from the motion model alone — it draws from p(xt∣xt−1,ut)p(x_t \mid x_{t-1}, u_t) p(xt​∣xt−1​,ut​) without incorporating the current measurement ztz_t zt​. This is the *proposal distribution*. The problem is that the motion model is often a poor guide to where the robot actually is: in high-noise conditions a particle can land far from any likely robot position through motion noise alone, receive a near-zero measurement likelihood, and be discarded — wasting the computational cost of its 800 EKF updates. A better proposal would incorporate the measurement to bias samples toward high-likelihood regions before the weight is computed. FastSLAM 2.0 (Thrun §10.4) does exactly this, using the EKF innovation to construct a measurement-informed proposal distribution. The cost is additional computation per particle; the benefit is dramatically better particle efficiency in high-noise environments.

**Particle depletion under loop closure.** When the robot returns to a previously visited area and re-observes familiar landmarks, the measurement likelihood spikes sharply for particles whose accumulated maps correctly predict those observations. This aggressive resampling can collapse the particle set to a handful of survivors. If the correct trajectory hypothesis has been gradually eliminated by earlier resampling — a common outcome after a long traverse — it may no longer be represented in the particle set when loop closure occurs, and the filter cannot recover. This is the most consequential practical failure mode of particle-based SLAM. Graph-based SLAM approaches address it by decoupling the loop closure detection from the recursive filter: the full trajectory is stored as a pose graph and loop closure constraints are incorporated as a global optimization after the fact, rather than as a recursive weight update that can deplete the particle set.

**Known data association.** The visualizer and the FastSLAM 1.0 derivation both assume that each measurement arrives labeled with the correct landmark index. In a real deployment this is unavailable — the robot must decide which landmark (if any) each measurement corresponds to. If a measurement is incorrectly associated with the wrong landmark, the EKF for that landmark is updated with inconsistent data, the landmark estimate is corrupted, and the particle weight is computed against the wrong distribution. The combinatorics of data association grow rapidly with the number of landmarks, and errors compound over time. FastSLAM addresses this by maintaining separate data association hypotheses per particle, but at significant computational cost.

---

## Visualizer Display Reference

| Element | Description |
|---------|-------------|
| Red arrow | True robot pose |
| Cyan dots | Particle cloud — each dot is one trajectory hypothesis |
| Colored ellipses | Aggregate landmark estimate — center is the mean position across particles, ellipse size reflects total uncertainty |
| Dashed ellipse (within colored ellipse) | Epistemic component of landmark uncertainty — between-particle disagreement only (visible when Epistemic/aleatoric split is enabled) |
| Colored bearing lines | Sensor rays to landmarks currently within range |
| Dashed red trace | True robot trajectory history |
| Green sensor ring | Current sensor range boundary |
| **STEP** | Number of motion steps taken |
| **SEEN** | Number of landmarks observed at least once |
| **N_eff** | Effective number of particles: $N_\text{eff} = 1/\sum_i w_i^2$. Equals $N$ when all weights are equal (maximum diversity); falls toward 1 when one particle dominates (near degeneracy). Computed from normalized weights before resampling. |
| **AVG σ²** | Mean trace of the aggregate landmark covariance across all observed landmarks — a scalar summary of total map uncertainty |

---

## Exercise 1 — The Particle Structure

Open the visualizer in **FastSLAM** mode. Enable **All particle maps (scatter)**. Do not drive yet.

**What you should see.** The particle cloud is tightly clustered at the start (Known Start initialization). No landmark ellipses are visible yet — the EKF estimates are not displayed until a landmark has been observed at least once. If any landmarks fall within the sensor range circle from the starting position, bearing lines will be drawn from the robot to those landmarks; otherwise the canvas shows only the particle cloud and the sensor range ring.

Take three steps forward. Watch the landmark ellipses and the per-particle scatter.

**What you should see.** As the first landmark enters the sensor range, its ellipse shrinks rapidly on first contact and more slowly thereafter. The per-particle ellipses for that landmark are slightly offset from each other — each particle placed the landmark from its own slightly different post-motion pose. This offset is the visual signature of between-particle trajectory disagreement. It persists as long as the particle cloud has not converged.

---

**Question 1.** Why does each particle carry its own map rather than all particles sharing a single map?

---

## Exercise 2 — The Update Cycle

Drive toward the nearest landmark cluster. Take 10–12 steps, pausing between each to observe the particle cloud and the landmark ellipses.

**What you should see.** Each motion step spreads the particle cloud slightly. Each measurement update reconcentrates it — particles whose pose hypotheses were inconsistent with the observed landmark are downweighted and eventually eliminated by resampling. The landmark ellipses shrink with each observation but converge to a nonzero asymptotic size. The N_eff stat in the status bar drops after motion steps and recovers after measurement updates.

---

**Question 2.** Why does N_eff drop after a motion step and recover after a measurement update? What does a low N_eff indicate about the particle set?

---

## Exercise 3 — SLAM vs. Known-Map Localization

Reset (Known Start). Drive a square loop: 6 steps forward, left turn, 6 steps forward, left turn, 6 steps forward, left turn, 6 steps forward. Note the final N_eff and the tightness of the particle cloud.

Reset, switch to **Known Map** mode, drive the same loop. Note the same quantities.

**What you should see.** Known-Map converges faster and reaches a tighter particle cloud at the end of the loop. N_eff is higher in Known-Map mode. In FastSLAM mode the particle cloud maintains higher diversity throughout — the map uncertainty adds noise to the weight computation that keeps the weights more uniform, slowing the rate at which the filter rules out trajectory hypotheses.

Enable **Reveal true landmarks** after the FastSLAM run. The MAP RMSE stat shows the error between the aggregate landmark estimates and ground truth. Enable the same in Known-Map mode — RMSE is zero by construction, since the true positions were used throughout.

---

**Question 3.** In your square loop run, why did the FastSLAM particle cloud converge more slowly than the Known-Map run? 

---

## Exercise 4 — Epistemic and Aleatoric Uncertainty

Enable **Epistemic/aleatoric split** in the Display panel. Reset and drive toward a landmark cluster in FastSLAM mode.

**What you should see.** When a landmark is first observed, both the solid (total) and dashed (epistemic) ellipses are large. Over subsequent observations the dashed ellipse shrinks first — particles are converging to consistent trajectory hypotheses and agreeing more closely on the landmark position. The solid ellipse continues shrinking more slowly as the within-particle EKF accumulates observations, approaching its asymptotic floor.

Now increase **σ_range** substantially and take several more steps.

**What you should see.** The solid ellipses grow. The dashed (epistemic) ellipses are largely unchanged. The increase in total uncertainty is carried almost entirely by the aleatoric (within-particle) component — higher sensor noise raises each EKF's posterior covariance, but between-particle disagreement is governed by trajectory convergence, which depends on weight discrimination, not directly on sensor noise.

---

**Question 4.** After increasing σ_range, the landmark ellipses grow but the dashed epistemic ellipses are unchanged. Explain why the two components respond differently to a change in sensor noise.

---

## Exercise 5 — The Proposal Distribution Problem

In the visualizer, increase **Motion noise** to 5.0× or higher. Reset and drive toward a nearby landmark cluster.

**What you should see.** The particle cloud spreads much more aggressively after each motion step. Despite this, measurement updates still pull it back — but N_eff drops more sharply after each update, and recovery is slower. With very high motion noise, it is possible to observe particle deprivation: the particle cloud collapses to a small number of survivors that may not include the correct trajectory hypothesis.

---

**Question 5.** What is the proposal distribution in FastSLAM 1.0, why is it a poor choice in high-noise environments, and what does FastSLAM 2.0 do differently?

---

