import { useState, useEffect, useRef, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════
//  WORLD CONSTANTS
// ═══════════════════════════════════════════════════════════════
const MW = 20, MH = 16;
const CW = 620, CH = 496;
const SX = CW / MW, SY = CH / MH;
const NUM_P = 100;
const NL    = 8;

const TRUE_LANDMARKS = [
  { x: 3,  y: 3  }, { x: 10, y: 2  }, { x: 17, y: 3  },
  { x: 2,  y: 8  }, { x: 18, y: 8  },
  { x: 3,  y: 13 }, { x: 10, y: 14 }, { x: 17, y: 13 },
];

const INIT_POSE = { x: 5, y: 8, theta: 0 };

const LM_COLORS = [
  '#f87171','#fb923c','#facc15','#4ade80',
  '#22d3ee','#818cf8','#e879f9','#f472b6',
];

// ═══════════════════════════════════════════════════════════════
//  MATH UTILITIES
// ═══════════════════════════════════════════════════════════════
const randn = () => {
  let u, v;
  do { u = Math.random(); } while (!u);
  do { v = Math.random(); } while (!v);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

const wrap = a => {
  while (a >  Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
};

const dist = (x1,y1,x2,y2) => Math.sqrt((x2-x1)**2 + (y2-y1)**2);
const tc   = (x, y) => [x * SX, (MH - y) * SY];

// 2×2 matrix ops stored as [a,b,c,d] ≡ [[a,b],[c,d]]
const m2det = ([a,b,c,d]) => a*d - b*c;
const m2inv = m => { const d = m2det(m); return [ m[3]/d, -m[1]/d, -m[2]/d, m[0]/d ]; };
const m2mul = ([a,b,c,d],[e,f,g,h]) => [a*e+b*g, a*f+b*h, c*e+d*g, c*f+d*h];
const m2mv  = ([a,b,c,d],[x,y])     => [a*x+b*y, c*x+d*y];
const m2add = (a, b) => a.map((v,i) => v + b[i]);
const m2sub = (a, b) => a.map((v,i) => v - b[i]);
const m2T   = ([a,b,c,d]) => [a, c, b, d];

function ellipseParams(sigma) {
  const [sxx, sxy,, syy] = sigma;
  const tr  = sxx + syy;
  const det = sxx * syy - sxy * sxy;
  const d   = Math.sqrt(Math.max(0, (tr / 2) ** 2 - det));
  return {
    l1:  Math.sqrt(Math.max(0.001, tr / 2 + d)),
    l2:  Math.sqrt(Math.max(0,     tr / 2 - d)),
    ang: Math.atan2(2 * sxy, sxx - syy) / 2,
  };
}

// ═══════════════════════════════════════════════════════════════
//  EKF LANDMARK UPDATE  (FastSLAM 1.0, Thrun §10.2)
// ═══════════════════════════════════════════════════════════════
function ekfLandmarkUpdate(mu, sigma, px, py, ptheta, zr, zb, sR, sB) {
  const [lx, ly] = mu;
  const dx = lx - px, dy = ly - py;
  const q  = dx*dx + dy*dy;
  const r  = Math.sqrt(q);
  if (r < 1e-6) return { mu, sigma, logw: -20 };

  const rh = r;
  const bh = wrap(Math.atan2(dy, dx) - ptheta);
  const vr = zr - rh;
  const vb = wrap(zb - bh);

  const H  = [ dx/r, dy/r, -dy/q, dx/q ];
  const Ht = m2T(H);
  const R  = [ sR*sR, 0, 0, sB*sB ];

  const Q    = m2add(m2mul(m2mul(H, sigma), Ht), R);
  const Qdet = m2det(Q);
  if (Math.abs(Qdet) < 1e-12) return { mu, sigma, logw: -20 };

  const Qi  = m2inv(Q);
  const K   = m2mul(m2mul(sigma, Ht), Qi);
  const Kv  = m2mv(K, [vr, vb]);

  const mu_n    = [ lx + Kv[0], ly + Kv[1] ];
  const sigma_n = m2mul(m2sub([1,0,0,1], m2mul(K, H)), sigma);

  const vTQv = vr*(Qi[0]*vr + Qi[1]*vb) + vb*(Qi[2]*vr + Qi[3]*vb);
  const logw = -0.5*vTQv - 0.5*Math.log(2*Math.PI*Math.abs(Qdet));

  return { mu: mu_n, sigma: sigma_n, logw };
}

function initLandmarkSigma(px, py, ptheta, zr, zb, sR, sB) {
  const a  = ptheta + zb;
  const ca = Math.cos(a), sa = Math.sin(a);
  const s11 = sR*sR * ca*ca + sB*sB * zr*zr * sa*sa;
  const s12 = sR*sR * sa*ca - sB*sB * zr*zr * sa*ca;
  const s22 = sR*sR * sa*sa + sB*sB * zr*zr * ca*ca;
  return [s11, s12, s12, s22];
}

// ═══════════════════════════════════════════════════════════════
//  MOTION MODELS  (Thrun Tables 5.3 / 5.6)
// ═══════════════════════════════════════════════════════════════
function velModel(p, v, w, dt, a) {
  const vh = v + randn() * Math.sqrt(a[0]*v*v + a[1]*w*w);
  const wh = w + randn() * Math.sqrt(a[2]*v*v + a[3]*w*w);
  const gh =     randn() * Math.sqrt(a[4]*v*v + a[5]*w*w);
  let { x, y, theta } = p;
  if (Math.abs(wh) < 1e-6) {
    x += vh * dt * Math.cos(theta);
    y += vh * dt * Math.sin(theta);
  } else {
    const r = vh / wh;
    x += r * (Math.sin(theta + wh*dt) - Math.sin(theta));
    y += r * (Math.cos(theta) - Math.cos(theta + wh*dt));
  }
  return { ...p, x, y, theta: wrap(theta + wh*dt + gh*dt) };
}

function odoModel(p, dr1, dt_, dr2, a) {
  const dr1h = dr1 - randn() * Math.sqrt(a[0]*dr1*dr1 + a[1]*dt_*dt_);
  const dth  = dt_ - randn() * Math.sqrt(a[2]*dt_*dt_ + a[3]*(dr1*dr1+dr2*dr2));
  const dr2h = dr2 - randn() * Math.sqrt(a[0]*dr2*dr2 + a[1]*dt_*dt_);
  let { x, y, theta } = p;
  x += dth * Math.cos(theta + dr1h);
  y += dth * Math.sin(theta + dr1h);
  return { ...p, x, y, theta: wrap(theta + dr1h + dr2h) };
}

// ═══════════════════════════════════════════════════════════════
//  WEIGHT NORMALIZATION + LOW-VARIANCE RESAMPLING
// ═══════════════════════════════════════════════════════════════
function normLogWeights(logws) {
  const mx  = Math.max(...logws);
  const ws  = logws.map(lw => Math.exp(lw - mx));
  const sum = ws.reduce((s, w) => s + w, 0);
  return ws.map(w => w / sum);
}

function lowVarResample(ps) {
  const n = ps.length;
  const out = [];
  const r   = Math.random() / n;
  let c = ps[0].w, i = 0;
  for (let m = 0; m < n; m++) {
    const U = r + m / n;
    while (U > c && i < n - 1) c += ps[++i].w;
    out.push({
      ...ps[i],
      w: 1 / n,
      landmarks: ps[i].landmarks.map(lm => ({ ...lm })),
    });
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════
//  AGGREGATE MAP  (law of total variance)
// ═══════════════════════════════════════════════════════════════
function aggregateMap(particles) {
  return Array.from({ length: NL }, (_, j) => {
    const seen = particles.filter(p => p.landmarks[j].seen);
    if (seen.length === 0) {
      return { mu: null, sigma: [1e4,0,0,1e4], withinSigma: [1e4,0,0,1e4], frac: 0 };
    }
    const n  = seen.length;
    const mx = seen.reduce((s, p) => s + p.landmarks[j].mu[0], 0) / n;
    const my = seen.reduce((s, p) => s + p.landmarks[j].mu[1], 0) / n;

    const bxx = seen.reduce((s, p) => s + (p.landmarks[j].mu[0]-mx)**2, 0) / n;
    const byy = seen.reduce((s, p) => s + (p.landmarks[j].mu[1]-my)**2, 0) / n;
    const bxy = seen.reduce((s, p) => s + (p.landmarks[j].mu[0]-mx)*(p.landmarks[j].mu[1]-my), 0) / n;

    const wxx = seen.reduce((s, p) => s + p.landmarks[j].sigma[0], 0) / n;
    const wyy = seen.reduce((s, p) => s + p.landmarks[j].sigma[3], 0) / n;
    const wxy = seen.reduce((s, p) => s + p.landmarks[j].sigma[1], 0) / n;

    return {
      mu:          [mx, my],
      sigma:       [wxx+bxx, wxy+bxy, wxy+bxy, wyy+byy],
      withinSigma: [wxx,     wxy,     wxy,     wyy    ],
      frac:        n / particles.length,
    };
  });
}

// ═══════════════════════════════════════════════════════════════
//  PARTICLE INIT + STATS
// ═══════════════════════════════════════════════════════════════
function mkParticle(x, y, theta) {
  return {
    x, y, theta, w: 1 / NUM_P,
    landmarks: Array.from({ length: NL }, () => ({
      mu: [0, 0], sigma: [1e4, 0, 0, 1e4], seen: false,
    })),
  };
}

function initParticles(mode) {
  if (mode === 'known') {
    return Array.from({ length: NUM_P }, () =>
      mkParticle(
        INIT_POSE.x + randn() * 0.4,
        INIT_POSE.y + randn() * 0.4,
        INIT_POSE.theta + randn() * 0.2,
      )
    );
  }
  return Array.from({ length: NUM_P }, () =>
    mkParticle(Math.random()*MW, Math.random()*MH, (Math.random()-0.5)*2*Math.PI)
  );
}

function particleStats(ps) {
  const n   = ps.length;
  const mx  = ps.reduce((s, p) => s + p.x, 0) / n;
  const my  = ps.reduce((s, p) => s + p.y, 0) / n;
  const sxx = ps.reduce((s, p) => s + (p.x-mx)**2, 0) / n;
  const syy = ps.reduce((s, p) => s + (p.y-my)**2, 0) / n;
  const sxy = ps.reduce((s, p) => s + (p.x-mx)*(p.y-my), 0) / n;
  const Neff = 1 / Math.max(1e-30, ps.reduce((s, p) => s + p.w*p.w, 0));
  return { mx, my, sigma: [sxx,sxy,sxy,syy], Neff };
}

// ═══════════════════════════════════════════════════════════════
//  CANVAS DRAWING — pure function, no hooks
// ═══════════════════════════════════════════════════════════════
function drawScene(canvas, sim, opts) {
  if (!canvas) return;
  const { pose, particles, history, agg } = sim;
  const { showCov, showTrace, showTrueLM, showSRange, showAllP, showBreakdown, sensorRange } = opts;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#060a13';
  ctx.fillRect(0, 0, CW, CH);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.035)';
  ctx.lineWidth = 0.5;
  for (let x = 0; x <= MW; x++) {
    const px = x * SX;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, CH); ctx.stroke();
  }
  for (let y = 0; y <= MH; y++) {
    const py = y * SY;
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(CW, py); ctx.stroke();
  }

  // Sensor range
  if (showSRange) {
    const [rx, ry] = tc(pose.x, pose.y);
    const rpx = sensorRange * SX;
    ctx.fillStyle = 'rgba(34,197,94,0.04)';
    ctx.beginPath(); ctx.arc(rx, ry, rpx, 0, 2*Math.PI); ctx.fill();
    ctx.strokeStyle = 'rgba(34,197,94,0.22)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 5]);
    ctx.beginPath(); ctx.arc(rx, ry, rpx, 0, 2*Math.PI); ctx.stroke();
    ctx.setLineDash([]);
  }

  // True trajectory trace
  if (showTrace && history.length > 1) {
    ctx.strokeStyle = 'rgba(248,113,113,0.18)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    history.forEach(({ x, y }, i) => {
      const [px, py] = tc(x, y);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Sensor rays
  const [rx0, ry0] = tc(pose.x, pose.y);
  TRUE_LANDMARKS.forEach((lm, i) => {
    if (dist(pose.x, pose.y, lm.x, lm.y) > sensorRange) return;
    const [lx, ly] = tc(lm.x, lm.y);
    ctx.strokeStyle = LM_COLORS[i] + '55';
    ctx.lineWidth = 0.9;
    ctx.beginPath(); ctx.moveTo(rx0, ry0); ctx.lineTo(lx, ly); ctx.stroke();
  });

  // Per-particle landmark scatter
  if (showAllP && agg) {
    particles.forEach(p => {
      p.landmarks.forEach((lm, j) => {
        if (!lm.seen) return;
        const [ex, ey] = tc(lm.mu[0], lm.mu[1]);
        ctx.fillStyle = LM_COLORS[j] + '18';
        ctx.beginPath(); ctx.arc(ex, ey, 1.8, 0, 2*Math.PI); ctx.fill();
      });
    });
  }

  // Aggregate landmark map
  if (agg) {
    agg.forEach((lmEst, j) => {
      if (!lmEst.mu) return;
      const [ex, ey] = tc(lmEst.mu[0], lmEst.mu[1]);
      const tr_total = lmEst.sigma[0] + lmEst.sigma[3];
      const baseCol  = LM_COLORS[j];
      const conf     = 1 / (1 + Math.exp((Math.log10(Math.max(1e-6, tr_total)) + 0.5) * 2));
      const alphaHex = Math.round(100 + conf * 155).toString(16).padStart(2, '0');

      const ep    = ellipseParams(lmEst.sigma);
      const SCALE = 2.0;
      const r1    = Math.max(3, Math.min(ep.l1 * SX * SCALE, 80));
      const r2    = Math.max(1, Math.min(ep.l2 * SY * SCALE, 80));

      ctx.save();
      ctx.translate(ex, ey);
      ctx.rotate(-ep.ang);
      ctx.fillStyle   = baseCol + '12';
      ctx.strokeStyle = baseCol + alphaHex;
      ctx.lineWidth   = 1.4;
      ctx.beginPath(); ctx.ellipse(0, 0, r1, r2, 0, 0, 2*Math.PI);
      ctx.fill(); ctx.stroke();
      ctx.restore();

      if (showBreakdown) {
        const epSig = lmEst.sigma.map((v, i) => v - lmEst.withinSigma[i]);
        if (epSig[0] + epSig[3] > 0.01) {
          const epB = ellipseParams(epSig);
          const rb1 = Math.max(2, Math.min(epB.l1 * SX * SCALE, 70));
          const rb2 = Math.max(1, Math.min(epB.l2 * SY * SCALE, 70));
          ctx.save();
          ctx.translate(ex, ey);
          ctx.rotate(-epB.ang);
          ctx.strokeStyle = baseCol + '60';
          ctx.lineWidth   = 0.8;
          ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.ellipse(0, 0, rb1, rb2, 0, 0, 2*Math.PI);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.restore();
        }
      }

      ctx.fillStyle   = baseCol;
      ctx.strokeStyle = '#060a13';
      ctx.lineWidth   = 1.5;
      ctx.beginPath(); ctx.arc(ex, ey, 4, 0, 2*Math.PI);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.28)';
      ctx.font = '9px monospace';
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(`L${j+1}`, ex + 6, ey - 6);
    });
  }

  // True landmarks
  if (showTrueLM) {
    TRUE_LANDMARKS.forEach((lm, j) => {
      const [cx, cy] = tc(lm.x, lm.y);
      const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, 20);
      grd.addColorStop(0, 'rgba(251,191,36,0.19)');
      grd.addColorStop(1, 'rgba(251,191,36,0)');
      ctx.fillStyle = grd;
      ctx.beginPath(); ctx.arc(cx, cy, 20, 0, 2*Math.PI); ctx.fill();

      ctx.fillStyle   = '#fbbf24';
      ctx.strokeStyle = '#78350f';
      ctx.lineWidth   = 0.5;
      ctx.beginPath();
      for (let k = 0; k < 10; k++) {
        const ang = (k * Math.PI) / 5 - Math.PI / 2;
        const r   = k % 2 === 0 ? 7 : 3.5;
        k === 0
          ? ctx.moveTo(cx + r*Math.cos(ang), cy + r*Math.sin(ang))
          : ctx.lineTo(cx + r*Math.cos(ang), cy + r*Math.sin(ang));
      }
      ctx.closePath(); ctx.fill(); ctx.stroke();

      if (agg && agg[j].mu) {
        const [ex, ey] = tc(agg[j].mu[0], agg[j].mu[1]);
        ctx.strokeStyle = 'rgba(251,191,36,0.38)';
        ctx.lineWidth   = 1;
        ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.setLineDash([]);
      }
    });
  }

  // Particle cloud
  const maxW = particles.reduce((m, p) => Math.max(m, p.w), 1e-20);
  particles.forEach(p => {
    const [cx, cy] = tc(p.x, p.y);
    const alpha = Math.min(0.88, 0.04 + (p.w / maxW) * 0.84);
    ctx.fillStyle = `rgba(56,189,248,${alpha.toFixed(3)})`;
    ctx.beginPath(); ctx.arc(cx, cy, 1.8, 0, 2*Math.PI); ctx.fill();
  });

  // Pose covariance ellipse
  if (showCov && particles.length > 2) {
    const st = particleStats(particles);
    const [cmx, cmy] = tc(st.mx, st.my);
    const ep = ellipseParams(st.sigma);
    [2, 4].forEach((scale, si) => {
      ctx.save();
      ctx.translate(cmx, cmy);
      ctx.rotate(-ep.ang);
      ctx.strokeStyle = si === 0 ? 'rgba(56,189,248,0.55)' : 'rgba(56,189,248,0.18)';
      ctx.lineWidth   = si === 0 ? 1.5 : 1;
      if (si === 1) ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.ellipse(0, 0, Math.max(2, ep.l1*SX*scale), Math.max(2, ep.l2*SY*scale), 0, 0, 2*Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    });
    ctx.fillStyle   = 'rgba(56,189,248,0.92)';
    ctx.strokeStyle = '#060a13';
    ctx.lineWidth   = 1.5;
    ctx.beginPath(); ctx.arc(cmx, cmy, 4, 0, 2*Math.PI);
    ctx.fill(); ctx.stroke();
  }

  // True robot arrow
  const [trx, tryY] = tc(pose.x, pose.y);
  ctx.save();
  ctx.translate(trx, tryY);
  ctx.rotate(-pose.theta);
  ctx.shadowColor = 'rgba(248,113,113,0.65)';
  ctx.shadowBlur  = 9;
  ctx.fillStyle   = '#f87171';
  ctx.strokeStyle = 'rgba(255,255,255,0.65)';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(13, 0); ctx.lineTo(-7.5, 7); ctx.lineTo(-4, 0); ctx.lineTo(-7.5, -7);
  ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.restore();
}

// ═══════════════════════════════════════════════════════════════
//  MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════
export default function FastSLAMViz() {
  const canvasRef = useRef(null);

  // All sim state in a single mutable ref — no stale closure risk
  const simRef = useRef({
    pose:      { ...INIT_POSE },
    particles: initParticles('known'),
    history:   [{ ...INIT_POSE }],
    stepN:     0,
    agg:       null,
  });

  const [renderCount, setRenderCount] = useState(0);
  const tick = useCallback(() => setRenderCount(n => n + 1), []);

  const [mModel,        setMModel]        = useState('velocity');
  const [slamMode,      setSlamMode]      = useState(true);
  const [motionNoise,   setMotionNoise]   = useState(1.0);
  const [sigmaR,        setSigmaR]        = useState(0.35);
  const [sigmaB,        setSigmaB]        = useState(0.1);
  const [sensorRange,   setSensorRange]   = useState(9);
  const [initMode,      setInitMode]      = useState('known');
  const [showCov,       setShowCov]       = useState(true);
  const [showTrace,     setShowTrace]     = useState(true);
  const [showTrueLM,    setShowTrueLM]    = useState(false);
  const [showSRange,    setShowSRange]    = useState(true);
  const [showAllP,      setShowAllP]      = useState(false);
  const [showBreakdown, setShowBreakdown] = useState(false);

  // Draw after every render — drawScene reads fresh from simRef each call
  useEffect(() => {
    drawScene(canvasRef.current, simRef.current, {
      showCov, showTrace, showTrueLM, showSRange, showAllP, showBreakdown, sensorRange,
    });
  });

  // Simulation step
  const step = useCallback((v, w) => {
    const motA = [0.04, 0.0004, 0.04, 0.0004, 0.01, 0.0001].map(x => x * motionNoise);
    const odoA = [0.1, 0.01, 0.1, 0.01].map(x => x * motionNoise);
    const { pose } = simRef.current;
    const dt = 0.5;

    let newPose;
    if (Math.abs(w) < 1e-6) {
      newPose = {
        x:     pose.x + v * dt * Math.cos(pose.theta),
        y:     pose.y + v * dt * Math.sin(pose.theta),
        theta: pose.theta,
      };
    } else {
      const r = v / w;
      newPose = {
        x:     pose.x + r * (Math.sin(pose.theta + w*dt) - Math.sin(pose.theta)),
        y:     pose.y + r * (Math.cos(pose.theta) - Math.cos(pose.theta + w*dt)),
        theta: wrap(pose.theta + w * dt),
      };
    }
    newPose.x = Math.max(0.4, Math.min(MW - 0.4, newPose.x));
    newPose.y = Math.max(0.4, Math.min(MH - 0.4, newPose.y));

    const dTrans = dist(pose.x, pose.y, newPose.x, newPose.y);
    let dRot1 = 0, dRot2 = 0;
    if (dTrans > 1e-6) {
      dRot1 = wrap(Math.atan2(newPose.y - pose.y, newPose.x - pose.x) - pose.theta);
      dRot2 = wrap(newPose.theta - pose.theta - dRot1);
    } else {
      dRot2 = wrap(newPose.theta - pose.theta);
    }

    const visible = TRUE_LANDMARKS.map((lm, idx) => {
      if (dist(newPose.x, newPose.y, lm.x, lm.y) > sensorRange) return null;
      const zr = dist(newPose.x, newPose.y, lm.x, lm.y) + randn() * sigmaR;
      const zb = wrap(Math.atan2(lm.y - newPose.y, lm.x - newPose.x) - newPose.theta)
                  + randn() * sigmaB;
      return { idx, zr, zb };
    }).filter(Boolean);

    const logws = [];
    const newParticles = simRef.current.particles.map(p => {
      const pp = mModel === 'velocity'
        ? velModel(p, v, w, dt, motA)
        : odoModel(p, dRot1, dTrans, dRot2, odoA);

      const np = {
        ...pp,
        x: Math.max(0, Math.min(MW, pp.x)),
        y: Math.max(0, Math.min(MH, pp.y)),
        landmarks: p.landmarks.map(lm => ({ ...lm })),
      };

      let logw = 0;
      visible.forEach(({ idx, zr, zb }) => {
        if (!np.landmarks[idx].seen) {
          const lx = np.x + zr * Math.cos(np.theta + zb);
          const ly = np.y + zr * Math.sin(np.theta + zb);
          const s0 = slamMode
            ? initLandmarkSigma(np.x, np.y, np.theta, zr, zb, sigmaR, sigmaB)
            : [1e4, 0, 0, 1e4];
          np.landmarks[idx] = { mu: [lx, ly], sigma: s0, seen: true };
        } else if (slamMode) {
          const { mu: nm, sigma: ns, logw: dlw } = ekfLandmarkUpdate(
            np.landmarks[idx].mu, np.landmarks[idx].sigma,
            np.x, np.y, np.theta, zr, zb, sigmaR, sigmaB,
          );
          np.landmarks[idx] = { mu: nm, sigma: ns, seen: true };
          logw += dlw;
        } else {
          const lm = TRUE_LANDMARKS[idx];
          const er = dist(np.x, np.y, lm.x, lm.y);
          const eb = wrap(Math.atan2(lm.y - np.y, lm.x - np.x) - np.theta);
          logw += -0.5 * ((zr - er) / sigmaR) ** 2
                - 0.5 * (wrap(zb - eb) / sigmaB) ** 2;
        }
      });
      logws.push(logw);
      return np;
    });

    let finalParticles = newParticles;
    if (visible.length > 0) {
      const normW = normLogWeights(logws);
      normW.forEach((w, i) => { finalParticles[i].w = w; });
      const Neff = 1 / normW.reduce((s, w) => s + w * w, 0);
      if (Neff < NUM_P / 2) finalParticles = lowVarResample(finalParticles);
    }

    simRef.current = {
      pose:      newPose,
      particles: finalParticles,
      history:   [...simRef.current.history.slice(-200), newPose],
      stepN:     simRef.current.stepN + 1,
      agg:       aggregateMap(finalParticles),
    };
    tick();
  }, [mModel, slamMode, motionNoise, sigmaR, sigmaB, sensorRange, tick]);

  // Keyboard
  useEffect(() => {
    const h = e => {
      if (e.key === 'ArrowUp'    || e.key === 'w') { e.preventDefault(); step( 2.0,  0   ); }
      if (e.key === 'ArrowDown'  || e.key === 's') { e.preventDefault(); step(-1.5,  0   ); }
      if (e.key === 'ArrowLeft'  || e.key === 'a') { e.preventDefault(); step( 1.0,  1.6 ); }
      if (e.key === 'ArrowRight' || e.key === 'd') { e.preventDefault(); step( 1.0, -1.6 ); }
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [step]);

  // Reset
  const reset = useCallback(() => {
    simRef.current = {
      pose:      { ...INIT_POSE },
      particles: initParticles(initMode),
      history:   [{ ...INIT_POSE }],
      stepN:     0,
      agg:       null,
    };
    tick();
  }, [initMode, tick]);

  // Derived stats for the stats bar
  const { pose, particles, stepN, agg } = simRef.current;
  const st        = particleStats(particles);
  const seenCount = agg ? agg.filter(l => l.mu).length : 0;
  const avgUncert = agg
    ? agg.filter(l => l.mu).reduce((s, l) => s + l.sigma[0] + l.sigma[3], 0) / Math.max(1, seenCount)
    : 0;
  let mapRMSE = null;
  if (agg && showTrueLM) {
    const ds = agg
      .map((l, i) => l.mu ? dist(l.mu[0], l.mu[1], TRUE_LANDMARKS[i].x, TRUE_LANDMARKS[i].y) : null)
      .filter(d => d !== null);
    if (ds.length) mapRMSE = Math.sqrt(ds.reduce((s, d) => s + d*d, 0) / ds.length);
  }

  const fonts = `'JetBrains Mono','Fira Code','Courier New',monospace`;
  const S = {
    root:     { fontFamily:fonts, background:'#04080f', color:'#cbd5e1', padding:16, userSelect:'none', maxWidth:970 },
    panel:    { background:'#0b1120', border:'1px solid rgba(99,102,241,0.14)', borderRadius:8, padding:'10px 14px' },
    head:     { color:'#818cf8', fontWeight:700, fontSize:11, letterSpacing:'0.12em', textTransform:'uppercase', marginBottom:8, display:'block' },
    radio:    { display:'flex', alignItems:'center', gap:7, cursor:'pointer', fontSize:12, padding:'2px 0' },
    label:    { color:'#94a3b8', fontSize:11, marginBottom:2, display:'block' },
    slider:   { width:'100%', marginTop:3, accentColor:'#818cf8', cursor:'pointer' },
    btn:      { background:'#0b1120', color:'#cbd5e1', border:'1px solid rgba(99,102,241,0.25)', borderRadius:6, padding:'7px 12px', cursor:'pointer', fontSize:13, fontFamily:fonts },
    driveBtn: { background:'#0b1120', color:'#e2e8f0', border:'1px solid rgba(99,102,241,0.3)', borderRadius:6, padding:'9px 0', cursor:'pointer', fontSize:16, width:42, fontFamily:fonts },
    stat:     { fontSize:10, color:'#475569', fontFamily:fonts },
    statV:    { fontWeight:700 },
    check:    { display:'flex', alignItems:'center', gap:7, fontSize:12, cursor:'pointer', marginTop:4 },
    modeBtn:  (active) => ({
      flex:1, padding:'7px 0', borderRadius:5, cursor:'pointer', fontSize:11, fontFamily:fonts,
      border:     active ? '1px solid rgba(129,140,248,0.7)' : '1px solid rgba(99,102,241,0.15)',
      background: active ? 'rgba(99,102,241,0.15)' : '#0b1120',
      color:      active ? '#a5b4fc' : '#475569',
      fontWeight: active ? 700 : 400,
    }),
  };

  const statColors = ['#7dd3fc','#4ade80','#a5b4fc','#fb923c','#f87171','#38bdf8','#facc15'];

  return (
    <div style={S.root}>
      <div style={{ marginBottom:12, borderBottom:'1px solid rgba(99,102,241,0.18)', paddingBottom:10 }}>
        <div style={{ color:'#a5b4fc', fontSize:16, fontWeight:700, letterSpacing:'0.05em' }}>
          FastSLAM — SIMULTANEOUS LOCALIZATION &amp; MAPPING
        </div>
        <div style={{ color:'#1e3a5f', fontSize:11, marginTop:2 }}>
          FastSLAM 1.0 · Thrun, Burgard &amp; Fox §10.2 · {NUM_P} particles × {NL} EKFs = {NUM_P*NL} updates/step
        </div>
      </div>

      {/* ── Top row: canvas + immediate controls ── */}
      <div style={{ display:'flex', gap:14, alignItems:'flex-start' }}>

        {/* Canvas */}
        <div>
          <canvas ref={canvasRef} width={CW} height={CH}
            style={{ border:'1px solid rgba(99,102,241,0.2)', borderRadius:6, display:'block' }} />

          {/* Stats bar */}
          <div style={{ display:'flex', gap:16, marginTop:7, flexWrap:'wrap' }}>
            {[
              ['STEP',   stepN,                                         0],
              ['SEEN',   `${seenCount}/${NL}`,                          1],
              ['N_eff',  `${st.Neff.toFixed(0)}/${NUM_P}`,              2],
              ['AVG σ²', avgUncert.toFixed(3),                          3],
              ['TRUE',   `(${pose.x.toFixed(1)},${pose.y.toFixed(1)})`, 4],
              ['EST',    `(${st.mx.toFixed(1)},${st.my.toFixed(1)})`,   5],
              ...(mapRMSE !== null ? [['MAP RMSE', mapRMSE.toFixed(3)+' m', 6]] : []),
            ].map(([k, v, ci]) => (
              <span key={k} style={S.stat}>
                {k}: <span style={{ ...S.statV, color:statColors[ci] }}>{v}</span>
              </span>
            ))}
          </div>

          {/* Landmark confidence bars */}
          {agg && seenCount > 0 && (
            <div style={{ marginTop:8, display:'flex', gap:3, alignItems:'flex-end', height:28 }}>
              {agg.map((l, j) => {
                if (!l.mu) return (
                  <div key={j} style={{ width:14, height:6, background:'#1e293b', borderRadius:2, alignSelf:'flex-end' }}
                    title={`L${j+1}: unseen`} />
                );
                const tr = Math.min(l.sigma[0] + l.sigma[3], 10);
                const h  = Math.max(3, Math.round((1 - tr/10) * 28));
                return (
                  <div key={j} style={{
                    width:14, height:h, background:LM_COLORS[j], borderRadius:'2px 2px 0 0',
                    opacity:0.7, alignSelf:'flex-end', transition:'height 0.3s',
                  }} title={`L${j+1}: σ²=${tr.toFixed(3)}`} />
                );
              })}
              <span style={{ color:'#334155', fontSize:9, marginLeft:4, alignSelf:'flex-end', marginBottom:1 }}>
                landmark confidence ↑
              </span>
            </div>
          )}

          {/* Description */}
          <div style={{
            marginTop:8, background:'#0b1120', borderLeft:'3px solid #6366f1',
            borderRadius:'0 6px 6px 0', padding:'8px 12px',
            color:'#94a3b8', fontSize:11, lineHeight:1.6, maxWidth:CW,
          }}>
            <span style={{ color:'#a5b4fc', fontWeight:700 }}>
              ▸ {slamMode ? 'FASTSLAM 1.0' : 'KNOWN-MAP LOCALIZATION'}:{' '}
            </span>
            {slamMode
              ? `Each of the ${NUM_P} particles carries its own EKF for every landmark (${NL} filters/particle = ${NUM_P*NL} EKFs total). Solid ellipses = total uncertainty (within-particle EKF var + between-particle disagreement). Drive a full loop to observe loop closure.`
              : `Known-map mode: sensor updates use true landmark positions. Switch to FastSLAM to see how simultaneous map estimation adds ${NL} EKFs per particle per step.`}
          </div>

          {/* Legend */}
          <div style={{ marginTop:7, display:'flex', gap:14, fontSize:10, color:'#334155', flexWrap:'wrap' }}>
            <span><span style={{color:'#f87171'}}>▶</span> True pose</span>
            <span><span style={{color:'#38bdf8'}}>●</span> Particles ({NUM_P})</span>
            <span><span style={{color:'#818cf8'}}>◯</span> Landmark est.</span>
            <span><span style={{color:'rgba(34,197,94,0.7)'}}>⊙</span> Sensor range</span>
            {showTrueLM    && <span><span style={{color:'#fbbf24'}}>★</span> True landmark</span>}
            {showBreakdown && <span><span style={{color:'#475569'}}>- -</span> Epistemic</span>}
          </div>
        </div>

        {/* ── Immediate controls: drive pad + mode + reset ── */}
        <div style={{ display:'flex', flexDirection:'column', gap:10, minWidth:175 }}>

          {/* Drive pad — top, closest to canvas */}
          <div style={S.panel}>
            <span style={S.head}>Drive Robot</span>
            <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:5, justifyItems:'center' }}>
              <span />
              <button onClick={() => step( 2.0,  0  )} style={S.driveBtn}>↑</button>
              <span />
              <button onClick={() => step( 1.0,  1.6)} style={S.driveBtn}>↶</button>
              <button onClick={() => step(-1.5,  0  )} style={S.driveBtn}>↓</button>
              <button onClick={() => step( 1.0, -1.6)} style={S.driveBtn}>↷</button>
            </div>
            <div style={{ color:'#1e3a5f', fontSize:10, marginTop:6, textAlign:'center' }}>
              WASD / arrow keys
            </div>
          </div>

          {/* Algorithm mode */}
          <div style={S.panel}>
            <span style={S.head}>Algorithm Mode</span>
            <div style={{ display:'flex', gap:6 }}>
              <button style={S.modeBtn(slamMode)}  onClick={() => setSlamMode(true)}>FastSLAM</button>
              <button style={S.modeBtn(!slamMode)} onClick={() => setSlamMode(false)}>Known Map</button>
            </div>
            <div style={{ color:'#1e3a5f', fontSize:10, marginTop:6 }}>
              {slamMode ? `∀ particle: ${NL} EKF filters` : 'Localization w/ oracle map'}
            </div>
          </div>

          {/* Init + Reset */}
          <div style={S.panel}>
            <span style={S.head}>Initialization</span>
            {[['known','Known start'],['uniform','Global prior']].map(([val,lbl]) => (
              <label key={val} style={{ ...S.radio, color: initMode===val ? '#a5b4fc' : '#64748b' }}>
                <input type="radio" value={val} checked={initMode===val} onChange={() => setInitMode(val)} />
                {lbl}
              </label>
            ))}
            <button onClick={reset} style={{
              ...S.btn, marginTop:10, width:'100%',
              borderColor:'rgba(248,113,113,0.3)', color:'#fca5a5',
            }}>
              ↺  RESET
            </button>
          </div>

        </div>
      </div>

      {/* ── Bottom row: set-and-forget sliders ── */}
      <div style={{ display:'flex', gap:10, marginTop:12, flexWrap:'wrap', alignItems:'flex-start' }}>

        {/* Motion model */}
        <div style={{ ...S.panel, minWidth:200, flex:1 }}>
          <span style={S.head}>Motion Model (Ch.5)</span>
          {[
            ['velocity', 'Velocity  (v, ω)',     'T5.3'],
            ['odometry', 'Odometry (δ₁,δt,δ₂)', 'T5.6'],
          ].map(([val, lbl, ref]) => (
            <label key={val} style={{ ...S.radio, color: mModel===val ? '#a5b4fc' : '#64748b' }}>
              <input type="radio" value={val} checked={mModel===val} onChange={() => setMModel(val)} />
              <span>{lbl} <span style={{color:'#1e3a5f',fontSize:10}}>{ref}</span></span>
            </label>
          ))}
          <div style={{marginTop:10}}>
            <span style={S.label}>Motion noise ×{motionNoise.toFixed(1)}</span>
            <input type="range" min="0.1" max="8" step="0.1" value={motionNoise}
              onChange={e => setMotionNoise(+e.target.value)} style={S.slider} />
          </div>
        </div>

        {/* Sensor */}
        <div style={{ ...S.panel, minWidth:200, flex:1 }}>
          <span style={S.head}>Sensor (range + bearing)</span>
          <div>
            <span style={S.label}>σ_range  {sigmaR.toFixed(2)} m</span>
            <input type="range" min="0.05" max="2" step="0.05" value={sigmaR}
              onChange={e => setSigmaR(+e.target.value)} style={S.slider} />
          </div>
          <div style={{marginTop:6}}>
            <span style={S.label}>σ_bearing  {sigmaB.toFixed(2)} rad</span>
            <input type="range" min="0.01" max="0.5" step="0.01" value={sigmaB}
              onChange={e => setSigmaB(+e.target.value)} style={S.slider} />
          </div>
          <div style={{marginTop:6}}>
            <span style={S.label}>Sensor range  {sensorRange.toFixed(0)} m</span>
            <input type="range" min="2" max="18" step="0.5" value={sensorRange}
              onChange={e => setSensorRange(+e.target.value)} style={S.slider} />
          </div>
        </div>

        {/* Display options */}
        <div style={{ ...S.panel, minWidth:200, flex:1 }}>
          <span style={S.head}>Display</span>
          {[
            [showCov,       setShowCov,       'Robot covariance ellipse'],
            [showTrace,     setShowTrace,      'True pose trace'],
            [showTrueLM,    setShowTrueLM,     'Reveal true landmarks'],
            [showSRange,    setShowSRange,     'Sensor range circle'],
            [showAllP,      setShowAllP,       'All particle maps (scatter)'],
            [showBreakdown, setShowBreakdown,  'Epistemic/aleatoric split'],
          ].map(([val, set, lbl]) => (
            <label key={lbl} style={{ ...S.check, color:'#64748b' }}>
              <input type="checkbox" checked={val} onChange={e => set(e.target.checked)} />
              {lbl}
            </label>
          ))}
        </div>

      </div>
    </div>
  );
}
