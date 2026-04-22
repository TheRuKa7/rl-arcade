/**
 * Vanilla-JS port of OpenAI Gym's CartPole-v1 dynamics. Matches the Python
 * reference (Barto et al. 1983 / Sutton & Barto reproduction) closely enough
 * that a policy trained in Gymnasium transfers without shift.
 */

const GRAVITY = 9.8;
const MASS_CART = 1.0;
const MASS_POLE = 0.1;
const TOTAL_MASS = MASS_POLE + MASS_CART;
const LENGTH = 0.5; // half pole length
const POLE_MASS_LENGTH = MASS_POLE * LENGTH;
const FORCE_MAG = 10.0;
const TAU = 0.02; // seconds between state updates

export const THETA_THRESHOLD_RADIANS = (12 * 2 * Math.PI) / 360;
export const X_THRESHOLD = 2.4;

export interface CartPoleState {
  x: number;
  xDot: number;
  theta: number;
  thetaDot: number;
}

export function resetCartPole(): CartPoleState {
  const r = () => (Math.random() - 0.5) * 0.1; // uniform(-0.05, 0.05)
  return { x: r(), xDot: r(), theta: r(), thetaDot: r() };
}

export function stepCartPole(
  s: CartPoleState,
  action: 0 | 1,
): { state: CartPoleState; done: boolean; reward: number } {
  const force = action === 1 ? FORCE_MAG : -FORCE_MAG;
  const cosTheta = Math.cos(s.theta);
  const sinTheta = Math.sin(s.theta);
  const temp =
    (force + POLE_MASS_LENGTH * s.thetaDot * s.thetaDot * sinTheta) / TOTAL_MASS;
  const thetaAcc =
    (GRAVITY * sinTheta - cosTheta * temp) /
    (LENGTH * (4.0 / 3.0 - (MASS_POLE * cosTheta * cosTheta) / TOTAL_MASS));
  const xAcc = temp - (POLE_MASS_LENGTH * thetaAcc * cosTheta) / TOTAL_MASS;

  const next: CartPoleState = {
    x: s.x + TAU * s.xDot,
    xDot: s.xDot + TAU * xAcc,
    theta: s.theta + TAU * s.thetaDot,
    thetaDot: s.thetaDot + TAU * thetaAcc,
  };

  const done =
    next.x < -X_THRESHOLD ||
    next.x > X_THRESHOLD ||
    next.theta < -THETA_THRESHOLD_RADIANS ||
    next.theta > THETA_THRESHOLD_RADIANS;

  return { state: next, done, reward: 1.0 };
}

export function toObs(s: CartPoleState): Float32Array {
  return new Float32Array([s.x, s.xDot, s.theta, s.thetaDot]);
}
