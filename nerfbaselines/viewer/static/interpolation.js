import * as THREE from 'three';


function onehot(index, length) {
  return Array.from({ length }, (_, i) => (i === index ? 1 : 0));
}


function zip(...arrays) {
  const length = Math.min(...arrays.map(arr => arr.length));
  return Array.from({ length }, (_, i) => arrays.map(arr => arr[i]));
}


class LinearInterpolation {
  constructor({ vertices, loop = false }) {
    this.vertices = vertices
    this.loop = loop;
  }

  evaluate(t) {
    const n = this.vertices.length;
    if (n < 2) return this.vertices[0];

    let segment, localt;
    if (!this.loop) {
      segment = Math.floor(t * (n - 1));
      localt = (t * (n - 1)) % 1;
    } else {
      segment = Math.floor(t * n);
      localt = (t * n) % 1;
    }
    const clamp_seg = (x) => Math.min(Math.max(x, 0), n - 1);
    const p0 = this.vertices[this.loop ? segment % n : clamp_seg(segment)];
    const p1 = this.vertices[this.loop ? (segment + 1) % n : clamp_seg(segment + 1)];

    function computeSingle(x0, x1) {
      return x0 + (x1 - x0) * localt;
    }
    if (p0 instanceof THREE.Vector3 || p0 instanceof THREE.Vector2) {
      return p0.clone().lerp(p1, localt);
    } else if (p0 instanceof THREE.Quaternion) {
      return p0.clone().slerp(p1, localt);
    } else if (Array.isArray(p0)) {
      return zip(p0, p1).map(computeSingle);
    } else {
      return computeSingle(p0, p1);
    }
  }
}

class KochanekBartelsInterpolation {
  constructor({ vertices, tension = 0, continuity = 0, bias = 0, loop = false }) {
    this.vertices = vertices;
    this.tension = tension;
    this.continuity = continuity;
    this.bias = bias;
    this.loop = loop;
  }

  evaluate(t) {
    t = Math.min(Math.max(t, 0), 1);
    const n = this.vertices.length;
    if (n < 2) return this.vertices[0];

    let segment, localt;
    if (!this.loop) {
      segment = Math.floor(t * (n - 1));
      localt = (t * (n - 1)) % 1;
    } else {
      segment = Math.floor(t * n);
      localt = (t * n) % 1;
    }

    const clamp_seg = (x) => Math.min(Math.max(x, 0), n - 1);
    const p0 = this.vertices[this.loop ? (segment - 1 + n) % n : clamp_seg(segment - 1)];
    const p1 = this.vertices[this.loop ? segment % n : clamp_seg(segment)];
    const p2 = this.vertices[this.loop ? (segment + 1) % n : clamp_seg(segment + 1)];
    const p3 = this.vertices[this.loop ? (segment + 2) % n : clamp_seg(segment + 2)];

    // Hermite basis
    const t2 = localt * localt;
    const t3 = t2 * localt;

    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + localt;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;

    function computeSingle(x0, x1, x2, x3) {
      const m0 = (1 - this.tension) / 2 *
          ((x1 - x0) * (1 + this.bias) * (1 - this.continuity) +
           (x2 - x1) * (1 - this.bias) * (1 + this.continuity));
      const m1 = (1 - this.tension) / 2 *
          ((x2 - x1) * (1 + this.bias) * (1 + this.continuity) +
           (x3 - x2) * (1 - this.bias) * (1 - this.continuity));

      return h00 * x1 + h01 * x2 +
             h10 * m0 + h11 * m1;
    }
    computeSingle = computeSingle.bind(this);

    if (p0 instanceof THREE.Vector3 || p0 instanceof THREE.Vector2) {
      const term1m0 = p1.clone().sub(p0).multiplyScalar((1 + this.bias) * (1 - this.continuity));
      const term2m0 = p2.clone().sub(p1).multiplyScalar((1 - this.bias) * (1 + this.continuity));
      const m0 = term1m0.add(term2m0).multiplyScalar((1 - this.tension) / 2);

      // Compute m1
      const term1m1 = p2.clone().sub(p1).multiplyScalar((1 + this.bias) * (1 + this.continuity));
      const term2m1 = p3.clone().sub(p2).multiplyScalar((1 - this.bias) * (1 - this.continuity));
      const m1 = term1m1.add(term2m1).multiplyScalar((1 - this.tension) / 2);
      
      return p1.clone().multiplyScalar(h00)
           .add(m0.multiplyScalar(h10))
           .add(p2.clone().multiplyScalar(h01))
           .add(m1.multiplyScalar(h11));
    } else if (p0 instanceof THREE.Quaternion) {
      const m0 = new THREE.Quaternion().slerpQuaternions(p1.clone().conjugate(), p2.clone(), (1 - this.tension) / 2);
      const m1 = new THREE.Quaternion().slerpQuaternions(p2.clone().conjugate(), p3.clone(), (1 - this.tension) / 2);
      return new THREE.Quaternion()
          .slerp(p1.clone(), h00)
          .slerp(m0.clone(), h10)
          .slerp(p2.clone(), h01)
          .slerp(m1.clone(), h11);
    } else if (Array.isArray(p0)) {
      return zip(p0, p1, p2, p3).map(computeSingle);
    } else {
      return computeSingle(p0, p1, p2, p3);
    }
  }
}


function pchip_interpolate(x, xValues, yValues) {
  const n = xValues.length;
  if (n < 2) throw new Error("At least two points are required for interpolation.");

  // Step 1: Compute slopes and differences
  const h = Array(n - 1).fill(0).map((_, i) => xValues[i + 1] - xValues[i]);
  const slopes = Array(n - 1).fill(0).map((_, i) => (yValues[i + 1] - yValues[i]) / h[i]);

  // Step 2: Compute derivatives
  const derivatives = Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    if (slopes[i - 1] * slopes[i] > 0) {
      derivatives[i] =
        (2 * slopes[i - 1] * slopes[i]) /
        (slopes[i - 1] + slopes[i]);
    }
  }
  derivatives[0] = slopes[0]; // Endpoint derivative
  derivatives[n - 1] = slopes[n - 2]; // Endpoint derivative

  // Step 3: Interpolation
  return x.map(value => {
    let segment = xValues.length - 2;
    for (let i = 0; i < xValues.length - 1; i++) {
      if (value >= xValues[i] && value <= xValues[i + 1]) {
        segment = i;
        break;
      }
    }

    const t = (value - xValues[segment]) / h[segment];
    const h00 = (1 + 2 * t) * (1 - t) ** 2;
    const h10 = t * (1 - t) ** 2;
    const h01 = t ** 2 * (3 - 2 * t);
    const h11 = t ** 2 * (t - 1);

    return (
      h00 * yValues[segment] +
      h10 * h[segment] * derivatives[segment] +
      h01 * yValues[segment + 1] +
      h11 * h[segment] * derivatives[segment + 1]
    );
  });
}


export function compute_camera_path(props) {
  const { 
    keyframes, 
    loop = false, 
    default_fov = 75, 
    default_transition_duration = 1, 
    framerate = 30, 
    interpolation = 'none', 
    ...rest_props 
  } = props;
  if (interpolation === 'none') {
    return {
      positions: keyframes.map(k => k.position),
      quaternions: keyframes.map(k => k.quaternion),
      fovs: keyframes.map(k => k.fov || default_fov),
      weights: keyframes.map((_, i) => onehot(i, keyframes.length))
    };
  }

  const times = keyframes.map(k => k.transition_duration || default_transition_duration);
  const transition_times_cumsum = times.reduce((acc, val, i) => {
    acc.push((acc[i - 1] || 0) + val);
    return acc;
  }, []);
  const total_duration = transition_times_cumsum[transition_times_cumsum.length - 1];
  const num_frames = Math.floor(total_duration * framerate);

  const Interpolation = {
    'linear': LinearInterpolation,
    'kochanek-bartels': KochanekBartelsInterpolation,
  }[interpolation];
  const position_spline = new Interpolation({ vertices: keyframes.map(k => k.position), loop: loop, ...rest_props});
  const quaternion_spline = new Interpolation({ vertices: keyframes.map(k => k.quaternion), loop: loop, ...rest_props});
  const fov_spline = new Interpolation({ vertices: keyframes.map(k => k.fov || default_fov), loop: loop, ...rest_props });

  const gtime = Array.from({ length: num_frames }, (_, i) => (i / (num_frames-1)));
  const positions = gtime.map(t => position_spline.evaluate(t));
  const quaternions = gtime.map(t => quaternion_spline.evaluate(t));
  const fovs = gtime.map(t => fov_spline.evaluate(t));
  return { positions, quaternions, fovs };
}
