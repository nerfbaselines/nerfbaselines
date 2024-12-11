import * as THREE from 'three';


function onehot(index, length) {
  return Array.from({ length }, (_, i) => (i === index ? 1 : 0));
}


function zip(...arrays) {
  const length = Math.min(...arrays.map(arr => arr.length));
  return Array.from({ length }, (_, i) => arrays.map(arr => arr[i]));
}


function logQ(q) {
  // Ensure the quaternion is normalized
  const qNorm = q.clone().normalize();

  // Angle (theta) is the arccosine of the w component
  const theta = Math.acos(qNorm.w);

  // If the angle is close to zero, return a zero vector
  if (Math.abs(theta) < 1e-6) {
    return new THREE.Vector3(0, 0, 0);
  }

  // Calculate the logarithmic map (axis-angle to vector)
  const sinTheta = Math.sin(theta);
  return new THREE.Vector3(qNorm.x, qNorm.y, qNorm.z).multiplyScalar(theta / sinTheta);
}

function expQ(v) {
  // Magnitude of the vector
  const theta = v.length();

  // If the vector is close to zero, return the identity quaternion
  if (Math.abs(theta) < 1e-6) {
    return new THREE.Quaternion(0, 0, 0, 1);
  }

  // Calculate the exponential map (vector to axis-angle)
  const sinTheta = Math.sin(theta);
  const axis = v.clone().normalize();
  return new THREE.Quaternion(
    axis.x * sinTheta,
    axis.y * sinTheta,
    axis.z * sinTheta,
    Math.cos(theta)
  );
}

function powQ(q, alpha) {
  if (q.scalar >= 1) {
    return q.clone();
  }
  return expQ(logQ(q).multiplyScalar(alpha));
}


function deCasteljauQuaternions(controlPoints, t) {
  if (controlPoints.length === 0) {
    throw new Error("No control points provided for interpolation.");
  }

  if (controlPoints.length === 1) {
    return controlPoints[0]; // Only one point, return it
  }

  let points = controlPoints.map(q => q.clone()); // Clone to avoid modifying input
  const n = points.length;

  // Apply De Casteljau iteration
  for (let level = 1; level < n; level++) {
    for (let i = 0; i < n - level; i++) {
      points[i].slerp(points[i + 1], t); // Perform slerp
    }
  }

  return points[0].normalize(); // Return the final interpolated quaternion
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
      return zip(p0, p1).map(x => computeSingle(...x));
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
      localt = (t * (n - 1)) - segment;
    } else {
      segment = Math.floor(t * n);
      localt = (t * n) - segment;
    }

    const clamp_seg = (x) => Math.min(Math.max(x, 0), n - 1);
    let p0 = this.vertices[this.loop ? (segment - 1 + n) % n : clamp_seg(segment - 1)];
    let p1 = this.vertices[this.loop ? segment % n : clamp_seg(segment)];
    let p2 = this.vertices[this.loop ? (segment + 1) % n : clamp_seg(segment + 1)];
    let p3 = this.vertices[this.loop ? (segment + 2) % n : clamp_seg(segment + 2)];

    const a = (1 - this.tension) * (1 + this.continuity) * (1 + this.bias) / 2;
    const b = (1 - this.tension) * (1 - this.continuity) * (1 - this.bias) / 2;
    const c = (1 - this.tension) * (1 - this.continuity) * (1 + this.bias) / 2;
    const d = (1 - this.tension) * (1 + this.continuity) * (1 - this.bias) / 2;

    // Hermite basis
    const t2 = localt * localt;
    const t3 = t2 * localt;

    const h00 =  2*t3 -3*t2 +1;
    const h01 = -2*t3 +3*t2;
    const h10 =    t3 -2*t2 +localt;
    const h11 =    t3   -t2;

    // Notes:
    // m0 is the outgoing tangent of p1
    // m1 is the incoming tangent of p2

    function computeSingle(x0, x1, x2, x3) {
      const m0 = (x1 - x0) * a + (x2 - x1) * b;
      const m1 = (x2 - x1) * c + (x3 - x2) * d;
      return h00 * x1 + 
             h01 * x2 +
             h10 * m0 + 
             h11 * m1;
    }
    computeSingle = computeSingle.bind(this);

    if (p0 instanceof THREE.Vector3 || p0 instanceof THREE.Vector2) {
      const m0 = p1.clone().sub(p0).multiplyScalar(a).add(p2.clone().sub(p1).multiplyScalar(b));
      const m1 = p2.clone().sub(p1).multiplyScalar(c).add(p3.clone().sub(p2).multiplyScalar(d));
      return p1.clone().multiplyScalar(h00)
        .add(p2.clone().multiplyScalar(h01))
        .add(m0.multiplyScalar(h10))
        .add(m1.multiplyScalar(h11));
    } else if (p0 instanceof THREE.Quaternion) {
      // Cannonicalize
      if (new THREE.Quaternion().dot(p0) < 0)
        p0 = new THREE.Quaternion(-p0.x, -p0.y, -p0.z, -p0.w);
      if (p0.dot(p1) < 0)
        p1 = new THREE.Quaternion(-p1.x, -p1.y, -p1.z, -p1.w);
      if (p1.dot(p2) < 0)
        p2 = new THREE.Quaternion(-p2.x, -p2.y, -p2.z, -p2.w);
      if (p2.dot(p3) < 0)
        p3 = new THREE.Quaternion(-p3.x, -p3.y, -p3.z, -p3.w);
      const m0_ = logQ(p1.clone().multiply(p0.clone().invert())).multiplyScalar(a)
             .add(logQ(p2.clone().multiply(p1.clone().invert())).multiplyScalar(b));
      const m1_ = logQ(p2.clone().multiply(p1.clone().invert())).multiplyScalar(c)
             .add(logQ(p3.clone().multiply(p2.clone().invert())).multiplyScalar(d));
      let m0 = expQ(m0_.multiplyScalar(1/3)).multiply(p1);
      let m1 = expQ(m1_.multiplyScalar(-1/3)).multiply(p2);
      if (!this.loop && n == 2) {
        // "cubic" spline, degree 3
        const offset = powQ(p2.clone().multiply(p1.clone().invert()), 1/3);
        m0 = offset.clone().multiply(p1);
        m1 = offset.clone().invert().multiply(p2);
      } else if (!this.loop && segment == 0) {
        m0 = powQ(m1.clone().multiply(p1.clone().invert()), 1/2).multiply(p1);
      } else if (!this.loop && segment == n - 2) {
        m1 = powQ(m0.clone().multiply(p2.clone().invert()), 1/2).multiply(p2);
      }
      return deCasteljauQuaternions([p1, m0, m1, p2], localt);
    } else if (Array.isArray(p0)) {
      return zip(p0, p1, p2, p3).map((x) => computeSingle(...x));
    } else {
      return computeSingle(p0, p1, p2, p3);
    }
  }
}


function interpolate_ellipse(camera_path_keyframes, num_frames) {
  /*
  if (num_frames <= 0 || camera_path_keyframes.length < 3) {
    return undefined;
  }

  const centroid = camera_path_keyframes.reduce(
    (acc, x) => acc.add(x), new THREE.Vector3()).multiplyScalar(1/camera_path_keyframes.length);
  const centered_pointas = camera_path_keyframes.map(x => x.position.clone().sub(centroid));

  // Singular Value Decomposition (SVD)
  const [U, S, Vt] = np.linalg.svd(centered_points);
  const normal_vector = Vt[Vt.length-1];

  // Project the points onto the plane
  const projection_matrix = THREE.Matrix3.eye(3) - THREE.outer(normal_vector, normal_vector)
  const projected_points = centered_points.map(x => x.multiply(projection_matrix));

  // Now, we have points in a 2D plane, fit a circle in 2D
  const A = np.c_[2*projected_points[:,0], 2*projected_points[:,1], np.ones(projected_points.shape[0])]
  const b = projected_points.map(x => x.x^2 + x.y^2 + x.z^2);
  const x = np.linalg.lstsq(A, b, rcond=None)[0]
  const center_2d = x[:2]
  const radius = np.sqrt(x[2] + np.sum(center_2d**2))

  // Reproject the center back to 3D
  const angles = np.linspace(0, 2*Math.PI, int(num_frames), endpoint=False)
  const points_array = angles.map(angle => {
    let position = [center_2d[0] + radius * Math.cos(angle),
                      center_2d[1] + radius * Math.sin(angle)];
    position = position.multiply(projection_matrix[:2, :2].T);
    let position3 = new THREE.Vector3(position[0], position[1], 0);
    return position3.add(centroid);
  });

  poses = np.stack([get_c2w(k.position, k.wxyz) for k in camera_path_keyframes], axis=0)
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]

  // Compute camera orientation
  let oriented_normal = undefined;
  const quaternions = points_array.map(x => {
    const dirz = focus_pt.clone().sub(x).normalize();
    if (oriented_normal === undefined)
      oriented_normal = normal_vector.dot(dirz) > 0 ? normal_vector.clone().multiplyScalar(-1) : normal_vector;
    dirx = dirz.clone().cross(oriented_normal);
    diry = dirz.clone().cross(dirx);
    R = np.stack([dirx, diry, dirz], axis=-1)
    return THREE.Quaternion().fromRotationMatrix(R);
  };

  // TODO: implement rest
  fovs = np.full(num_frames, render_fov, dtype=np.float32)
  weights = _onehot(0, len(camera_path_keyframes))[np.newaxis].repeat(num_frames, axis=0)
  return points_array, orientation_array, fovs, weights
  */
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

  if (keyframes.length < 2) {
    return undefined;
  }

  let times = keyframes.map(k => k.transition_duration || default_transition_duration);
  if (!loop) {
    times.pop();
  }
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
  const weights_spline = new Interpolation({ vertices: keyframes.map((_, i) => onehot(i, keyframes.length)), loop: loop, ...rest_props });

  const gtime = Array.from({ length: num_frames }, (_, i) => (i / (num_frames-1)));
  const positions = gtime.map(t => position_spline.evaluate(t));
  const quaternions = gtime.map(t => quaternion_spline.evaluate(t));
  const fovs = gtime.map(t => fov_spline.evaluate(t));
  const weights = gtime.map(t => weights_spline.evaluate(t));
  return { positions, quaternions, fovs, weights };
}
