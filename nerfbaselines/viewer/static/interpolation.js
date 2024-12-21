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
    this.evaluate = this.evaluate.bind(this);
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
    this.evaluate = this.evaluate.bind(this);
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
  const k_positions = keyframes.map(k => k.position);
  const k_quaternions = keyframes.map(k => k.quaternion);
  const k_fovs = keyframes.map(k => k.fov || default_fov);
  const k_weights = keyframes.map((_, i) => onehot(i, keyframes.length));

  if (interpolation === 'none') {
    return {
      positions: k_positions,
      quaternions: k_quaternions,
      fovs: k_fovs,
      weights: k_weights,
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
  
  if (interpolation === 'circle') {
    const up = new THREE.Vector3(0, 0, 0);
    k_quaternions.forEach((q, i) => up.add(new THREE.Vector3(0, 1, 0).applyQuaternion(q)));
    up.normalize();
    const { center, normal, radius, points2D, center2D } = fitCircleToPoints3D({ points: k_positions, up });
    let angles = points2D.map(p => Math.atan2(p.y - center2D.y, p.x - center2D.x));
    angles = angles.map(x => (x - angles[0] + 3 * Math.PI) % (2 * Math.PI) - Math.PI);
    let angleDiffs = angles.map((x, i) => (angles[(i+1)%angles.length]-x+5*Math.PI) % (2*Math.PI)-Math.PI);

    // Unify winding direction
    const angleDiffsPos = angleDiffs.map(x => (x >= 0)? x : (2*Math.PI+x));
    const angleDiffsNeg = angleDiffs.map(x => (-x >= 0)? x : (-2*Math.PI+x));
    if (angleDiffsPos.reduce((a,x) => a+Math.abs(x), 0) < angleDiffsNeg.reduce((a,x) => a+Math.abs(x), 0)) {
      angleDiffs = angleDiffsPos;
    } else {
      angleDiffs = angleDiffsNeg;
    }

    // Create an elliptical path
    const positions = [];
    const quaternions = [];
    const fovs = [];
    const weights = [];

    const angleStep = 2 * Math.PI / (num_frames - 1);
    let q = new THREE.Quaternion();
    let v = new THREE.Vector3();
    let m = new THREE.Matrix4();

    const start = keyframes[0].position.clone().sub(center).cross(normal).cross(normal).multiplyScalar(-1).normalize();
    for (let i = 0; i < num_frames; i++) {
      const t = i / (num_frames - 1);

      let segment, localt;
      const n = keyframes.length;
      if (!loop) {
        segment = Math.floor(t * (n - 1));
        localt = (t * (n - 1)) % 1;
      } else {
        segment = Math.floor(t * n);
        localt = (t * n) % 1;
      }

      const angle = angles[segment%angles.length] + angleDiffs[segment%angles.length] * localt;
      q.setFromAxisAngle(normal, angle);
      const point = start.clone().applyQuaternion(q).multiplyScalar(radius).add(center);
      // const point = new THREE.Vector3(radius, 0, 0).applyQuaternion(q).add(center);
      m.lookAt(center, point, normal).decompose(v, q, v);

      positions.push(point);
      quaternions.push(q.clone());
      fovs.push(keyframes[0].fov || default_fov);
      weights.push(onehot(0, keyframes.length));
    }

    return { positions, quaternions, fovs, weights };
  }


  const Interpolation = {
    'linear': LinearInterpolation,
    'kochanek-bartels': KochanekBartelsInterpolation,
  }[interpolation];
  const position_spline = new Interpolation({ vertices: k_positions, loop: loop, ...rest_props}).evaluate;
  const quaternion_spline = new Interpolation({ vertices: k_quaternions, loop: loop, ...rest_props}).evaluate;
  const fov_spline = new Interpolation({ vertices: k_fovs, loop: loop, ...rest_props }).evaluate;
  const weights_spline = new Interpolation({ vertices: k_weights, loop: loop, ...rest_props }).evaluate;

  const gtime = Array.from({ length: num_frames }, (_, i) => (i / (num_frames-1)));
  const positions = gtime.map(t => position_spline(t));
  const quaternions = gtime.map(t => quaternion_spline(t));
  const fovs = gtime.map(t => fov_spline(t));
  const weights = gtime.map(t => weights_spline(t));
  return { positions, quaternions, fovs, weights };
}


// Circle interpolation
// 1. Compute the best-fitting plane
function computeBestFitPlane(points) {
  // Compute centroid
  let centroid = new THREE.Vector3();
  for (let p of points) {
    centroid.add(p);
  }
  centroid.multiplyScalar(1 / points.length);

  // Compute covariance matrix
  let xx = 0, xy = 0, xz = 0;
  let yy = 0, yz = 0, zz = 0;

  for (let p of points) {
    let x = p.x - centroid.x;
    let y = p.y - centroid.y;
    let z = p.z - centroid.z;

    xx += x * x;
    xy += x * y;
    xz += x * z;
    yy += y * y;
    yz += y * z;
    zz += z * z;
  }

  xx /= points.length;
  xy /= points.length;
  xz /= points.length;
  yy /= points.length;
  yz /= points.length;
  zz /= points.length;

  // Form the covariance matrix
  let cov = new THREE.Matrix3();
  cov.set(xx, xy, xz,
          xy, yy, yz,
          xz, yz, zz);

  let { eigenvalues, eigenvectors } = eigenDecomposition3x3(cov);

  // Find the smallest eigenvalue index
  let minIndex = 0;
  for (let i = 1; i < 3; i++) {
    if (eigenvalues[i] < eigenvalues[minIndex]) minIndex = i;
  }

  // The normal of the best-fitting plane is the eigenvector with the smallest eigenvalue.
  let normal = eigenvectors[minIndex];
  return { centroid, normal };
}

function eigenDecomposition3x3(matrix) {
    if (!(matrix instanceof THREE.Matrix3)) {
        throw new Error("Input must be a THREE.Matrix3 object.");
    }

    // Extract elements from the THREE.Matrix3 object
    const elements = matrix.elements;

    // Convert to a standard 2D array for easier manipulation
    const A = [
        [elements[0], elements[1], elements[2]],
        [elements[3], elements[4], elements[5]],
        [elements[6], elements[7], elements[8]]
    ];

    // Helper function: Calculate eigenvalues using the characteristic polynomial
    function computeEigenvalues(matrix) {
        const a = matrix[0][0], b = matrix[0][1], c = matrix[0][2];
        const d = matrix[1][0], e = matrix[1][1], f = matrix[1][2];
        const g = matrix[2][0], h = matrix[2][1], i = matrix[2][2];

        // Compute coefficients of the characteristic polynomial: det(A - λI)
        const p1 = -(a + e + i); // -trace(A)
        const p2 = a * e + a * i + e * i - b * d - c * g - f * h; // sum of 2x2 determinants
        const p3 = -(
            a * (e * i - f * h) -
            b * (d * i - f * g) +
            c * (d * h - e * g)
        ); // -det(A)

        // Solve the cubic equation λ^3 + p1λ^2 + p2λ + p3 = 0 for eigenvalues
        return solveCubicEquation(1, p1, p2, p3);
    }

    // Helper function: Solve a cubic equation using numerical methods
    function solveCubicEquation(a, b, c, d) {
        // Normalize coefficients
        b /= a; c /= a; d /= a;

        const p = (3 * c - b * b) / 3;
        const q = (2 * b * b * b - 9 * b * c + 27 * d) / 27;
        const delta = (q * q) / 4 + (p * p * p) / 27;

        if (delta > 0) {
            // One real root
            const sqrtDelta = Math.sqrt(delta);
            const u = Math.cbrt(-q / 2 + sqrtDelta);
            const v = Math.cbrt(-q / 2 - sqrtDelta);
            return [u + v - b / 3];
        } else if (delta === 0) {
            // All roots real, at least two are equal
            const u = Math.cbrt(-q / 2);
            return [2 * u - b / 3, -u - b / 3];
        } else {
            // Three distinct real roots
            const r = Math.sqrt(-(p * p * p) / 27);
            const phi = Math.acos(-q / (2 * r));
            const root1 = 2 * Math.cbrt(r) * Math.cos(phi / 3) - b / 3;
            const root2 = 2 * Math.cbrt(r) * Math.cos((phi + 2 * Math.PI) / 3) - b / 3;
            const root3 = 2 * Math.cbrt(r) * Math.cos((phi + 4 * Math.PI) / 3) - b / 3;
            return [root1, root2, root3];
        }
    }

    // Compute eigenvalues
    const eigenvalues = computeEigenvalues(A);

    // Helper function: Compute eigenvectors for a given eigenvalue
    function computeEigenvector(matrix, eigenvalue) {
        const size = matrix.length;
        const m = matrix.map((row, i) =>
            row.map((val, j) => (i === j ? val - eigenvalue : val))
        );

        const v3 = 1.0;
        // Now we solve:
        // v1*m[0][0] + v2*m[1][0] = -v3*m[2][0]
        // v1*m[0][1] + v2*m[1][1] = -v3*m[2][1]
        const c = -m[1][0] / m[1][1];
        const v1 = -v3 * (m[2][0] + m[2][1]*c) / (m[0][0] + m[0][1]*c);
        const v2 = -(v3 * m[2][1] + v1 * m[0][1]) / m[1][1];
        return new THREE.Vector3(v1, v2, v3).normalize();
    }

    // Compute eigenvectors for each eigenvalue
    const eigenvectors = eigenvalues.map((λ) => computeEigenvector(A, λ));

    return {
        eigenvalues,
        eigenvectors
    };
}

// 2. Project points onto the best-fit plane
function projectPointsOntoPlane(points, centroid, normal) {
  // Create an orthonormal basis (u,v) for the plane
  // u is perpendicular to normal and arbitrary axis
  let arbitrary = new THREE.Vector3(1,0,0);
  if (Math.abs(normal.dot(arbitrary)) > 0.9) {
    arbitrary.set(0,1,0);
  }
  let u = new THREE.Vector3().crossVectors(normal, arbitrary).normalize();
  let v = new THREE.Vector3().crossVectors(normal, u).normalize();

  let projectedPoints = [];
  for (let p of points) {
    let diff = new THREE.Vector3().subVectors(p, centroid);
    let x = diff.dot(u);
    let y = diff.dot(v);
    projectedPoints.push({x, y});
  }

  return { projectedPoints, u, v };
}

// 3. Fit a circle in 2D
// Using a linear least squares approach to circle fitting:
// We solve the system for A,B,C in x² + y² + A x + B y + C = 0.
function fitCircle2D(projectedPoints) {
  let X = [], Y = [], Z = [];

  for (let pt of projectedPoints) {
    let {x, y} = pt;
    X.push(x);
    Y.push(y);
    Z.push(x*x + y*y);
  }

  // We want to solve:
  // [X Y 1][A  ] = [-Z]
  //      [B  ]
  //      [C  ]
  //
  // i.e. M * param = -Z
  // param = (A,B,C)

  let M = [0,0,0, 0,0,0, 0,0,0];
  let R = [0,0,0];

  let n = projectedPoints.length;
  for (let i=0; i<n; i++) {
    M[0] += X[i]*X[i]; M[1] += X[i]*Y[i]; M[2] += X[i];
    M[3] += X[i]*Y[i]; M[4] += Y[i]*Y[i]; M[5] += Y[i];
    M[6] += X[i];    M[7] += Y[i];    M[8] += 1;

    R[0] += X[i]*Z[i];
    R[1] += Y[i]*Z[i];
    R[2] += Z[i];
  }

  // Solve M * [A;B;C] = -R
  // We need to invert M and multiply by -R
  let invM = invert3x3(M);
  if (!invM) {
    throw new Error("Matrix inversion failed. Points may be degenerate.");
  }

  let param = [
    -(invM[0]*R[0] + invM[1]*R[1] + invM[2]*R[2]),
    -(invM[3]*R[0] + invM[4]*R[1] + invM[5]*R[2]),
    -(invM[6]*R[0] + invM[7]*R[1] + invM[8]*R[2])
  ];

  let A = param[0], B = param[1], C = param[2];
  let xC = -A/2;
  let yC = -B/2;
  let r = Math.sqrt(xC*xC + yC*yC - C);

  return {xC, yC, r};
}

// Invert a 3x3 matrix given as array [m11,m12,m13,m21,m22,m23,m31,m32,m33]
function invert3x3(m) {
  let det = m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6]) + m[2]*(m[3]*m[7]-m[4]*m[6]);
  if (Math.abs(det)<1e-14) return null;

  let invDet = 1.0/det;
  let inv = [
    (m[4]*m[8]-m[5]*m[7])*invDet,
    (m[2]*m[7]-m[1]*m[8])*invDet,
    (m[1]*m[5]-m[2]*m[4])*invDet,
    (m[5]*m[6]-m[3]*m[8])*invDet,
    (m[0]*m[8]-m[2]*m[6])*invDet,
    (m[2]*m[3]-m[0]*m[5])*invDet,
    (m[3]*m[7]-m[4]*m[6])*invDet,
    (m[1]*m[6]-m[0]*m[7])*invDet,
    (m[0]*m[4]-m[1]*m[3])*invDet
  ];
  return inv;
}

// 4. Map the 2D circle back to 3D
function circle2DTo3D(xC, yC, r, centroid, u, v) {
  let center3D = new THREE.Vector3().addVectors(
    centroid,
    new THREE.Vector3().addScaledVector(u, xC).addScaledVector(v, yC)
  );
  return { center3D, radius: r };
}

// Example usage:
function fitCircleToPoints3D({ points, up }) {
  // Compute the best-fitting plane
  let { centroid, normal } = computeBestFitPlane(points);
  if (normal.dot(up) < 0) {
    normal.negate();
  }

  // Project onto plane
  let { projectedPoints, u, v } = projectPointsOntoPlane(points, centroid, normal);

  // Fit circle in 2D
  let {xC, yC, r} = fitCircle2D(projectedPoints);

  // Map back to 3D
  let {center3D, radius} = circle2DTo3D(xC, yC, r, centroid, u, v);

  // 'center' is a THREE.Vector3 for the circle center
  // 'normal' is a THREE.Vector3 for the circle normal
  // 'radius' is the radius of the best fitting circle
  return { 
    center: center3D,
    normal, 
    radius, 
    points2D: projectedPoints.map(({x, y}) => new THREE.Vector2(x, y)),
    center2D: new THREE.Vector2(xC, yC),
  };
}

window.eigenDecomposition3x3 = eigenDecomposition3x3;
window.THREE = THREE;
