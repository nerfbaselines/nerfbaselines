/*
 Currently, there are the following bugs:
 - Interpolation for fov and weights needs to ignore undefined values. Also, we need to support grid which does not start from 0.

 */
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


function bisectRight(arr, x) {
  let left = 0;
  let right = arr.length;

  while (left < right) {
    const mid = Math.floor((left + right) / 2);
    if (x < arr[mid]) {
      right = mid;
    } else {
      left = mid + 1;
    }
  }
  return left;
}


function getLocalTime(grid, t) {
  const maxT = grid[grid.length - 1];
  t = Math.min(Math.max(t, 0), maxT);
  if (t > maxT - 1e-6) {
    return { segment: grid.length - 2, localt: 1 };
  }
  let segment = bisectRight(grid, t) - 1;
  segment = Math.min(segment, grid.length - 2);
  const localt = (t - grid[segment]) / (grid[segment + 1] - grid[segment]);
  return { segment, localt };
}


function cumsum(arr) {
  const out = [0];
  arr.forEach((x, i) => out.push(out[i] + x));
  return out;
}


class LinearInterpolation {
  constructor({ vertices, grid }) {
    this.vertices = vertices
    this.evaluate = this.evaluate.bind(this);
    this.grid = grid;
  }

  static getSegmentLengths({ positions, loop }) {
    const n = positions.length;
    if (n === 0) return [];
    if (n === 1) return [0];

    const out = [];
    const maxI = loop ? n : n - 1;
    for (let i = 1; i <= maxI; i++) {
      const p0 = positions[i - 1];
      const p1 = positions[i % positions.length];
      const length = p0.distanceTo(p1);
      out.push(length);
    }
    return out;
  }

  evaluate(t) {
    const n = this.vertices.length;
    if (n < 2) return this.vertices[0];

    const { segment, localt } = getLocalTime(this.grid, t);
    const p0 = this.vertices[segment % n];
    const p1 = this.vertices[(segment + 1) % n];

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


function reduceGrid(Interpolation, {grid, vertices, defaultValue = undefined, ...kwargs}) {
  // Reduce the grid to only include vertices that are not null or undefined
  const loop = vertices.length < grid.length;
  const maxT = grid[grid.length - 1];
  let reducedGrid = vertices.map((v, i) => 
    (v !== undefined && v !== null) ? grid[i] : undefined
  ).filter(x => x !== undefined);
  const reducedVertices = vertices.filter(v => v !== undefined && v !== null);
  if (reducedGrid.length === 0) {
    return { evaluate: (t) => defaultValue };
  }
  if (!loop) {
    if (reducedGrid[0] > 0) {
      reducedGrid.unshift(0);
      reducedVertices.unshift(reducedVertices[0]);
    }
    if (reducedGrid[reducedGrid.length - 1] < maxT) {
      reducedGrid.push(maxT);
      reducedVertices.push(reducedVertices[reducedVertices.length - 1]);
    }
    return new Interpolation({ vertices: reducedVertices, grid: reducedGrid, ...kwargs });
  } else {
    const offset = reducedGrid[0];
    reducedGrid = reducedGrid.map(x => x - offset);
    reducedGrid.push(maxT);
    const interpolation = new Interpolation({ 
      vertices: reducedVertices, 
      grid: reducedGrid,
      ...kwargs 
    });
    return {
      evaluate: (t) => {
        t = (maxT + t - offset) % maxT;
        return interpolation.evaluate(t);
      }
    };
  }
}


class KochanekBartelsInterpolation {
  constructor({ vertices, tension = 0, continuity = 0, bias = 0, grid }) {
    this.vertices = vertices;
    this.tension = tension;
    this.continuity = continuity;
    this.bias = bias;
    this.grid = grid;
    this.loop = vertices.length < grid.length;
    this.evaluate = this.evaluate.bind(this);
  }

  static getSegmentLengths({ positions, loop = false, ...kwargs}) {
    const n = positions.length;
    if (n === 0) return [];
    if (n === 1) return [0];
    const maxI = loop ? n : n - 1;
    const grid = cumsum(Array.from({ length: maxI }, () => 1));

    // Approximate the length of each segment by sampling points along the curve
    const interpolation = new KochanekBartelsInterpolation({ vertices: positions, grid, ...kwargs });
    const numSamples = 20;
    const out = [];
    for (let i = 1; i <= maxI; i++) {
      const positions = Array.from({ length: numSamples }, (_, j) => interpolation.evaluate(i - 1 + j / numSamples));
      const length = positions.reduce((acc, p, j, arr) => {
        if (j === 0) return acc;
        return acc + p.distanceTo(arr[j - 1]);
      }, 0);
      out.push(length);
    }
    return out;
  }

  evaluate(t) {
    const n = this.vertices.length;
    if (n < 2) return this.vertices[0];
    const { segment, localt } = getLocalTime(this.grid, t);

    const clamp_seg = (x) => Math.min(Math.max(x, 0), n - 1);
    let p0 = this.vertices[this.loop ? (segment - 1 + n) % n : clamp_seg(segment - 1)];
    let p1 = this.vertices[this.loop ? segment % n : clamp_seg(segment)];
    let p2 = this.vertices[this.loop ? (segment + 1) % n : clamp_seg(segment + 1)];
    let p3 = this.vertices[this.loop ? (segment + 2) % n : clamp_seg(segment + 2)];

    let d0 = segment > 0 ? this.grid[segment] - this.grid[segment-1]:
      (!this.loop) ? 0 : this.grid[this.grid.length-1] - this.grid[this.grid.length-2];
    let d1 = this.grid[segment+1] - this.grid[segment];
    let d2 = segment+2<this.grid.length?this.grid[segment+2] - this.grid[segment+1]:
      (!this.loop) ? 0 : this.grid[1] - this.grid[0];

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
      let m0 = ((x1 - x0) * a * (d1/d0) + (x2 - x1) * b * (d0/d1)) / (d0+d1);
      let m1 = ((x2 - x1) * c * (d2/d1) + (x3 - x2) * d * (d1/d2)) / (d1+d2);
      if (!this.loop && n == 2) {
        m0 = m1 = (x2 - x1)/d1;
      } else if (!this.loop && segment == 0) {
        m0 = 3/2 * (x2 - x1)/d1 - m1/2;
      } else if (!this.loop && segment == n-2) {
        m1 = 3/2 * (x2 - x1)/d1 - m0/2;
      }
      return h00 * x1 + 
             h01 * x2 +
             h10 * m0 * d1 + 
             h11 * m1 * d1;
    }
    computeSingle = computeSingle.bind(this);

    if (p0 instanceof THREE.Vector3 || p0 instanceof THREE.Vector2) {
      let m0 = p1.clone().sub(p0).multiplyScalar(a * (d1/d0)).add(p2.clone().sub(p1).multiplyScalar(b * d0/d1)).multiplyScalar(1/(d0+d1));
      let m1 = p2.clone().sub(p1).multiplyScalar(c * (d2/d1)).add(p3.clone().sub(p2).multiplyScalar(d * d1/d2)).multiplyScalar(1/(d1+d2));
      if (!this.loop && n == 2) {
        m0 = m1 = p2.clone().sub(p1).multiplyScalar(1/d1);
      } else if (!this.loop && segment == 0) {
        m0 = p2.clone().sub(p1).multiplyScalar(3/2/d1).sub(m1.clone().multiplyScalar(1/2));
      } else if (!this.loop && segment == n - 2) {
        m1 = p2.clone().sub(p1).multiplyScalar(3/2/d1).sub(m0.clone().multiplyScalar(1/2));
      }
      return p1.clone().multiplyScalar(h00)
        .add(p2.clone().multiplyScalar(h01))
        .add(m0.multiplyScalar(h10 * d1))
        .add(m1.multiplyScalar(h11 * d1));
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
      const m0_ = logQ(p1.clone().multiply(p0.clone().invert())).multiplyScalar(a*d1/d0)
             .add(logQ(p2.clone().multiply(p1.clone().invert())).multiplyScalar(b*d0/d1))
             .multiplyScalar(1/(d0+d1));
      const m1_ = logQ(p2.clone().multiply(p1.clone().invert())).multiplyScalar(c*d2/d1)
             .add(logQ(p3.clone().multiply(p2.clone().invert())).multiplyScalar(d*d1/d2))
            .multiplyScalar(1/(d1+d2));
      let m0 = expQ(m0_.multiplyScalar(d1/3)).multiply(p1);
      let m1 = expQ(m1_.multiplyScalar(-d1/3)).multiply(p2);
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


class CircleInterpolation {
  constructor({ positions, quaternions, loop = false }) {
    const up = new THREE.Vector3(0, 0, 0);
    quaternions.forEach((q, i) => up.add(new THREE.Vector3(0, 1, 0).applyQuaternion(q)));
    up.normalize();
    const { center, normal, radius, points2D, center2D } = fitCircleToPoints3D({ points: positions, up });
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

    this._q = new THREE.Quaternion();
    this._v = new THREE.Vector3();
    this._m = new THREE.Matrix4();

    this._start = positions[0].clone().sub(center).cross(normal).cross(normal).multiplyScalar(-1).normalize();
    this._center = center;
    this._radius = radius;
    this._normal = normal;
    this._angles = angles;
    this._angleDiffs = angleDiffs;
    this._n = positions.length;
    this._segmentLengths = this._getSegmentLengths({ loop });
    this.grid = cumsum(this._segmentLengths);
    this.evaluatePosition = this.evaluatePosition.bind(this);
    this.evaluateQuaternion = this.evaluateQuaternion.bind(this);
  }

  getSegmentLengths() {
    return this._segmentLengths;
  }

  _getSegmentLengths({ loop }) {
    if (this._n === 0) return [];
    if (this._n === 1) return [0];
    const out = [];
    const maxI = loop ? this._n + 1 : this._n;
    for (let i = 1; i < maxI; i++) {
      const length = Math.abs(this._angleDiffs[i-1]) * this._radius;
      out.push(length);
    }
    return out;
  }

  evaluatePosition(t) {
    const n = this._n;
    const { segment, localt } = getLocalTime(this.grid, t);

    const angle = this._angles[segment%this._angles.length] + this._angleDiffs[segment%this._angles.length] * localt;
    this._q.setFromAxisAngle(this._normal, angle);
    return this._start.clone().applyQuaternion(this._q).multiplyScalar(this._radius).add(this._center);
  }

  evaluateQuaternion(t) {
    const point = this.evaluatePosition(t);
    this._m.lookAt(this._center, point, this._normal).decompose(this._v, this._q, this._v);
    return this._q.clone();
  }
}


class PchipInterpolation {
  constructor({ x, y, loop=false }) {
    if (y.length === x.length-1) {
      y = [...y, y[0]];
      loop = true;
    }
    if (x.length !== y.length) {
      throw new Error("Input arrays x and y must have the same length.");
    }
    this.n = x.length;
    this.grid = x;
    this.y = y;

    // Step 1: Compute slopes and differences
    this.h = Array(this.n - 1).fill(0).map((_, i) => x[i + 1] - x[i]);
    this.slopes = Array(this.n - 1).fill(0).map((_, i) => (y[i + 1] - y[i]) / this.h[i]);

    // Step 2: Compute derivatives
    this.derivatives = Array(this.n).fill(0);
    for (let i = 1; i < this.n - 1; i++) {
      if (this.slopes[i - 1] * this.slopes[i] > 0) {
        const sumSlopes = this.slopes[i - 1] + this.slopes[i];
        if (Math.abs(sumSlopes) > 1e-12) {
          this.derivatives[i] = (2 * this.slopes[i - 1] * this.slopes[i]) / sumSlopes;
        }
      }
    }

    if (loop) {
      // Circular derivatives at endpoints
      if (this.slopes[this.n - 2] * this.slopes[0] > 0) {
        const sumSlopes = this.slopes[this.n - 2] + this.slopes[0];
        this.derivatives[0] = (2 * this.slopes[this.n - 2] * this.slopes[0]) / sumSlopes;
        this.derivatives[this.n - 1] = this.derivatives[0]; // Continuity in loop
      }
    } else {
      // Non-looping endpoint derivatives
      this.derivatives[0] = this.slopes[0] || 0;
      this.derivatives[this.n - 1] = this.slopes[this.n - 2] || 0;
    }
  }

  evaluate(x) {
    const { segment, localt } = getLocalTime(this.grid, x);
    const t = localt;
    const h00 = (1 + 2 * t) * (1 - t) ** 2;
    const h10 = t * (1 - t) ** 2;
    const h01 = t ** 2 * (3 - 2 * t);
    const h11 = t ** 2 * (t - 1);
    return (
      h00 * this.y[segment] +
      h10 * this.h[segment] * this.derivatives[segment] +
      h01 * this.y[segment + 1] +
      h11 * this.h[segment] * this.derivatives[segment + 1]
    );
  }

  evaluateIntegral(x) {
    // 1) If we haven't already, build an array of the *cumulative area* at each knot.
    if (!this._segmentArea) this._precomputeSegmentAreas();

    // 2) Figure out which segment x is in, plus local fraction t
    const { segment, localt } = getLocalTime(this.grid, x);
    // Sum of the full areas from segment 0 up to segment-1
    const baseArea = this._segmentArea[segment];

    // 3) Add partial area from the start of segment => up to localt
    const partial = this._partialSegmentArea(segment, localt);
    return baseArea + partial;
  }

  getSegmentIntegrals() {
    if (!this._segmentArea) this._precomputeSegmentAreas();
    return this._segmentArea;
  }

  _precomputeSegmentAreas() {
    const nSeg = this.loop ? this.n : (this.n - 1); 
    const areas = [0]; // cumulative area

    for (let seg = 0; seg < nSeg; seg++) {
      const areaSeg = this._partialSegmentArea(seg, 1.0);
      areas.push(areas[areas.length - 1] + areaSeg);
    }
    this._segmentArea = areas;
  }

  _partialSegmentArea(segment, tEnd) {
    if (tEnd <= 0) return 0;
    tEnd = Math.min(tEnd, 1); // clamp to [0..1]
    const t = tEnd;
    const H00 = t - t**3 + 0.5*t**4;
    const H10 = (t**2)/2 - (2*t**3)/3 + (t**4)/4;
    const H01 = t**3 - 0.5*t**4;
    const H11 = (t**4)/4 - (t**3)/3;
    return (
      H00 * this.y[segment] +
      H10 * this.h[segment] * this.derivatives[segment] +
      H01 * this.y[segment + 1] +
      H11 * this.h[segment] * this.derivatives[segment + 1]
    ) * this.h[segment];
  }
}


function _normalize(xs, value=1) {
  const sum = xs.reduce((a, x) => a + x, 0);
  return xs.map(x => x * value / sum);
}


function buildTimeDistanceMap({ grid, velocities, duration }) {
  // Note, we assume we have velocities defined at each knot.
  // We then want to interpolate the v(t) function, and integrate it to get the distance function.
  // However, we do not have the times for each know so we have to solve for them.
  // While parametrizing v(s) would be a solution, this leads to a non-linear differential equation.
  // Instead, we opt for an iterative approach where we adjust the times to match the distances.
  const distances = grid.map((x, i) => {
    if (i == 0) return 0;
    return x - grid[i - 1];
  }).slice(1);
  const totalDistance = grid[grid.length - 1];
  let evalDistancesSum;
  let previousEvalDistances = distances;
  let evalDistances = distances;
  let interpolator;
  let velocityMultiplier = 1;
  let times = distances.map((x, i) => {
    const velocity = (velocities[i] + velocities[(i+1)%velocities.length]) / 2;
    return x / velocity;
  });
  for (let iter = 0; iter < 100; iter++) {
    times = times.map((x, i) => {
      return x * (previousEvalDistances[i] / evalDistances[i]);
    });
    previousEvalDistances = evalDistances;
    times = _normalize(times, duration);
    interpolator = new PchipInterpolation({ x: cumsum(times), y: velocities });
    const evalDistancesAgg = interpolator.getSegmentIntegrals();
    evalDistancesSum = evalDistancesAgg[evalDistancesAgg.length - 1];
    evalDistances = evalDistancesAgg.map((x, i) => i == 0 ? x : x - evalDistancesAgg[i - 1]).slice(1);
    velocityMultiplier = totalDistance / evalDistancesSum;
    evalDistances = evalDistances.map(x => x * velocityMultiplier);
    const error = evalDistances.reduce((a, x, i) => Math.max(a, Math.abs(x - distances[i])), 0);
    if (error < 1e-6) {
      break;
    }
  }
  interpolator = new PchipInterpolation({ x: cumsum(times), y: velocities.map(x => x * velocityMultiplier) });
  return interpolator.evaluateIntegral.bind(interpolator);
}


function fixWeights(appearance) {
  if (appearance.length === 0) return [];
  if (appearance.length === 1) return [1];
  const indices = appearance.map((x,i) => [x,i]).sort((a,b) => a[0]<b[0]?1:-1).map(x=>x[1]);
  const [maxI, secondI] = indices.slice(0, 2);
  const maxV = Math.max(0, appearance[maxI]);
  const secondV = Math.max(0, appearance[secondI]);
  const out = Array.from({ length: appearance.length }, (_, i) => 0);
  out[maxI] = maxV === 0 ? 0 : maxV / (maxV + secondV);
  out[secondI] = secondV === 0 ? 0 : secondV / (maxV + secondV);
  return out;
}


function getStartsAndDurations(gdist, gtime, grid, num_keyframes, duration) {
  const timeGrid = Array.from({ length: num_keyframes }, () => undefined);
  for (let i = 0; i < gdist.length; i++) {
    const { segment } = getLocalTime(grid, gdist[i]);
    if (timeGrid[segment] === undefined) timeGrid[segment] = gtime[i];
  }
  timeGrid[0] = 0;
  timeGrid.push(duration);
  for (let i = timeGrid.length-2; i > 0; i--) {
    if (timeGrid[i] === undefined) {
      timeGrid[i] = timeGrid[i + 1];
    }
  }
  const keyframeStarts = timeGrid.slice(0, -1);
  const keyframeDurations = timeGrid.slice(1).map((x, i) => x - keyframeStarts[i]);
  return { 
    keyframeStarts,
    keyframeDurations,
  };
}


export function compute_camera_path(props) {
  const { 
    keyframes, 
    loop = false, 
    default_fov = 75, 
    framerate = 30, 
    interpolation = 'none', 
    time_interpolation = 'velocity',
    default_transition_duration,
    distance_alpha,
    ...rest_props 
  } = props;
  let duration = props.duration;
  const k_positions = keyframes.map(k => k.position);
  const k_quaternions = keyframes.map(k => k.quaternion);
  const k_fovs = keyframes.map(k => k.fov);
  const num_appearances = keyframes.reduce((a, x) => a + (x.appearance_train_index !== undefined), 0);
  let app_counter = 0;
  const k_weights = keyframes.map((k, i) => 
    k.appearance_train_index === undefined ? undefined : onehot(app_counter++, num_appearances));
  const appearanceTrainIndices = keyframes.map(k => k.appearance_train_index).filter(x => x !== undefined);
  let keyframeDurations = keyframes.map(k => (k.duration === undefined || k.duration === null) ? 
    default_transition_duration : k.duration);

  if (time_interpolation !== 'velocity' && time_interpolation !== 'time') {
    throw new Error(`Unknown time interpolation method: ${time_interpolation}`);
  }
  if (interpolation === 'none') {
    const keyframeStarts = cumsum(keyframeDurations).slice(0, -1);
    return {
      positions: k_positions,
      quaternions: k_quaternions,
      fovs: k_fovs.map(x => x === undefined ? default_fov : x),
      weights: k_weights,
      distance: 0,
      appearanceTrainIndices,
      keyframeStarts,
      keyframeDurations,
    };
  }
  if (keyframes.length === 0) { return undefined; }

  if (!loop) keyframeDurations.pop();
  if (time_interpolation === 'time') {
    duration = keyframeDurations.reduce((a, x) => a + x, 0);
  }

  let num_frames = Math.max(0, Math.floor(duration * framerate));
  if (isNaN(num_frames)) num_frames = 0;
  if (keyframes.length === 1) {
    return {
      positions: Array.from({ length: num_frames }, () => k_positions[0]),
      quaternions: Array.from({ length: num_frames }, () => k_quaternions[0]),
      fovs: Array.from({ length: num_frames }, () => k_fovs[0]),
      weights: Array.from({ length: num_frames }, () => k_weights[0]),
      distance: 0,
      appearanceTrainIndices,
      keyframeStarts: [0],
      keyframeDurations: [duration],
    };
  }
  
  let grid;
  let position_spline;
  let quaternion_spline;
  let fov_spline;
  let weights_spline;
  let lengths;
  let totalDistance;
  let Interpolation;
  let isTimeInterpolation = time_interpolation === 'time';
  if (interpolation === 'circle' && k_positions.length >= 3) {
    const circleInterpolation = new CircleInterpolation({
      positions: k_positions, quaternions: k_quaternions, loop,
    });
    lengths = circleInterpolation.getSegmentLengths();
    totalDistance = lengths.reduce((a, x) => a + x, 0);
    lengths = lengths.map(x => x ** distance_alpha);
    grid = cumsum(lengths);
    circleInterpolation.grid = grid;
    position_spline = circleInterpolation.evaluatePosition;
    quaternion_spline = circleInterpolation.evaluateQuaternion;
    Interpolation = LinearInterpolation;
  } else if (interpolation === 'linear' || (interpolation === 'circle' && k_positions.length < 3)) {
    Interpolation = LinearInterpolation;
    lengths = Interpolation.getSegmentLengths({ positions: k_positions, loop });
    totalDistance = lengths.reduce((a, x) => a + x, 0);
    lengths = lengths.map(x => x ** distance_alpha);
    grid = cumsum(lengths);
    position_spline = new Interpolation({ vertices: k_positions, grid, ...rest_props}).evaluate;
    quaternion_spline = new Interpolation({ vertices: k_quaternions, grid, ...rest_props}).evaluate;
  } else if (interpolation === 'kochanek-bartels') {
    Interpolation = KochanekBartelsInterpolation;
    lengths = Interpolation.getSegmentLengths({ positions: k_positions, loop: loop, ...rest_props});
    totalDistance = lengths.reduce((a, x) => a + x, 0);
    lengths = lengths.map(x => x ** distance_alpha);
    grid = cumsum(lengths);
    position_spline = new Interpolation({ vertices: k_positions, grid, ...rest_props}).evaluate;
    quaternion_spline = new Interpolation({ vertices: k_quaternions, grid, ...rest_props}).evaluate;
  } else {
    throw new Error(`Unknown interpolation method: ${interpolation}`);
  }
  fov_spline = reduceGrid(Interpolation, { vertices: k_fovs, grid, defaultValue: default_fov, ...rest_props }).evaluate;
  weights_spline = reduceGrid(Interpolation, { vertices: k_weights, grid, defaultValue: [], ...rest_props }).evaluate;

  let gdist, keyframeStarts;
  const gtime = Array.from({ length: num_frames }, (_, i) => i * duration / (num_frames - 1));
  if (time_interpolation === 'velocity') {
    const velocities = keyframes.map(x => x.velocity_multiplier || 1);
    const distanceMap = buildTimeDistanceMap({ grid, velocities: velocities, duration });
    gdist = gtime.map(t => distanceMap(t));
    const _out = getStartsAndDurations(gdist, gtime, grid, keyframes.length, duration);
    keyframeStarts = _out.keyframeStarts;
    keyframeDurations = _out.keyframeDurations;
    if (!loop) keyframeDurations[keyframeDurations.length - 1] = null;
  } else if (time_interpolation === 'time') {
    let x = cumsum(keyframeDurations);
    let y = cumsum(lengths);
    const interpolator = new PchipInterpolation({ x, y, loop });
    gdist = gtime.map(t => interpolator.evaluate(t));

    if (!loop) keyframeDurations.push(null);
    keyframeStarts = x.slice(0, keyframeDurations.length);
  } else {
    throw new Error(`Unknown time interpolation method: ${time_interpolation}`);
  }

  const positions = gdist.map(t => position_spline(t));
  const quaternions = gdist.map(t => quaternion_spline(t));
  const fovs = gdist.map(t => fov_spline(t));
  const weights = gdist.map(t => fixWeights(weights_spline(t)));
  return { 
    positions, 
    quaternions, 
    fovs, 
    weights, 
    distance: totalDistance,
    appearanceTrainIndices,
    keyframeStarts,
    keyframeDurations,
  };
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
