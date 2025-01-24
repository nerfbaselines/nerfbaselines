import * as THREE from 'three';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';


const registeredInvisibleLineMaterials = new Set();
const getAxisVector = (axisIndex) => new THREE.Vector3(...[0, 0, 0].map((v, i) => (i === axisIndex ? 1 : 0)));


function buildSetHover(obj, hoveredColor) {
  let isHovered = false;
  return (hover) => {
    if (isHovered === hover) return
    for (let child of obj.children) {
      if (!child.material || !child.material.color) continue;
      if (!isHovered)
        child.material._backup_color = child.material.color.clone();
      if (hover) {
        child.material.color.set(hoveredColor);
      } else if (child.material._backup_color) {
        child.material.color.set(child.material._backup_color);
      }
    }
    isHovered = hover;
  }
}


class AxisArrow extends THREE.Group {
  constructor({
    scale,
    axis,
    linewidth,
    fixed,
    axisColors,
    hoveredColor,
    onDragStart,
    onDrag,
    onDragEnd,
  }) {
    super();
    const direction = getAxisVector(axis);

    const coneWidth = fixed ? (linewidth / scale) * 1.6 : scale / 20;
    const coneLength = fixed ? 0.2 : scale / 5;
    const cylinderLength = fixed ? 1 - coneLength : scale - coneLength;
    const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
    const matrixL = new THREE.Matrix4().makeRotationFromQuaternion(quaternion);
    this.group = new THREE.Group();
    this.group.matrixAutoUpdate = false;
    this.group.matrix = matrixL;
    this.add(this.group);

    // Add line
    const geometry = new LineGeometry();
    geometry.setPositions([0, 0, 0, 0, cylinderLength, 0]);
    const line = new Line2(
      geometry,
        new LineMaterial({
            color: axisColors[axis],
            linewidth: linewidth,
            transparent: true,
        })
    );
    this.group.add(line);

    // Add cone
    const cone = new THREE.Mesh(
        new THREE.ConeGeometry(coneWidth, coneLength, 24),
        new THREE.MeshBasicMaterial({
            color: axisColors[axis],
            transparent: true,
        })
    );
    cone.position.set(0, cylinderLength + coneLength / 2, 0);
    this.group.add(cone);

    this._addPointerInteractions({
      coneWidth,
      coneLength,
      cylinderLength,
      hoveredColor,
      axis,
      direction,
      onDragStart,
      onDrag,
      onDragEnd,
    });
  }

  _addPointerInteractions({ coneWidth, coneLength, cylinderLength, hoveredColor, axis, direction, onDragStart, onDrag, onDragEnd }) {
    const setHover = buildSetHover(this.group, hoveredColor);
    let clickInfo = undefined;

    // Invisible mesh for raycasting
    const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(coneWidth * 1.4, coneWidth * 1.4, cylinderLength + coneLength, 8, 1),
        new THREE.MeshBasicMaterial({ visible: false })
    );
    mesh.position.set(0, (cylinderLength + coneLength) / 2, 0);
    this.group.add(mesh);

    mesh.addEventListener('pointerdown', (e) => {
      e.stopPropagation();
      const rotation = new THREE.Matrix4().extractRotation(this.matrixWorld);
      const origin = new THREE.Vector3().setFromMatrixPosition(this.matrixWorld);
      const dir = direction.clone().applyMatrix4(rotation).normalize();

      clickInfo = { clickPoint: e.point.clone(), dir };
      onDragStart({ component: 'Arrow', axis, origin, directions: [dir] });
      e.setPointerCapture(e.pointerId);
    });

    mesh.addEventListener('pointermove', (e) => {
      e.stopPropagation();
      setHover(true);
      if (!clickInfo) return;
      const { clickPoint, dir } = clickInfo;
      let offset = this._calculateOffset(clickPoint, dir, e.ray.origin, e.ray.direction);
      const offsetMatrix = new THREE.Matrix4().makeTranslation(dir.x * offset, dir.y * offset, dir.z * offset);
      onDrag(offsetMatrix);
    });
    
    mesh.addEventListener('pointerup', (e) => {
      e.stopPropagation();
      clickInfo = undefined;
      onDragEnd();
      e.releasePointerCapture(e.pointerId);
    });

    mesh.addEventListener('pointerout', () => setHover(false));
  }

  _calculateOffset(clickPoint, normal, rayStart, rayDir) {
    const e1 = normal.dot(normal)
    const e2 = normal.dot(clickPoint) - normal.dot(rayStart)
    const e3 = normal.dot(rayDir)

    if (e3 === 0) {
      return -e2 / e1
    }

    const vec1 = rayDir.clone()
      .multiplyScalar(e1 / e3)
      .sub(normal);
    const vec2 = rayDir.clone()
      .multiplyScalar(e2 / e3)
      .add(rayStart)
      .sub(clickPoint)

    return -vec1.dot(vec2) / vec1.dot(vec1)
  }
}


class AxisRotator extends THREE.Group {
  constructor({ 
    axis,
    scale,
    fixed,
    linewidth,
    axisColors,
    hoveredColor,
    onDragStart,
    onDrag,
    onDragEnd,
  }) {
    super();

    this.matrixAutoUpdate = false;

    // Calculate basis matrix
    const dir1N = getAxisVector((axis + 1) % 3);
    const dir2N = getAxisVector((axis + 2) % 3);
    const matrixL = new THREE.Matrix4().makeBasis(dir1N, dir2N, dir1N.clone().cross(dir2N));
    this.matrix.copy(matrixL);

    const r = fixed ? 0.65 : scale * 0.65;
    const arcPoints = this._calculateArcPoints(r);
    const geometry = new LineGeometry();
    geometry.setPositions(arcPoints);

    const visibleLine = new Line2(
      geometry,
      new LineMaterial({
        color: axisColors[axis],
        linewidth: linewidth,
        transparent: true,
      })
    );
    this.add(visibleLine);

    this._addPointerInteractions({
      geometry,
      linewidth,
      hoveredColor,
      axis,
      onDragStart,
      onDrag,
      onDragEnd,
    });
  }

  _calculateArcPoints(radius) {
    const segments = 32;
    const points = [];
    for (let j = 0; j <= segments; j++) {
      const angle = (j * (Math.PI / 2)) / segments;
      points.push(Math.cos(angle) * radius, Math.sin(angle) * radius, 0);
    }
    return points;
  }

  _addPointerInteractions({ geometry, linewidth, hoveredColor, axis, onDragStart, onDrag, onDragEnd }) {
    const setHover = buildSetHover(this, hoveredColor);
    let clickInfo = undefined;

    // Add invisible mesh
    const invisibleMaterial = new LineMaterial({
      linewidth: linewidth * 4,
      visible: false,
    });
    const mesh = new Line2(geometry, invisibleMaterial);
    // Fix rayhit for invisible line
    registeredInvisibleLineMaterials.add(mesh);
    invisibleMaterial.addEventListener('dispose', () => {
      registeredInvisibleLineMaterials.delete(mesh);
    });
    this.add(mesh);

    mesh.addEventListener('pointerdown', (e) => {
      e.stopPropagation();

      const clickPoint = e.point.clone();
      const origin = new THREE.Vector3().setFromMatrixPosition(this.matrixWorld);
      const e1 = new THREE.Vector3().setFromMatrixColumn(this.matrixWorld, 0).normalize();
      const e2 = new THREE.Vector3().setFromMatrixColumn(this.matrixWorld, 1).normalize();
      const normal = new THREE.Vector3().setFromMatrixColumn(this.matrixWorld, 2).normalize();
      const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, origin);

      clickInfo = { clickPoint, origin, e1, e2, normal, plane };
      onDragStart({ component: 'Rotator', axis, origin, directions: [e1, e2, normal] });

      e.setPointerCapture(e.pointerId);
    });

    mesh.addEventListener('pointermove', (e) => {
      e.stopPropagation();
      setHover(true);

      if (!clickInfo) return;
      const { clickPoint, origin, e1, e2, normal, plane } = clickInfo;

      const ray = new THREE.Ray();
      const intersection = new THREE.Vector3();
      ray.copy(e.ray).intersectPlane(plane, intersection);

      let deltaAngle = this._calculateAngle(clickPoint, intersection, origin, e1, e2);
      let degrees = (deltaAngle * 180) / Math.PI;

      if (e.shiftKey) {
          degrees = Math.round(degrees / 10) * 10;
          deltaAngle = (degrees * Math.PI) / 180;
      }

      const rotMatrix = new THREE.Matrix4().makeRotationAxis(normal, deltaAngle);
      const posNew = new THREE.Vector3().copy(origin).applyMatrix4(rotMatrix).sub(origin).negate();
      rotMatrix.setPosition(posNew);

      onDrag(rotMatrix);
    });

    mesh.addEventListener('pointerup', (e) => {
      e.stopPropagation();
      clickInfo = undefined;
      onDragEnd();
      e.releasePointerCapture(e.pointerId);
    });

    mesh.addEventListener('pointerout', () => setHover(false));
  }

  _calculateAngle(clickPoint, intersectionPoint, origin, e1, e2) {
    const clickDir = clickPoint.clone().sub(origin);
    const intersectionDir = intersectionPoint.clone().sub(origin);

    const dote1e1 = e1.dot(e1);
    const dote2e2 = e2.dot(e2);

    const uClick = clickDir.dot(e1) / dote1e1;
    const vClick = clickDir.dot(e2) / dote2e2;

    const uIntersection = intersectionDir.dot(e1) / dote1e1;
    const vIntersection = intersectionDir.dot(e2) / dote2e2;

    const angleClick = Math.atan2(vClick, uClick);
    const angleIntersection = Math.atan2(vIntersection, uIntersection);

    return angleIntersection - angleClick;
  }
}


class PlaneSlider extends THREE.Group {
  constructor({
    axis,
    linewidth,
    scale,
    fixed,
    axisColors,
    hoveredColor,
    addPlaneLines = false,
    onDragStart,
    onDrag,
    onDragEnd,
  }) {
    super();
    this.matrixAutoUpdate = false;

    // Calculate basis matrix
    const dir1N = getAxisVector((axis + 1) % 3);
    const dir2N = getAxisVector((axis + 2) % 3);
    const matrixL = new THREE.Matrix4().makeBasis(dir1N, dir2N, dir1N.clone().cross(dir2N));
    this.matrix.copy(matrixL);

    const pos1 = fixed ? 1 / 7 : scale / 7;
    const length = fixed ? 0.225 : scale * 0.225;
    const color = axisColors[axis];

    const planeMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(),
      new THREE.MeshBasicMaterial({
        transparent: true,
        color: color,
        polygonOffset: true,
        polygonOffsetFactor: -10,
        side: THREE.DoubleSide,
        fog: false,
      })
    );
    planeMesh.scale.set(length, length, 1);
    planeMesh.position.set(pos1 * 1.7, pos1 * 1.7, 0);
    this.add(planeMesh);

    if (addPlaneLines) {
      const lineGeometry = new LineGeometry();
      lineGeometry.setPositions([0, 0, 0, 0, length, 0, length, length, 0, length, 0, 0, 0, 0, 0]);
      const line = new Line2(
        lineGeometry,
        new LineMaterial({
          transparent: true,
          color: color,
          linewidth: linewidth,
          polygonOffset: true,
          polygonOffsetFactor: -10,
          fog: false,
        })
      );
      line.position.set(-length / 2, -length / 2, 0);
      this.add(line);
    }

    this._addPointerInteractions({
      mesh: planeMesh,
      hoveredColor,
      axis,
      onDragStart,
      onDrag,
      onDragEnd,
    });
  }

  _addPointerInteractions({ mesh, hoveredColor, axis, onDragStart, onDrag, onDragEnd }) {
    const setHover = buildSetHover(this, hoveredColor);
    let clickInfo = undefined;

    mesh.addEventListener('pointerdown', (e) => {
      e.stopPropagation();
  
      const clickPoint = e.point.clone();
      const origin = new THREE.Vector3().setFromMatrixPosition(this.matrixWorld);
      const e1 = new THREE.Vector3().setFromMatrixColumn(this.matrixWorld, 0).normalize();
      const e2 = new THREE.Vector3().setFromMatrixColumn(this.matrixWorld, 1).normalize();
      const normal = new THREE.Vector3().setFromMatrixColumn(this.matrixWorld, 2).normalize();
      const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, origin);
  
      clickInfo = { clickPoint, e1, e2, plane };
  
      onDragStart({ component: 'Slider', axis: axis, origin, directions: [e1, e2, normal] });
  
      e.setPointerCapture(e.pointerId);
    });

    mesh.addEventListener('pointermove', (e) => {
      e.stopPropagation();
      setHover(true);
      if (!clickInfo) return;
      const { clickPoint, e1, e2, plane } = clickInfo;

      const ray = new THREE.Ray();
      const intersection = new THREE.Vector3();
      ray.copy(e.ray).intersectPlane(plane, intersection);

      intersection.sub(clickPoint);
      let [offsetX, offsetY] = this._decomposeIntoBasis(e1, e2, intersection);

      const offsetMatrix = new THREE.Matrix4().makeTranslation(
        offsetX * e1.x + offsetY * e2.x,
        offsetX * e1.y + offsetY * e2.y,
        offsetX * e1.z + offsetY * e2.z
      );
      onDrag(offsetMatrix);
    });

    mesh.addEventListener('pointerup', (e) => {
      e.stopPropagation();
      clickInfo = undefined;
      onDragEnd();
      e.releasePointerCapture(e.pointerId);
    });

    mesh.addEventListener('pointerout', () => setHover(false));
  }

  _decomposeIntoBasis(e1, e2, offset) {
    const i1 =
      Math.abs(e1.x) >= Math.abs(e1.y) && Math.abs(e1.x) >= Math.abs(e1.z)
        ? 0
        : Math.abs(e1.y) >= Math.abs(e1.x) && Math.abs(e1.y) >= Math.abs(e1.z)
        ? 1
        : 2;

    const e2DegrowthOrder = [0, 1, 2].sort((a, b) => Math.abs(e2.getComponent(b)) - Math.abs(e2.getComponent(a)));
    const i2 = i1 === e2DegrowthOrder[0] ? e2DegrowthOrder[1] : e2DegrowthOrder[0];
    const a1 = e1.getComponent(i1);
    const a2 = e1.getComponent(i2);
    const b1 = e2.getComponent(i1);
    const b2 = e2.getComponent(i2);
    const c1 = offset.getComponent(i1);
    const c2 = offset.getComponent(i2);

    const y = (c2 - c1 * (a2 / a1)) / (b2 - b1 * (a2 / a1));
    const x = (c1 - y * b1) / a1;

    return [x, y];
  }
}


const mL0 = /* @__PURE__ */ new THREE.Matrix4()
const mW0 = /* @__PURE__ */ new THREE.Matrix4()
const mP = /* @__PURE__ */ new THREE.Matrix4()
const mPInv = /* @__PURE__ */ new THREE.Matrix4()
const mW = /* @__PURE__ */ new THREE.Matrix4()
const mL = /* @__PURE__ */ new THREE.Matrix4()
const mL0Inv = /* @__PURE__ */ new THREE.Matrix4()
const mdL = /* @__PURE__ */ new THREE.Matrix4()
const mG = /* @__PURE__ */ new THREE.Matrix4()


export class PivotControls extends THREE.Group {
  constructor(options) {
    const {
      matrix,
      autoTransform = true,
      disableAxes = false,
      disableSliders = false,
      disableRotations = false,
      scale = 1,
      linewidth = 4,
      fixed = false,
      axisColors = ['#ff2060', '#20df80', '#2080ff'],
      hoveredColor = '#ffff40',
      onDragStart,
      onDrag,
      onDragEnd,
    } = options;
    super();

    this.ref = new THREE.Group();
    if (matrix)
      this.ref.matrix.copy(matrix);
    this.ref.matrixAutoUpdate = false;
    this.gizmoRef = new THREE.Group();

    const objectProps = {
      scale,
      linewidth,
      fixed,
      axisColors,
      hoveredColor,
      onDragStart: (props) => {
        mL0.copy(this.ref.matrix)
        mW0.copy(this.ref.matrixWorld)
        this.dispatchEvent({ type: 'dragstart' });
      },
      onDrag: (mdW) => {
        mP.copy(this.matrixWorld)
        mPInv.copy(mP).invert()
        // After applying the delta
        mW.copy(mW0).premultiply(mdW)
        mL.copy(mW).premultiply(mPInv)
        mL0Inv.copy(mL0).invert()
        mdL.copy(mL).multiply(mL0Inv)
        if (autoTransform) {
          this.ref.matrix.copy(mL)
        }
        this.dispatchEvent({ 
          type: 'drag',
          matrix: mL,
        });
      },
      onDragEnd: () => {
        this.dispatchEvent({ type: 'dragend' });
      },
    };

    // Add Axis Arrows
    if (!disableAxes) {
      this.gizmoRef.add(new AxisArrow({...objectProps, axis: 0}));
      this.gizmoRef.add(new AxisArrow({...objectProps, axis: 1}));
      this.gizmoRef.add(new AxisArrow({...objectProps, axis: 2}));
    }

    // Add Plane Sliders
    if (!disableSliders) {
      this.gizmoRef.add(new PlaneSlider({...objectProps, axis: 0}));
      this.gizmoRef.add(new PlaneSlider({...objectProps, axis: 1}));
      this.gizmoRef.add(new PlaneSlider({...objectProps, axis: 2}));
    }

    // Add Axis Rotators
    if (!disableRotations) {
      this.gizmoRef.add(new AxisRotator({...objectProps, axis: 0}));
      this.gizmoRef.add(new AxisRotator({...objectProps, axis: 1}));
      this.gizmoRef.add(new AxisRotator({...objectProps, axis: 2}));
    }

    this.add(this.ref);
    this.ref.add(this.gizmoRef);
  }

  setMatrix(matrix) {
    this.ref.matrix.copy(matrix);
    this.ref.updateWorldMatrix(true, true);
  }

  dispose() {
    // Dispose all children and their materials
    this.traverse(child => {
      if (child === this) return;
      if (child.dispose) child.dispose();
      if (child.material) child.material.dispose();
      if (child.geometry) child.geometry.dispose();
    });
  }
}


export class MouseInteractions {
  constructor(renderer, camera, scene, element) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera
    this.enabled = true;

    this._raycaster = new THREE.Raycaster();
    this._pointer = new THREE.Vector2();
    this._currentlyIntersected = [];
    this._currentEvent = undefined;
    element = element || renderer.domElement;

    element.addEventListener('pointermove', this._onPointerMove.bind(this));
    element.addEventListener('pointerdown', this._onPointerDown.bind(this));
    element.addEventListener('pointerup', this._onPointerUp.bind(this));

    this._captureTarget = document.createElement('div');
    this._captureTarget.addEventListener('pointermove', this._onCapturePointerMove.bind(this));
    this._captureTarget.addEventListener('pointerup', this._onPointerUp.bind(this));
    this._captureTarget.addEventListener('pointerdown', this._onPointerDown.bind(this));
    this._capturedPointerIds = new Set();
    document.body.appendChild(this._captureTarget);
  }

  isCaptured() {
    return this._capturedPointerIds.size > 0;
  }

  _onCapturePointerMove(event) {
    if (!this.enabled) return;
    event.preventDefault();
    this._onPointerMove(event);
  }

  _capturePointer(pointerId) {
    this._captureTarget.setPointerCapture(pointerId);
    this._capturedPointerIds.add(pointerId);
  }

  _releasePointer(pointerId) {
    this._captureTarget.releasePointerCapture(pointerId);
    this._capturedPointerIds.delete(pointerId);
  }

  _dispatchIntersected({ type, pointerId, ray }, objects) {
    objects = objects || this._currentlyIntersected;
    let propagate = true;
    for (let obj of objects) {
      const [object, intersect] = obj;
      object.dispatchEvent({ 
        ...intersect,
        type: type,
        setPointerCapture: this._capturePointer.bind(this),
        releasePointerCapture: this._releasePointer.bind(this),
        pointerId: pointerId,
        pointer: this._pointer,
        ray: ray,
        stopPropagation: () => { propagate = false },
      });
      if (!propagate) break;
    }
    return propagate;
  }

  _onPointerMove(event) {
    if (!this.enabled) return;
    const x = event.clientX - this.renderer.domElement.getBoundingClientRect().left;
    const y = event.clientY - this.renderer.domElement.getBoundingClientRect().top;
    const width = this.renderer.domElement.width;
    const height = this.renderer.domElement.height;
    this._pointer.x = ( x / width ) * 2 - 1;
    this._pointer.y = - ( y / height ) * 2 + 1;
    this._currentEvent = event;
  }

  _onPointerUp(event) {
    if (!this.enabled) return;
    if (event._handled) return;
    if (!this._dispatchIntersected(event))
      event.preventDefault();
    if (!this._capturedPointerIds.has(event.pointerId)) {
      const pointerEvent = new PointerEvent('pointerup', event);
      pointerEvent._handled = true;
      this.renderer.domElement.dispatchEvent(pointerEvent);
    }
  }

  _onPointerDown(event) {
    if (!this.enabled) return;
    if (!this._dispatchIntersected(event))
      event.preventDefault();
  }

  update() {
    if (!this.enabled) return;
    const event = this._currentEvent;
    if (!event) return;

    for (let invisibleMaterial of registeredInvisibleLineMaterials) {
      invisibleMaterial.onBeforeRender(this.renderer);
    }

    this._raycaster.setFromCamera(this._pointer, this.camera);
    if (!this._capturedPointerIds.has(event.pointerId)) {
      const objectsToIntersect = [];
      const addObjects = (obj) => {
        if (obj.type === "Points") return;  // Skip Points3D
        obj.children.forEach(addObjects);
        objectsToIntersect.push(obj);
      };
      objectsToIntersect.pop(); // Remove scene
      addObjects(this.scene);
      const pointEvents = ['pointerdown', 'pointermove', 'pointerup', 'pointerout'];
      const intersectSet = new Set();
      const intersects = this._raycaster.intersectObjects(objectsToIntersect, false);
      const newIntersected = [];
      const newIntersectedSet = new Set();
      let propagate = true;
      for (let intersect of intersects) {
        let obj = intersect.object;
        let visible = true;
        while (obj) {
          visible = visible && obj.visible;
          if (!visible) break;
          obj = obj.parent;
        }
        if (!visible) continue;
        obj = intersect.object;
        while (obj && obj !== this.scene) {
          if (!newIntersectedSet.has(obj) && 
              obj._listeners && pointEvents.some(type => obj._listeners[type] && 
              obj._listeners[type].length > 0)) {
            obj.dispatchEvent({ 
              ...intersect,
              type: 'pointermove',
              setPointerCapture: this._capturePointer.bind(this),
              releasePointerCapture: this._releasePointer.bind(this),
              pointerId: event.pointerId,
              point: this._pointer,
              ray: this._raycaster.ray,
              stopPropagation: () => { propagate = false },
            });
            newIntersected.push([obj, intersect]);
            newIntersectedSet.add(obj);
            if (!propagate) break;
          }
          obj = obj.parent;
        }
        if (!propagate) break;
      }

      // Dispatch pointer out
      const removed = this._currentlyIntersected.filter(obj => !newIntersectedSet.has(obj[0]));
      this._dispatchIntersected({ type: 'pointerout', pointerId: event.pointerId }, removed);
      this._currentlyIntersected = newIntersected;
    } else {
      event.ray = this._raycaster.ray;
      this._dispatchIntersected(event, this._currentlyIntersected);
    }
    this._currentEvent = undefined;
  }
}


export class CameraFrustum extends THREE.Group {
  constructor({
    fov = 75,
    linewidth = 3,
    aspect = 1,
    scale = 1,
    color,
    position,
    quaternion,
    hoveredColor = '#ffff40',
    interactive = false,
    originSphereScale,
  }) {
    super();
    this._scale = scale;
    if (position) this.position.copy(position);
    if (quaternion) this.quaternion.copy(quaternion);

    this._hasImage = false;
    this._fov = fov;
    this._aspect = aspect;
    this._interactive = interactive;
    this._pointerdown = false;
    this._focused = false;
    this._hoveredColor = hoveredColor;
    this._color = new THREE.Color(color);

    Object.defineProperty(this, 'color', {
      get: () => this._color,
      set: (value) => {
        value = new THREE.Color(value);
        if (value === this._color) return;
        this._color = value;
        this._updateColor();
      },
    });

    // Define fov property
    Object.defineProperty(this, 'fov', {
      get: () => this._fov,
      set: (value) => {
        if (value === this._fov) return;
        this._fov = value;
        this._updateGeometry(scale);
      },
    });

    Object.defineProperty(this, 'aspect', {
      get: () => this._aspect,
      set: (value) => {
        if (value === this._aspect) return;
        this._aspect = value;
        this._updateGeometry(scale);
      },
    });

    Object.defineProperty(this, 'focused', {
      get: () => this._focused,
      set: (value) => {
        if (value === this._focused) return;
        this._focused = value;
        this._hover = false;
        this.originSphere && (this.originSphere.visible = !value);
        this._updateColor();
      },
    });

    this.geometry = new LineSegmentsGeometry();
    this._updateGeometry(scale);
    this.material = new LineMaterial({
      color: this.color,
      linewidth,
    });

    // Attach material to geometry.
    const lineSegment = new LineSegments2(this.geometry, this.material);
    this.add(lineSegment);

    if (originSphereScale) {
      const ballGeometry = new THREE.SphereGeometry(originSphereScale*scale, 32, 32);
      this.originSphereMaterial = new THREE.MeshBasicMaterial({ color: this.color });
      this.originSphere = new THREE.Mesh(ballGeometry, this.originSphereMaterial);
      this.originSphere.visible = !this._focused;
      this.add(this.originSphere);
    }

    if (this._interactive) {
      // Add invisible mesh for raycasting.
      this.invisibleMaterial = new LineMaterial({
        linewidth: linewidth * 4,
        visible: false,
      });
      const invisibleMesh = new LineSegments2(this.geometry, this.invisibleMaterial);
      invisibleMesh.addEventListener('pointermove', this._onPointerMove.bind(this));
      invisibleMesh.addEventListener('pointerout', this._onPointerOut.bind(this));
      invisibleMesh.addEventListener('pointerup', this._onPointerUp.bind(this));
      invisibleMesh.addEventListener('pointerdown', this._onPointerDown.bind(this));
      registeredInvisibleLineMaterials.add(invisibleMesh);
      this.add(invisibleMesh);
    }
  }

  _updateColor() {
    if (this._hover || this._focused) {
      this.material.color.set(this._hoveredColor);
      this.originSphereMaterial?.color.set(this._hoveredColor);
    } else {
      this.material.color.set(this._color);
      this.originSphereMaterial?.color.set(this._color);
    }
  }

  _onPointerDown(e) {
    if (this._focused) return;
    e.stopPropagation();
    this._pointerdown = true;
  }

  _onPointerUp(e) {
    if (this._focused) return;
    e.stopPropagation();
    if (this._pointerdown) {
      this.dispatchEvent({ type: 'click' });
    }
  }

  _onPointerMove(e) {
    if (this._focused) return;
    e.stopPropagation();
    this._hover = true;
    this._updateColor();
  }

  _onPointerOut(e) {
    if (this._focused) return;
    e.stopPropagation();
    this._pointerdown = false;
    this._hover = false;
    this._updateColor();
  }

  _computeXyz() {
    const fovRad = THREE.MathUtils.degToRad(this.fov);
    let y = Math.tan(fovRad / 2.0);
    let x = y * this.aspect;
    let z = 1.0;

    const volumeScale = Math.cbrt((x * y * z) / 3.0);
    x /= volumeScale;
    y /= volumeScale;
    z /= volumeScale;
    x *= this._scale;
    y *= this._scale;
    z *= this._scale;
    return [x, y, z];
  }

  _updateGeometry() {
    const [x, y, z] = this._computeXyz();
    const points = [
      // Rectangle.
      [-1, -1, 1],
      [1, -1, 1],
      [1, -1, 1],
      [1, 1, 1],
      [1, 1, 1],
      [-1, 1, 1],
      [-1, 1, 1],
      [-1, -1, 1],
      // Lines to origin.
      [-1, -1, 1],
      [0, 0, 0],
      [0, 0, 0],
      [1, -1, 1],
      // Lines to origin.
      [-1, 1, 1],
      [0, 0, 0],
      [0, 0, 0],
      [1, 1, 1],
      // Up direction indicator.
      // Don't overlap with the image if the image is present.
      [0.0, -1.2, 1.0],
      this._hasImage ? [0.0, -1.0, 1.0] : [0.0, -0.9, 1.0],
    ].map((xyz) => [xyz[0] * x, xyz[1] * y, xyz[2] * z]);
    this.geometry.setPositions(points.flat())
  }

  setImageTexture(texture) {
    if (this._hasImage)
      throw new Error('Image texture is already set.');
    this._hasImage = true;
    this._updateGeometry();
    const [x, y, z] = this._computeXyz();
    const imageGeometry = new THREE.PlaneGeometry(this.aspect * y * 2, y * 2);
    const imageMaterial = new THREE.MeshBasicMaterial({ 
      transparent: true,
      side: THREE.DoubleSide,
      toneMapped: false,
      map: texture,
    });
    imageMaterial.needsUpdate = true;
    const mesh = new THREE.Mesh(imageGeometry, imageMaterial)
    mesh.position.set(0.0, 0.0, z * 0.999999);
    mesh.rotation.set(Math.PI, 0.0, 0.0);
    this.add(mesh);

    if (this._interactive) {
      mesh.addEventListener('pointermove', this._onPointerMove.bind(this));
      mesh.addEventListener('pointerout', this._onPointerOut.bind(this));
      mesh.addEventListener('pointerup', this._onPointerUp.bind(this));
      mesh.addEventListener('pointerdown', this._onPointerDown.bind(this));
    }
  }

  dispose() {
    // Dispose all children and their materials
    this.traverse(child => {
      if (child === this) return;
      if (child.dispose) child.dispose();
      if (child.material) child.material.dispose();
      if (child.geometry) child.geometry.dispose();
    });
  }
}

export class TrajectoryCurve extends THREE.Group {
  constructor({
    positions,
    color,
    linewidth = 4,
  }) {
    super()
    this.geometry = undefined;
    this.lineSegments = undefined;
    this.material = new LineMaterial({ color, linewidth });
    this.setPositions(positions);
    this._color = new THREE.Color(color);

    Object.defineProperty(this, 'color', {
      get: () => this._color,
      set: (value) => {
        value = new THREE.Color(value);
        if (value === this._color) return;
        this._color = value;
        this.material.color = value;
      },
    });
  }

  setPositions(points) {
    if (points.length < 2) return;
    const segment_points = [];
    for (let i = 0; i < points.length; i++) {
      segment_points.push(points[i].x, points[i].y, points[i].z);
      segment_points.push(points[i].x, points[i].y, points[i].z);
    }
    segment_points.splice(0, 3);
    segment_points.splice(segment_points.length - 3, 3);
    if (this.lineSegments) {
      this.remove(this.lineSegments);
      this.lineSegments = undefined;
      this.geometry.dispose();
    }
    this.geometry = new LineSegmentsGeometry();
    this.geometry.setPositions(segment_points);
    this.lineSegments = new LineSegments2(this.geometry, this.material);
    this.add(this.lineSegments);
  }

  dispose() {
    this.geometry.dispose();
    this.material.dispose();
  }
}
