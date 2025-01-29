import {
  Euler,
	Controls,
	MOUSE,
	Quaternion,
	Spherical,
	TOUCH,
	Vector2,
	Vector3,
	Plane,
	Ray,
	MathUtils
} from 'three';

// OrbitControls performs orbiting, dollying (zooming), and panning.
// Unlike TrackballControls, it maintains the "up" direction object.up (+Y by default).
//
//    Orbit - left mouse / touch: one-finger move
//    Zoom - middle mouse, or mousewheel / touch: two-finger spread or squish
//    Pan - right mouse, or left mouse + ctrl/meta/shiftKey, or arrow keys / touch: two-finger move

const _changeEvent = { type: 'change' };
const _startEvent = { type: 'start' };
const _endEvent = { type: 'end' };
const _ray = new Ray();
const _plane = new Plane();
const _TILT_LIMIT = Math.cos( 70 * MathUtils.DEG2RAD );

const _v = new Vector3();
const _q = new Quaternion();
const _quatInv = new Quaternion();
const _euler = new Euler();
const _twoPI = 2 * Math.PI;

const _STATE = {
	NONE: - 1,
	ROTATE: 0,
	DOLLY: 1,
	PAN: 2,
	TOUCH_ROTATE: 3,
	TOUCH_PAN: 4,
	TOUCH_DOLLY_PAN: 5,
	TOUCH_DOLLY_ROTATE: 6
};
const _EPS = 0.000001;


class KeyboardManager {
  constructor(element) {
    this._activeKeys = {};
    window.addEventListener('keydown', this._onKeyDown.bind(this));
    window.addEventListener('keyup', this._onKeyUp.bind(this), true);
    this._element = element;
    this._element.addEventListener('pointerover', this._onPointerOver.bind(this), true);
    this._element.addEventListener('pointerout', this._onPointerOut.bind(this), true);
    this._pointerActivated = false;
    this._activePointer = new Set();
    this._handledKeys = new Set([
      'ShiftLeft', 'ShiftRight',
      'Space',
      'AltLeft', 'AltRight',
      'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
      'KeyW', 'KeyA', 'KeyS', 'KeyD',
      'KeyQ', 'KeyE',
      'KeyZ', 'KeyX',
    ]);
  }

  _onPointerOver(event) {
    this._pointerActivated = true;
    this._activePointer.add(event.pointerId);
  }

  _onPointerOut(event) {
    this._activePointer.delete(event.pointerId);
    if (this._activePointer.size <= 0) {
      this._activeKeys = {};
    }
  }

  _onKeyDown(event) {
    if (this._pointerActivated && this._activePointer.size <= 0) return;
    if (this._handledKeys.has(event.code)) {
      event.preventDefault();
      event.stopPropagation();
    }
    this._element.focus();
    this._activeKeys[event.code] = true;
    event.shiftKey ? this._activeKeys["Shift"] = true : delete this._activeKeys["Shift"];
    event.ctrlKey ? this._activeKeys["Ctrl"] = true : delete this._activeKeys["Ctrl"];
    event.altKey ? this._activeKeys["Alt"] = true : delete this._activeKeys["Alt"];
    event.metaKey ? this._activeKeys["Meta"] = true : delete this._activeKeys["Meta"];
  }

  _onKeyUp(event) {
    delete this._activeKeys[event.code];
    event.shiftKey ? this._activeKeys["Shift"] = true : delete this._activeKeys["Shift"];
    event.ctrlKey ? this._activeKeys["Ctrl"] = true : delete this._activeKeys["Ctrl"];
    event.altKey ? this._activeKeys["Alt"] = true : delete this._activeKeys["Alt"];
    event.metaKey ? this._activeKeys["Meta"] = true : delete this._activeKeys["Meta"];
  }

  dispose() {
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
    this._element.removeEventListener('pointerover', this._onPointerOver);
    this._element.removeEventListener('pointerout', this._onPointerOut);
  }

  isPressed(key) {
    return this._activeKeys[key];
  }
}

class ViewerControls extends Controls {
	constructor(object, domElement = null) {
		super(object, domElement);
		this.state = _STATE.NONE;
    this.mode = "fps";

		// Set to false to disable this control
		this.enabled = true;

		// "target" sets the location of focus, where the object orbits around
		this.target = new Vector3();

		// Sets the 3D cursor (similar to Blender), from which the maxTargetRadius takes effect
		this.cursor = new Vector3();

		// How far you can dolly in and out ( PerspectiveCamera only )
		this.minDistance = 0;
		this.maxDistance = Infinity;

		// How far you can zoom in and out ( OrthographicCamera only )
		this.minZoom = 0;
		this.maxZoom = Infinity;

		// Limit camera target within a spherical area around the cursor
		this.minTargetRadius = 0;
		this.maxTargetRadius = Infinity;

		// Set to true to enable damping (inertia)
		// If damping is enabled, you must call controls.update() in your animation loop
		this.enableDamping = false;
		this.dampingFactor = 0.3;

		// This option actually enables dollying in and out; left as "zoom" for backwards compatibility.
		// Set to false to disable zooming
		this.enableZoom = true;
		this.zoomSpeed = 1.0;

		// Set to false to disable rotating
		this.enableRotate = true;
		this.rotateSpeed = 1.0;

		// Set to false to disable panning
		this.enablePan = true;
		this.panSpeed = 1.0;
		this.keyPanSpeed = 14.0;	// pixels moved per arrow key push
		this.keyRotateSpeed = 7.0;

		// Set to true to automatically rotate around the target
		// If auto-rotate is enabled, you must call controls.update() in your animation loop
		this.autoRotate = false;
		this.autoRotateSpeed = 2.0; // 30 seconds per orbit when fps is 60

		// Mouse buttons
		this.mouseButtons = { LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN };

		// Touch fingers
		this.touches = { ONE: TOUCH.ROTATE, TWO: TOUCH.DOLLY_PAN };

		// for reset
		this.target0 = this.target.clone();
		this.position0 = this.object.position.clone();
		this.zoom0 = this.object.zoom;

		// internals
		this._lastPosition = new Vector3();
		this._lastQuaternion = new Quaternion();
		this._lastTargetPosition = new Vector3();

		// so camera.up is the orbit axis
		this._quat = new Quaternion().setFromUnitVectors(object.up, new Vector3(0, 1, 0));
		this._quatInverse = this._quat.clone().invert();

		// current position in spherical coordinates
		this._spherical = new Spherical();
		this._sphericalDelta = new Spherical();
		this._deltaZ = 0;

		this._scale = 1;
		this._panOffset = new Vector3();
    
		this._rotateStart = new Vector2();
		this._rotateEnd = new Vector2();
		this._rotateDelta = new Vector2();

		this._panStart = new Vector2();
		this._panEnd = new Vector2();
		this._panDelta = new Vector2();

		this._dollyStart = new Vector2();
		this._dollyEnd = new Vector2();
		this._dollyDelta = new Vector2();

		this._dollyDirection = new Vector3();
		this._mouse = new Vector2();

		this._pointers = [];
		this._pointerPositions = {};

		this._controlActive = false;

		// event listeners
		this._onPointerMove = onPointerMove.bind(this);
		this._onPointerDown = onPointerDown.bind(this);
		this._onPointerUp = onPointerUp.bind(this);
		this._onContextMenu = onContextMenu.bind(this);
		this._onMouseWheel = onMouseWheel.bind(this);

		this._onTouchStart = onTouchStart.bind(this);
		this._onTouchMove = onTouchMove.bind(this);

		this._onMouseDown = onMouseDown.bind(this);
		this._onMouseMove = onMouseMove.bind(this);

		this._interceptControlDown = interceptControlDown.bind(this);
		this._interceptControlUp = interceptControlUp.bind(this);

		if ( this.domElement !== null ) {
			this.connect();
		}

		this.update();
	}

  updateUp() {
		// so camera.up is the orbit axis
		this._quat = new Quaternion().setFromUnitVectors(this.object.up, new Vector3(0, 1, 0));
		this._quatInverse = this._quat.clone().invert();
  }

	connect() {
		this.domElement.addEventListener('pointerdown', this._onPointerDown);
		this.domElement.addEventListener('pointercancel', this._onPointerUp);
		this.domElement.addEventListener('contextmenu', this._onContextMenu);
		this.domElement.addEventListener('wheel', this._onMouseWheel, { passive: false });

		const document = this.domElement.getRootNode(); // offscreen canvas compatibility
		document.addEventListener('keydown', this._interceptControlDown, { passive: true, capture: true });

    this.domElement.setAttribute('tabindex', 10000);
    this.domElement.style.outline = 'none';
    this._keyboardManager = new KeyboardManager(this.domElement);

		this.domElement.style.touchAction = 'none'; // disable touch scroll
	}

	disconnect() {
		this.domElement.removeEventListener('pointerdown', this._onPointerDown);
		this.domElement.removeEventListener('pointermove', this._onPointerMove);
		this.domElement.removeEventListener('pointerup', this._onPointerUp);
		this.domElement.removeEventListener('pointercancel', this._onPointerUp);
		this.domElement.removeEventListener('wheel', this._onMouseWheel);
		this.domElement.removeEventListener('contextmenu', this._onContextMenu);

    if (this._keyboardManager) {
      this._keyboardManager.dispose();
      this._keyboardManager = undefined;
    }

		const document = this.domElement.getRootNode(); // offscreen canvas compatibility
		document.removeEventListener('keydown', this._interceptControlDown, { capture: true });

		this.domElement.style.touchAction = 'auto';
	}

	dispose() {
		this.disconnect();
	}

	saveState() {
		this.target0.copy( this.target );
		this.position0.copy( this.object.position );
		this.zoom0 = this.object.zoom;
	}

  resetDamping() {
    this._sphericalDelta.set(0, 0, 0);
    this._panOffset.set(0, 0, 0);
  }

	reset() {
		this.target.copy( this.target0 );
		this.object.position.copy( this.position0 );
		this.object.zoom = this.zoom0;
    this.object.updateWorldMatrix(false, false);
		this.object.updateProjectionMatrix();
		this.dispatchEvent( _changeEvent );

		this.update();
		this.state = _STATE.NONE;
	}

	update(deltaTime = undefined) {
    if (deltaTime !== undefined)
      deltaTime = Math.min(deltaTime, 1000);
    const m = -(this.enableDamping ? this.dampingFactor : 1.0);

    // Update pressed keys
    this._updateKeys(deltaTime);

    // Rotate the camera around Z axis
    if (Math.abs(this._deltaZ) > 1e-6) {
      _v.set(0, 0, 1).applyQuaternion(this.object.quaternion);
      _quatInv.setFromAxisAngle(_v, -this._deltaZ * m * 0.4);
      this.object.quaternion.premultiply(_quatInv);

      // Rotate the up vector around lookAt
      if (this.mode === "orbit" || this.mode === "fps") {
        _v.crossVectors(this.object.up, _v).normalize();
        _v.crossVectors(_v, this.object.up).normalize();
        this.object.up.applyQuaternion(_quatInv);

        // Update quat
        this._quat = new Quaternion().setFromUnitVectors(this.object.up, new Vector3(0, 1, 0));
        this._quatInverse = this._quat.clone().invert();
      }
      if (this.enableDamping) {
        this._deltaZ *= (1 - this.dampingFactor);
      } else {
        this._deltaZ = 0;
      }
    }

    if (this.mode === "fps") {
      // 1. Apply pan
      this.object.position.addScaledVector(this._panOffset, m);

      // 2. Apply the rotation
      _q.copy(this.object.quaternion);
      _q.premultiply(this._quat);
      _quatInv.copy(this._quat).invert();

      _euler.setFromQuaternion(_q, 'YXZ');
      _euler.x -= this._sphericalDelta.phi * m;
      _euler.y -= this._sphericalDelta.theta * m;
      _euler.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, _euler.x));
      _q.setFromEuler(_euler);
      _q.premultiply(_quatInv);
      this.object.quaternion.copy(_q).normalize();

      if (this.enableDamping) {
        this._sphericalDelta.theta *= (1 - this.dampingFactor);
        this._sphericalDelta.phi *= (1 - this.dampingFactor);
        this._panOffset.multiplyScalar(1 - this.dampingFactor);
      } else {
        // Move the camera
        this._sphericalDelta.set(0, 0, 0);
        this._panOffset.set(0, 0, 0);
      }
      this.object.updateWorldMatrix(false, false);
      return
    }

		const position = this.object.position;
		_v.copy( position ).sub( this.target );
		// rotate offset to "y-axis-is-up" space
		_v.applyQuaternion( this._quat );
		// angle from z-axis around y-axis
		this._spherical.setFromVector3( _v );
		if ( this.enableDamping ) {
			this._spherical.theta += this._sphericalDelta.theta * this.dampingFactor;
			this._spherical.phi += this._sphericalDelta.phi * this.dampingFactor;
		} else {
			this._spherical.theta += this._sphericalDelta.theta;
			this._spherical.phi += this._sphericalDelta.phi;
		}
		this._spherical.makeSafe();


		// move target to panned location
		if (this.enableDamping === true) {
			this.target.addScaledVector(this._panOffset, -this.dampingFactor);
		} else {
			this.target.addScaledVector(this._panOffset, -1);
		}

		// Limit the target distance from the cursor to create a sphere around the center of interest
		this.target.sub(this.cursor);
		this.target.clampLength(this.minTargetRadius, this.maxTargetRadius);
		this.target.add(this.cursor);

		let zoomChanged = false;
		// adjust the camera position based on zoom only if we're not zooming to the cursor or if it's an ortho camera
		// we adjust zoom later in these cases
    const prevRadius = this._spherical.radius;
    this._spherical.radius = this._clampDistance(this._spherical.radius * this._scale);
    zoomChanged = prevRadius != this._spherical.radius;

		_v.setFromSpherical(this._spherical);
		// rotate offset back to "camera-up-vector-is-up" space
		_v.applyQuaternion( this._quatInverse );
		position.copy(this.target).add(_v);
		this.object.lookAt(this.target);

		if (this.enableDamping === true) {
			this._sphericalDelta.theta *= (1 - this.dampingFactor);
			this._sphericalDelta.phi *= (1 - this.dampingFactor);
			this._panOffset.multiplyScalar(1 - this.dampingFactor);
		} else {
			this._sphericalDelta.set(0, 0, 0);
			this._panOffset.set(0, 0, 0);
		}

		this._scale = 1;

		// update condition is:
		// min(camera displacement, camera rotation in radians)^2 > EPS
		// using small-angle approximation cos(x/2) = 1 - x^2 / 8

		if (zoomChanged ||
			  this._lastPosition.distanceToSquared(this.object.position) > _EPS ||
			  8 * ( 1 - this._lastQuaternion.dot(this.object.quaternion)) > _EPS ||
			  this._lastTargetPosition.distanceToSquared(this.target) > _EPS) {

			this.dispatchEvent(_changeEvent);

			this._lastPosition.copy(this.object.position);
			this._lastQuaternion.copy(this.object.quaternion);
			this._lastTargetPosition.copy(this.target);
      this.object.updateWorldMatrix(false, false);
			return true;
		}
    this.object.updateWorldMatrix(false, false);
		return false;
	}

	_getZoomScale(delta) {
		const normalizedDelta = Math.abs( delta * 0.05);
		return Math.pow(0.95, this.zoomSpeed * normalizedDelta);
	}

	_panAligned(deltaX, deltaY, deltaZ=0) {
		const element = this.domElement;

    let multiplier = 2.0;
    _v.setFromMatrixColumn(this.object.matrix, 0);
    _v.crossVectors(this.object.up, _v).normalize();
    this._panOffset.addScaledVector(_v, -multiplier * deltaY / element.clientHeight);
    _v.crossVectors(this.object.up, _v).normalize();
    this._panOffset.addScaledVector(_v, -multiplier * deltaX / element.clientHeight);
    this._panOffset.addScaledVector(this.object.up, -multiplier * deltaZ / element.clientHeight);
	}

	// deltaX and deltaY are in pixels; right and down are positive
	_pan(deltaX, deltaY, deltaZ=0) {
		const element = this.domElement;
    let multiplier = 2.0;
    if (this.mode === "orbit") {
      _v.copy(this.object.position).sub(this.target);

      // half of the fov is center to top of screen
      multiplier *= _v.length() * Math.tan((this.object.fov / 2) * Math.PI / 180.0);
    }
    _v.set(1, 0, 0).applyQuaternion(this.object.quaternion);
    this._panOffset.addScaledVector(_v, multiplier * deltaX / element.clientHeight);
    _v.set(0, 0, 1).applyQuaternion(this.object.quaternion);
    this._panOffset.addScaledVector(_v, multiplier * deltaY / element.clientHeight);
    _v.set(0, 1, 0).applyQuaternion(this.object.quaternion);
    this._panOffset.addScaledVector(_v, -multiplier * deltaZ / element.clientHeight);
	}

	_clampDistance( dist ) {
		return Math.max( this.minDistance, Math.min( this.maxDistance, dist ) );
	}

	//
	// event callbacks - update the object state
	//

	_handleMouseDownRotate(event) {
		this._rotateStart.set(event.clientX, event.clientY);
	}

	_handleMouseDownDolly(event) {
		this._dollyStart.set( event.clientX, event.clientY);
	}

	_handleMouseDownPan(event) {
		this._panStart.set(event.clientX, event.clientY);
	}

	_handleMouseMoveRotate(event) {
		this._rotateEnd.set(event.clientX, event.clientY);
		this._rotateDelta.subVectors(this._rotateEnd, this._rotateStart).multiplyScalar(this.rotateSpeed);

		const element = this.domElement;
    let m = 1.0;
		this._sphericalDelta.theta -= m*_twoPI * this._rotateDelta.x / element.clientHeight; // yes, height
		this._sphericalDelta.phi -= m*_twoPI * this._rotateDelta.y / element.clientHeight;

		this._rotateStart.copy(this._rotateEnd);

		this.update();
	}

	_handleMouseMoveDolly(event) {
		this._dollyEnd.set(event.clientX, event.clientY);
		this._dollyDelta.subVectors(this._dollyEnd, this._dollyStart);

		if (this._dollyDelta.y > 0) {
      this._scale /= this._getZoomScale( this._dollyDelta.y );
		} else if ( this._dollyDelta.y < 0 ) {
			this._scale *= this._getZoomScale(this._dollyDelta.y);
		}

		this._dollyStart.copy( this._dollyEnd );
		this.update();
	}

	_handleMouseMovePan(event) {
		this._panEnd.set( event.clientX, event.clientY );
		this._panDelta.subVectors(this._panEnd, this._panStart).multiplyScalar(this.panSpeed);

		this._pan(this._panDelta.x, 0, this._panDelta.y);
		this._panStart.copy(this._panEnd);

		this.update();
	}

	_handleMouseWheel(event) {
    if (this.mode === "fps") {
      _v.setFromMatrixColumn(this.object.matrix, 2);
      this._panOffset.addScaledVector(_v, -event.deltaY * 0.01 * this.zoomSpeed);
    } else if (this.mode === "orbit") {
      if ( event.deltaY < 0 ) {
        this._scale *= this._getZoomScale(event.deltaY);
      } else if ( event.deltaY > 0 ) {
        this._scale /= this._getZoomScale( event.deltaY );
      }
    } else {
      console.error(`Unknown mode: ${this.mode}`);
    }
		this.update();
	}

  _updateKeys(deltaTime) {
    if (!this._keyboardManager) return;
    let d = (deltaTime || (1000.0 / 60.0)) * 0.02;
    const isPressed = this._keyboardManager.isPressed.bind(this._keyboardManager);
    let m = 1;
    if (isPressed('Shift')) m *= 4;
    let pan;
    if (isPressed('Alt')) {
      pan = this._pan.bind(this);
    } else {
      pan = this._panAligned.bind(this);
    }
    if (isPressed('KeyW'))
      pan(0, Math.abs(this.keyPanSpeed) * d * m);
    if (isPressed('KeyS'))
      pan(0, -Math.abs(this.keyPanSpeed) * d * m);
    if (isPressed('KeyE'))
      pan(0, 0, Math.abs(this.keyPanSpeed) * d * m);
    if (isPressed('KeyQ'))
      pan(0, 0, -Math.abs(this.keyPanSpeed) * d * m);
    if (isPressed('KeyA'))
      pan(Math.abs(this.keyPanSpeed) * d * m, 0);
    if (isPressed('KeyD'))
      pan(-Math.abs(this.keyPanSpeed * d * m), 0);
    if (isPressed('KeyZ')) {
      // Rotate left
      this._deltaZ += _twoPI * Math.abs(this.keyRotateSpeed) * d * 0.001 * m;
    }
    if (isPressed('KeyX')) {
      // Rotate right
      this._deltaZ -= _twoPI * Math.abs(this.keyRotateSpeed) * d * 0.001 * m;
    }
    let mr = 0.0005 * m;
    if (isPressed('ArrowUp')) {
      this._sphericalDelta.phi += _twoPI * Math.abs(this.keyRotateSpeed) * d * mr;
    } 
    if (isPressed('ArrowDown')) {
      this._sphericalDelta.phi -= _twoPI * Math.abs(this.keyRotateSpeed) * d * mr;
    } 
    if (isPressed('ArrowLeft')) {
      this._sphericalDelta.theta += _twoPI * Math.abs(this.keyRotateSpeed) * d * mr;
    } 
    if (isPressed('ArrowRight')) {
      this._sphericalDelta.theta -= _twoPI * Math.abs(this.keyRotateSpeed) * d * mr;
    }
  }

	_handleTouchStartRotate(event) {
		if ( this._pointers.length === 1 ) {
			this._rotateStart.set(event.pageX, event.pageY);
		} else {
			const position = this._getSecondPointerPosition(event);
			const x = 0.5 * (event.pageX + position.x);
			const y = 0.5 * (event.pageY + position.y);
			this._rotateStart.set(x, y);
		}
	}

	_handleTouchStartPan(event) {
		if (this._pointers.length === 1) {
			this._panStart.set(event.pageX, event.pageY);
		} else {
			const position = this._getSecondPointerPosition(event);
			const x = 0.5 * (event.pageX + position.x);
			const y = 0.5 * (event.pageY + position.y);
			this._panStart.set(x, y);
		}
	}

	_handleTouchStartDolly(event) {
		const position = this._getSecondPointerPosition(event);
		const dx = event.pageX - position.x;
		const dy = event.pageY - position.y;
		const distance = Math.sqrt( dx * dx + dy * dy );
		this._dollyStart.set(0, distance);
	}

	_handleTouchStartDollyPan(event) {
		if (this.enableZoom) this._handleTouchStartDolly(event);
		if (this.enablePan) this._handleTouchStartPan(event);
	}

	_handleTouchStartDollyRotate(event) {
		if (this.enableZoom) this._handleTouchStartDolly(event);
		if (this.enableRotate) this._handleTouchStartRotate(event);
	}

	_handleTouchMoveRotate(event) {
		if ( this._pointers.length == 1 ) {
			this._rotateEnd.set(event.pageX, event.pageY);
		} else {
			const position = this._getSecondPointerPosition(event);
			const x = 0.5 * (event.pageX + position.x);
			const y = 0.5 * (event.pageY + position.y);
			this._rotateEnd.set(x, y);
		}

		this._rotateDelta.subVectors(this._rotateEnd, this._rotateStart).multiplyScalar(this.rotateSpeed);
		const element = this.domElement;

		this._sphericalDelta.theta -= _twoPI * this._rotateDelta.x / element.clientHeight; // yes, height
		this._sphericalDelta.phi -= _twoPI * this._rotateDelta.y / element.clientHeight;
		this._rotateStart.copy(this._rotateEnd);

	}

	_handleTouchMovePan(event) {
		if (this._pointers.length === 1) {
			this._panEnd.set(event.pageX, event.pageY);
		} else {
			const position = this._getSecondPointerPosition(event);

			const x = 0.5 * (event.pageX + position.x);
			const y = 0.5 * (event.pageY + position.y);
			this._panEnd.set(x, y);
		}

		this._panDelta.subVectors(this._panEnd, this._panStart).multiplyScalar(this.panSpeed);
		this._pan(this._panDelta.x, this._panDelta.y);
		this._panStart.copy(this._panEnd);
	}

	_handleTouchMoveDolly(event) {
		const position = this._getSecondPointerPosition( event );
		const dx = event.pageX - position.x;
		const dy = event.pageY - position.y;
		const distance = Math.sqrt(dx * dx + dy * dy);

    if (this.mode === "fps") {
      _v.setFromMatrixColumn(this.object.matrix, 2);
      this._panOffset.addScaledVector(_v, -distance * 0.01 * this.zoomSpeed);
    } else if (this.mode === "orbit") {
      this._dollyEnd.set(0, distance);
      this._dollyDelta.set(0, Math.pow(this._dollyEnd.y / this._dollyStart.y, this.zoomSpeed));
      this._scale /= this._dollyDelta.y;
      this._dollyStart.copy(this._dollyEnd);
    } else {
      console.error(`Unknown mode: ${this.mode}`);
    }
	}

	_handleTouchMoveDollyPan(event) {
		if (this.enableZoom) this._handleTouchMoveDolly(event);
		if (this.enablePan) this._handleTouchMovePan(event);
	}

	_handleTouchMoveDollyRotate(event) {
		if (this.enableZoom) this._handleTouchMoveDolly(event);
		if (this.enableRotate) this._handleTouchMoveRotate(event);
	}

	// pointers

	_addPointer(event) {
		this._pointers.push( event.pointerId );
	}

	_removePointer(event) {
		delete this._pointerPositions[event.pointerId];
		for (let i = 0; i < this._pointers.length; i++) {
			if (this._pointers[ i ] == event.pointerId) {
				this._pointers.splice(i, 1);
				return;
			}
		}
	}

	_isTrackingPointer(event) {
		for (let i = 0; i < this._pointers.length; i++) {
			if (this._pointers[i] == event.pointerId) return true;
		}
		return false;
	}

	_trackPointer(event) {
		let position = this._pointerPositions[event.pointerId];
		if (position === undefined) {
			position = new Vector2();
			this._pointerPositions[event.pointerId] = position;
		}
		position.set(event.pageX, event.pageY);
	}

	_getSecondPointerPosition(event) {
		const pointerId = (event.pointerId === this._pointers[0] ) ? this._pointers[1] : this._pointers[0];
		return this._pointerPositions[pointerId];
	}

	//

	_customWheelEvent(event) {
		const mode = event.deltaMode;

		// minimal wheel event altered to meet delta-zoom demand
		const newEvent = {
			clientX: event.clientX,
			clientY: event.clientY,
			deltaY: event.deltaY,
		};

		switch (mode) {
			case 1: // LINE_MODE
				newEvent.deltaY *= 16;
				break;

			case 2: // PAGE_MODE
				newEvent.deltaY *= 100;
				break;

		}

		// detect if event was triggered by pinching
		if (event.ctrlKey && ! this._controlActive) {
			newEvent.deltaY *= 10;
		}

		return newEvent;
	}
}

function onPointerDown(event) {
	if (this.enabled === false) return;

	if (this._pointers.length === 0) {
		this.domElement.setPointerCapture(event.pointerId);

		this.domElement.addEventListener('pointermove', this._onPointerMove);
		this.domElement.addEventListener('pointerup', this._onPointerUp);
	}

	if (this._isTrackingPointer(event)) return;
	this._addPointer(event);

	if (event.pointerType === 'touch') {
		this._onTouchStart(event);
	} else {
		this._onMouseDown(event);
	}
}

function onPointerMove(event) {
	if (this.enabled === false) return;
	if (event.pointerType === 'touch') {
		this._onTouchMove(event);
	} else {
		this._onMouseMove(event);
	}
}

function onPointerUp(event) {
	this._removePointer(event);
	switch (this._pointers.length) {
		case 0:
			this.domElement.releasePointerCapture(event.pointerId);
			this.domElement.removeEventListener('pointermove', this._onPointerMove);
			this.domElement.removeEventListener('pointerup', this._onPointerUp);
			this.dispatchEvent(_endEvent);
			this.state = _STATE.NONE;
			break;
		case 1:
			const pointerId = this._pointers[0];
			const position = this._pointerPositions[pointerId];

			// minimal placeholder event - allows state correction on pointer-up
			this._onTouchStart({ pointerId: pointerId, pageX: position.x, pageY: position.y });
			break;
	}
}

function onMouseDown(event) {
	let mouseAction;

	switch (event.button) {
		case 0:
			mouseAction = this.mouseButtons.LEFT;
			break;
		case 1:
			mouseAction = this.mouseButtons.MIDDLE;
			break;
		case 2:
			mouseAction = this.mouseButtons.RIGHT;
			break;
		default:
			mouseAction = -1;
	}

	switch (mouseAction) {
		case MOUSE.DOLLY:
			if (this.enableZoom === false) return;
			this._handleMouseDownDolly(event);
			this.state = _STATE.DOLLY;
			break;

		case MOUSE.ROTATE:
			if (event.ctrlKey || event.metaKey || event.shiftKey) {
				if (this.enablePan === false) return;
				this._handleMouseDownPan(event);
				this.state = _STATE.PAN;
			} else {
				if (this.enableRotate === false) return;
				this._handleMouseDownRotate(event);
				this.state = _STATE.ROTATE;
			}

			break;

		case MOUSE.PAN:
			if (event.ctrlKey || event.metaKey || event.shiftKey) {
				if (this.enableRotate === false) return;
				this._handleMouseDownRotate(event);
				this.state = _STATE.ROTATE;
			} else {
				if (this.enablePan === false) return;
				this._handleMouseDownPan(event);
				this.state = _STATE.PAN;
			}

			break;

		default:
			this.state = _STATE.NONE;

	}

	if (this.state !== _STATE.NONE) {
		this.dispatchEvent(_startEvent);
	}
}

function onMouseMove(event) {
	switch (this.state) {
		case _STATE.ROTATE:
			if (this.enableRotate === false) return;
			this._handleMouseMoveRotate(event);
			break;
		case _STATE.DOLLY:
			if (this.enableZoom === false) return;
			this._handleMouseMoveDolly(event);
			break;
		case _STATE.PAN:
			if (this.enablePan === false) return;
			this._handleMouseMovePan(event);
			break;
	}
}

function onMouseWheel(event) {
	if (this.enabled === false || this.enableZoom === false || this.state !== _STATE.NONE) return;
	event.preventDefault();

	this.dispatchEvent(_startEvent);
	this._handleMouseWheel(this._customWheelEvent(event));
	this.dispatchEvent(_endEvent);
}


function onTouchStart(event) {
	this._trackPointer(event);
	switch (this._pointers.length) {
		case 1:
			switch (this.touches.ONE) {
				case TOUCH.ROTATE:
					if (this.enableRotate === false) return;
					this._handleTouchStartRotate(event);
					this.state = _STATE.TOUCH_ROTATE;
					break;
				case TOUCH.PAN:
					if (this.enablePan === false) return;
					this._handleTouchStartPan(event);
					this.state = _STATE.TOUCH_PAN;
					break;
				default:
					this.state = _STATE.NONE;
			}
			break;
		case 2:
			switch (this.touches.TWO) {
				case TOUCH.DOLLY_PAN:
					if (this.enableZoom === false && this.enablePan === false) return;
					this._handleTouchStartDollyPan(event);
					this.state = _STATE.TOUCH_DOLLY_PAN;
					break;
				case TOUCH.DOLLY_ROTATE:
					if (this.enableZoom === false && this.enableRotate === false) return;
					this._handleTouchStartDollyRotate(event);
					this.state = _STATE.TOUCH_DOLLY_ROTATE;
					break;
				default:
					this.state = _STATE.NONE;
			}
			break;
		default:
			this.state = _STATE.NONE;
	}

	if (this.state !== _STATE.NONE) {
		this.dispatchEvent(_startEvent);
	}
}

function onTouchMove(event) {
	this._trackPointer(event);
	switch (this.state) {
		case _STATE.TOUCH_ROTATE:
			if (this.enableRotate === false) return;
			this._handleTouchMoveRotate(event);
			this.update();
			break;
		case _STATE.TOUCH_PAN:
			if (this.enablePan === false) return;
			this._handleTouchMovePan(event);
			this.update();
			break;
		case _STATE.TOUCH_DOLLY_PAN:
			if (this.enableZoom === false && this.enablePan === false) return;
			this._handleTouchMoveDollyPan(event);
			this.update();
			break;
		case _STATE.TOUCH_DOLLY_ROTATE:
			if (this.enableZoom === false && this.enableRotate === false) return;
			this._handleTouchMoveDollyRotate(event);
			this.update();
			break;
		default:
			this.state = _STATE.NONE;
	}
}

function onContextMenu(event) {
	if (this.enabled === false) return;
	event.preventDefault();
}

function interceptControlDown(event) {
	if (event.key === 'Control') {
		this._controlActive = true;
		const document = this.domElement.getRootNode(); // offscreen canvas compatibility
		document.addEventListener('keyup', this._interceptControlUp, { passive: true, capture: true });
	}
}

function interceptControlUp(event) {
	if (event.key === 'Control') {
		this._controlActive = false;
		const document = this.domElement.getRootNode(); // offscreen canvas compatibility
		document.removeEventListener('keyup', this._interceptControlUp, { passive: true, capture: true });
	}
}

export { ViewerControls };
