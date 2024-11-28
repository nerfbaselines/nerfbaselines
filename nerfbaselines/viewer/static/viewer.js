import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { compute_camera_path } from './interpolation.js';
import { PivotControls, MouseInteractions } from './threejs_utils.js';



const viewport = document.querySelector(".viewport");
const renderers = [];
const state = {
  renderers: {},
};

class HTTPRenderer {
  constructor(baseUrl) {
    this._baseUrl = baseUrl;
    // Generate uuid
    this.state = null;
    this.onUpdateFrame = null;
    this.onUpdateState = null;

  }

  async updateRenderParams(renderParams) {
    await fetch(`${this._baseUrl}/set-feed-params?feedid=${this.state.feedid}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(renderParams),
    });
  }

  async _run(reader) {
    const boundary = "frame"; // Boundary from server response
    const boundaryBytes = new TextEncoder().encode(`--${boundary}`);
    const headerSeparator = new Uint8Array([13, 10, 13, 10]); // "\r\n\r\n"

    let buffer = new Uint8Array(0);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // Concatenate new data into buffer
      const tempBuffer = new Uint8Array(buffer.length + value.length);
      tempBuffer.set(buffer);
      tempBuffer.set(value, buffer.length);
      buffer = tempBuffer;

      // Process all frames in the current buffer
      while (true) {
        const boundaryIndex = this._findSequence(buffer, boundaryBytes);
        if (boundaryIndex === -1) break;

        const headerEndIndex = this._findSequence(buffer, headerSeparator, boundaryIndex + boundaryBytes.length);
        if (headerEndIndex === -1) break;

        const frameStart = headerEndIndex + headerSeparator.length;
        const nextBoundaryIndex = this._findSequence(buffer, boundaryBytes, frameStart);
        if (nextBoundaryIndex === -1) break;

        const frameData = buffer.slice(frameStart, nextBoundaryIndex);

        // Create a Blob and update the image element
        const blob = new Blob([frameData], { type: "image/jpeg" });
        const image = new Image();
        image.src = URL.createObjectURL(blob);
        image.onload = () => {
          if (this.onUpdateFrame) {
            this.onUpdateFrame(image);
          }
        };

        // Remove processed data from the buffer
        buffer = buffer.slice(nextBoundaryIndex);
      }
    }
  }

  async _run_polling() {
    while (this._running) {
      try {
        const response = await fetch(`/get-state?poll=${this.state.version}&feedid=${this.state.feedid}`);
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        const data = await response.json();
        const oldVersion = this.state.version;
        if (data) {
          this.state = Object.assign(this.state, data);
        }
        if (this.state.version != oldVersion && this.onUpdateState) {
          this.onUpdateState(data);
        }
      } catch (error) {
        console.error("Error fetching updates:", error);

        // Retry after a delay in case of an error
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }
  }


  async start() {
    this.state = {};
    const setFeedResponse = await fetch(this._baseUrl + "/set-feed-params", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        width: viewport.clientWidth,
        height: viewport.clientHeight,
      }),
    }).then((response) => response.json());
    if (setFeedResponse.status !== "ok") throw exception("Failed to set feed params");
    this.state.feedid = setFeedResponse.feedid;
    this.state.version = 0;
    const response = await fetch(`${this._baseUrl}/video-feed?feedid=${this.state.feedid}`);
    if (!response.body) {
      throw new Error("ReadableStream not yet supported in this browser");
    }
    const reader = response.body.getReader();


    // NOTE: We start the async loop here
    this._running = true;
    this._run(reader).catch((err) => {
      console.error("Error:", err);
    });
    this._run_polling().catch((err) => {
      console.error("Error:", err);
    });
  }

  _findSequence(buffer, sequence, startIndex = 0) {
    for (let i = startIndex; i <= buffer.length - sequence.length; i++) {
      let found = true;
      for (let j = 0; j < sequence.length; j++) {
        if (buffer[i + j] !== sequence[j]) {
          found = false;
          break;
        }
      }
      if (found) return i;
    }
    return -1;
  }
}

const lastFrames = {};
function draw() {
  // Will be defined later
}
window._draw = draw;

async function start() {
  const renderer = new HTTPRenderer("http://localhost:5001");
  await renderer.start();

  renderer.onUpdateFrame = (frame) => {
    lastFrames[0] = frame;
    window._draw();
  };
  // renderer.onUpdateState = (state) => {
  //   console.log("State:", renderer.state, state);
  // };
  renderers.push(renderer);
}

start();

window.addEventListener("resize", async () => {
  draw();
  for (const renderer of renderers) {
    await renderer.updateRenderParams({
      width: viewport.clientWidth,
      height: viewport.clientHeight,
    }).catch((err) => {
      console.error("Error updating render params:", err);
    });
  }
});

// Implement WebRTC


const _point_cloud_vertex_shader = `
precision mediump float;

varying vec3 vPosition;
varying vec3 vColor; // in the vertex shader
uniform float scale;

void main() {
    vPosition = position;
    vColor = color;
    vec4 world_pos = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * world_pos;
    gl_PointSize = (scale / -world_pos.z);
}`

const _point_cloud_fragment_shader = `
varying vec3 vPosition;
varying vec3 vColor;
uniform float point_ball_norm;

void main() {
    if (point_ball_norm < 1000.0) {
        float r = pow(
            pow(abs(gl_PointCoord.x - 0.5), point_ball_norm)
            + pow(abs(gl_PointCoord.y - 0.5), point_ball_norm),
            1.0 / point_ball_norm);
        if (r > 0.5) discard;
    }
    gl_FragColor = vec4(vColor, 1.0);
}`

class ThreeJSRenderer {
  constructor(viewport) {
    this._R_threecam_cam = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI, 0.0, 0.0));
    this._backgroundTexture = null;
    this._renderer = new THREE.WebGLRenderer({ antialias: true });
    this._renderer.setSize(viewport.clientWidth, viewport.clientHeight);
    viewport.appendChild(this._renderer.domElement);
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;

    this._camera = new THREE.PerspectiveCamera( 70, width / height, 0.01, 10 );
    this._camera.position.z = 1;

    this._scene = new THREE.Scene();
    this._scene.add(new PivotControls({}));

    const geometry = new THREE.BoxGeometry( 0.2, 0.2, 0.2 );
    const material = new THREE.MeshNormalMaterial();

    this._mesh = new THREE.Mesh( geometry, material );
    this._scene.add(this._mesh);

    this._controls = new OrbitControls(this._camera, viewport);
    this._controls.listenToKeyEvents(window);


    this._controls.enableDamping = true; // an animation loop is required when either damping or auto-rotation are enabled
   	this._controls.dampingFactor = 0.05;
		this._controls.screenSpacePanning = false;
		this._controls.maxPolarAngle = Math.PI / 2;

    this._enabled = true;
    this._renderer.setAnimationLoop((time) => this._animate(time));
    window.addEventListener("resize", () => this._resize());

    this._mouse_interactions = new MouseInteractions(this._renderer, this._camera, this._scene);
  }

  set_enabled(enabled) {
    this._enabled = enabled;
    this._controls.enabled = enabled;
  }

  _addPointCloud({ points, colors, pointball_norm = 2.0}) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float16BufferAttribute(points, 3));
    geometry.computeBoundingSphere();
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));

    const material = new THREE.ShaderMaterial({ 
      uniforms: {
        scale: { value: 10.0 },
        point_ball_norm: { value: pointball_norm },
      },
      vertexShader: _point_cloud_vertex_shader,
      fragmentShader: _point_cloud_fragment_shader,
    });
    material.vertexColors = true;

    const three_points = new THREE.Points(geometry, material)
    this._scene.add(three_points);

    return {
      remove: () => {
        this._scene.remove(three_points);
        material.dispose();
        geometry.dispose();
      },
    };
  }

  add_trajectory_curve({ points, color }) {
    const point_geometry = new THREE.BufferGeometry();
    const point_material = new THREE.ShaderMaterial({ 
      uniforms: {
        scale: { value: 10.0 },
        point_ball_norm: { value: 2.0 },
      },
      vertexShader: _point_cloud_vertex_shader,
      fragmentShader: _point_cloud_fragment_shader,
    });
    point_material.vertexColors = true;
    const geometry = new LineSegmentsGeometry();
    const material = new LineMaterial({
      color: color,
      linewidth: 2,
      resolution: new THREE.Vector2(viewport.clientWidth, viewport.clientHeight),
    });

    function set_positions(points) {
      point_geometry.setAttribute("position", new THREE.Float16BufferAttribute(points, 3));
      point_geometry.computeBoundingSphere();
      // Make Uint8Array of colors
      const colors = new Uint8Array(points.length * 3);
      const threeColor = new THREE.Color(color);
      for (let i = 0; i < points.length; i++) {
        colors[i * 3] = threeColor.r * 255;
        colors[i * 3 + 1] = threeColor.g * 255;
        colors[i * 3 + 2] = threeColor.b * 255;
      }
      point_geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));

      if (points.length < 2) return;
      const segment_points = [];
      for (let i = 0; i < points.length; i++) {
        segment_points.push(points[i].x, points[i].y, points[i].z);
        segment_points.push(points[i].x, points[i].y, points[i].z);
      }
      segment_points.splice(0, 3);
      segment_points.splice(segment_points.length - 3, 3);
      geometry.setPositions(segment_points);
    }
    set_positions(points);
    // Attach material to geometry.
    const segments = new LineSegments2(geometry, material);
    const three_points = new THREE.Points(point_geometry, point_material);
    this._scene.add(segments);
    this._scene.add(three_points);
    return {
      remove: () => {
        this._scene.remove(segments);
        this._scene.remove(three_points);
        geometry.dispose();
        material.dispose();
        point_material.dispose();
        point_geometry.dispose();
      },
      update: set_positions,
    };
  }

  add_camera_frustum({ fov, aspect, scale, color, quaternion, position }) {
    const imageTexture = undefined;

    scale = scale || 1;
    let y = Math.tan(fov / 2.0);
    let x = y * aspect;
    let z = 1.0;

    const volumeScale = Math.cbrt((x * y * z) / 3.0);
    x /= volumeScale;
    y /= volumeScale;
    z /= volumeScale;
    x *= scale;
    y *= scale;
    z *= scale;

    const frustumPoints = [
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
      imageTexture === undefined ? [0.0, -0.9, 1.0] : [0.0, -1.0, 1.0],
    ].map((xyz) => [xyz[0] * x, xyz[1] * y, xyz[2] * z]);
    const geometry = new LineSegmentsGeometry();
    geometry.setPositions(frustumPoints.flat())
    geometry.applyMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(quaternion));
    geometry.applyMatrix4(new THREE.Matrix4().makeTranslation(position));
    const material = new LineMaterial({
      color: color,
      linewidth: 2,
      resolution: new THREE.Vector2(viewport.clientWidth, viewport.clientHeight),
    });
    // Attach material to geometry.
    const segments = new LineSegments2(geometry, material);
    this._scene.add(segments);
    return {
      remove: () => {
        this._scene.remove(segments);
        geometry.dispose();
        material.dispose();
      }
    };
  }

  _resize() {
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    this._camera.aspect = width / height;
    this._camera.updateProjectionMatrix();
    this._drawBackground();
    this._renderer.setSize(width, height);
  }

  _animate(time) {
    if (this._enabled) {
      this._mouse_interactions.update();
      if (!this._mouse_interactions.isCaptured())
        this._controls.update();
      this._renderer.render(this._scene, this._camera);
    }
  }

  _drawBackground() {
    if (this._backgroundTexture === null) {
      this._backgroundTexture = new THREE.Texture(lastFrames[0]);
      this._scene.background = this._backgroundTexture;
    } else if (this._backgroundTexture.image.width !== lastFrames[0].width || this._backgroundTexture.image.height !== lastFrames[0].height) {
      // Dispose the old texture
      this._backgroundTexture.dispose();
      this._backgroundTexture = new THREE.Texture(lastFrames[0]);
      this._scene.background = this._backgroundTexture;
    } else {
      this._backgroundTexture.image = lastFrames[0];
    }
    this._backgroundTexture.needsUpdate = true;
  }

  getCameraPose() {
    const quaternion = this._camera.quaternion.clone().multiply(this._R_threecam_cam);
    return {
      quaternion: quaternion,
      position: this._camera.position.clone(),
    };
  }
}

class Viewer {
  constructor(viewport) {
    this._threejs_renderer = new ThreeJSRenderer(viewport);
    this._preview_canvas = document.createElement("canvas");
    this._preview_canvas.style.width = "100%";
    this._preview_canvas.style.height = "100%";
    this._preview_canvas.style.display = "none";
    this._preview_canvas.width = viewport.clientWidth;
    this._preview_canvas.height = viewport.clientHeight;
    viewport.appendChild(this._preview_canvas);
    this._preview_context = this._preview_canvas.getContext("2d");
    this._preview_context.fillStyle = "black";
    this._preview_context.fillRect(0, 0, this._preview_canvas.width, this._preview_canvas.height);

    this._keyframes = [];
    this._gui_state = state;
    this._camera_path = null;

    this._trajectory_curve = undefined;
    this._player_frustum = undefined;

    this._on_camera_path_change_callbacks = [];
    this._on_gui_change_callbacks = [];
  }

  _update_camera_path() {
    // Unpack dependencies
    const { 
      camera_path_loop = false,
      camera_path_interpolation = 'none',
      camera_path_tension = 0.5,
      camera_path_show_spline = true,
    } = this._gui_state;

    this._camera_path = compute_camera_path({
      keyframes: this._keyframes,
      loop: camera_path_loop,
      interpolation: camera_path_interpolation,
      tension: camera_path_tension,
    });
    for (const callback of this._on_camera_path_change_callbacks) {
      callback();
    }

    if (this._trajectory_curve) {
      this._trajectory_curve.remove();
      this._trajectory_curve = undefined;
    }

    if (camera_path_show_spline && 
        camera_path_interpolation !== 'none' && 
        this._camera_path && 
        this._camera_path.positions.length > 0) {
      const { positions } = this._camera_path;
      if (this._trajectory_curve !== undefined) {
        this._trajectory_curve.update(positions);
      } else {
        this._trajectory_curve = this._threejs_renderer.add_trajectory_curve({
          points: positions,
          color: new THREE.Color(0x00ffff),
        });
      }
    } else {
      if (this._trajectory_curve) {
        this._trajectory_curve.remove();
        this._trajectory_curve = undefined;
      }
    }
  }

  clearKeyframes() {
    const keyframes = this._keyframes;
    this._keyframes = [];
    for (const keyframe of keyframes) {
      keyframe.frustum.remove();
    }
    this._update_camera_path();
    this._update_player_frustum();
    this._update_show_keyframes();
  }

  _update_show_keyframes() {
    const { camera_path_show_keyframes } = this._gui_state;
    for (const keyframe of this._keyframes) {
      if (camera_path_show_keyframes) {
        if (keyframe.frustum === null) {
          keyframe.frustum = this._threejs_renderer.add_camera_frustum(keyframe.frustum_props);
        }
      } else if (keyframe.frustum !== null) {
        keyframe.frustum.remove();
        keyframe.frustum = null;
      }
    }
  }

  add_keyframe() {
    const { camera_path_show_keyframes } = this._gui_state;
    const { quaternion, position } = this._threejs_renderer.getCameraPose();
    const frustum_props = {
      fov: this._gui_state.render_fov / 180 * Math.PI,
      aspect: this._gui_state.render_resolution_1 / this._gui_state.render_resolution_2,
      scale: 0.1,
      color: 0xff0000,
      quaternion: quaternion.clone(),
      position: position.clone(),
    };
    this._keyframes.push({
      quaternion: quaternion,
      position: position,
      frustum_props: frustum_props,
      frustum: null,
    });
    this._update_camera_path();
    this._update_player_frustum();
    this._update_show_keyframes();
  }

  _update_player_frustum() {
    const { 
      preview_frame,
      render_resolution_1,
      render_resolution_2,
    } = this._gui_state;

    if (this._camera_path === null || this._camera_path.positions.length === 0) {
      if (this._player_frustum !== undefined) {
        this._player_frustum.remove();
        this._player_frustum = undefined;
      }
      return;
    }
    const { positions, quaternions, fovs } = this._camera_path;
    const num_frames = positions.length;
    const frame = Math.min(Math.max(0, Math.floor(preview_frame)), num_frames - 1);
    const quaternion = new THREE.Quaternion().copy(quaternions[frame]);
    const position = new THREE.Vector3().copy(positions[frame]);
    const fov = fovs[frame];
    const frustum_props_update = {
      fov: fov,
      aspect: render_resolution_1 / render_resolution_2,
      quaternion: quaternion,
      position: position,
    };
    if (this._player_frustum) {
      this._player_frustum.remove();
      this._player_frustum = undefined;
    }
    this._player_frustum = this._threejs_renderer.add_camera_frustum({
      ...frustum_props_update,
      scale: 0.1,
      color: 0x00ff00,
    });
  }

  _update_preview_mode() {
    const { 
      preview_is_preview_mode,
    } = this._gui_state;
    this._threejs_renderer.set_enabled(!preview_is_preview_mode);
    this._threejs_renderer._renderer.domElement.style.display = preview_is_preview_mode ? "none" : "block";
    this._preview_canvas.style.display = preview_is_preview_mode ? "block" : "none";
  }

  _update_preview_is_playing() {
    const {
      camera_path_framerate,
      camera_path_interpolation,
      camera_path_default_transition_duration,
      preview_is_playing,
    } = this._gui_state;

    // Add preview timer
    if (this._preview_interval) {
      clearInterval(this._preview_interval);
      this._preview_interval = null;
    }
    if (preview_is_playing) {
      const fps = camera_path_interpolation === 'none' ? 1 / camera_path_default_transition_duration : camera_path_framerate;
      const n = this._camera_path ? this._camera_path.positions.length : 0;
      this._preview_interval = setInterval(() => {
        this._gui_state.preview_frame = (this._gui_state.preview_frame + 1) % n;
        this._on_gui_change({ property: 'preview_frame', trigger: 'preview' });
      }, 1000 / fps);
    }
  }

  _on_gui_change({ property, trigger }) {
    if (property === 'camera_path_loop' ||
        property === 'camera_path_interpolation' ||
        property === 'camera_path_tension' ||
        property === 'camera_path_show_spline') {
      this._update_camera_path();
      this._update_player_frustum();
      this._update_preview_is_playing();
    }

    if (property === 'preview_frame' ||
        property === 'render_resolution_1' ||
        property === 'render_resolution_2') {
      this._update_player_frustum();
    }

    if (property === 'preview_is_playing' ||
        property === 'camera_path_framerate' ||
        property === 'camera_path_interpolation' ||
        property === 'camera_path_default_transition_duration') {
      this._update_preview_is_playing();
    }

    if (property === 'camera_path_show_keyframes') {
      this._update_show_keyframes();
    }

    if (property === 'preview_is_preview_mode') {
      this._update_preview_mode();
    }

    if (property === null) {
      this._update_camera_path();
      this._update_player_frustum();
      this._update_preview_is_playing();
      this._update_show_keyframes();
      this._update_preview_mode();
    }

    for (const callback of this._on_gui_change_callbacks) {
      callback({ property, trigger });
    }
  }

  attach_gui(root) {
    // Handle state change
    function getValue(element) {
      const { name, value, type, checked } = element;
      if (type === "checkbox") return checked;
      if (type === "number" || type === "range") return parseFloat(value);
      return value;
    }
    function setValue(element, value) {
      const { type } = element;
      if (type === "checkbox") {
        element.checked = value;
      } else {
        element.value = value;
      }
    }
    root.querySelectorAll("input[name],select[name]").forEach(element => {
      element.addEventListener("change", (event) => {
        state[name] = getValue(event.target);
        this._on_gui_change({ property: name, trigger: 'gui_change' });
      });
      element.addEventListener("input", (event) => {
        state[name] = getValue(event.target);
        this._on_gui_change({ property: name, trigger: 'gui_changing' });
        if (name === "preview_frame" && type === "range") {
          // Changing preview_frame stops the preview
          state.preview_is_playing = false;
          this._on_gui_change({ property: "preview_is_playing", trigger: 'preview_frame_change' });
        }
      });
      const { name, value, type, checked } = element;
      state[name] = getValue(element);
      if (name === "preview_frame" && type === "range") {
        // Update maximum when the camera path changes
        this._on_camera_path_change_callbacks.push(() => {
          element.max = this._camera_path ? (this._camera_path.positions.length - 1) : 0;
          const old_value = state[name];
          const new_value = Math.min(element.value, element.max);
          if (old_value !== new_value) {
            element.value = new_value;
            element.dispatchEvent(new Event("input"));
          }
        });
      }
      this._on_gui_change_callbacks.push(({ property, trigger }) => {
        if (property !== name) return;
        if (trigger === 'init' || trigger === 'gui_change' || trigger === 'gui_changing') return;
        setValue(element, state[name]);
      });
    });
    this._on_gui_change({ property: null, trigger: 'init' });
  }

  _draw_background() {
    if (!this._gui_state.preview_is_preview_mode) {
      this._threejs_renderer._drawBackground();
    } else {
      // Manually draw the background to the canvas
      const image = lastFrames[0];
      if (image === undefined) return;
      const { width, height } = this._preview_canvas;
      this._preview_context.drawImage(image, 0, 0, width, height);
    }
  }
}


// Attach GUI
document.getElementById("button_add_keyframe").addEventListener("click", () => viewer.add_keyframe());
document.getElementById("button_clear_keyframes").addEventListener("click", () => viewer.clearKeyframes());
document.querySelectorAll('.row > input[type="range"] + input[type="number"]').forEach((numberInput) => {
  const rangeInput = numberInput.previousElementSibling;
  rangeInput.addEventListener("input", () => {
    numberInput.value = rangeInput.value;
  });
  numberInput.addEventListener("input", () => {
    rangeInput.value = numberInput.value;
    rangeInput.dispatchEvent(new Event("input"));
  });
});

const viewer = new Viewer(viewport);
viewer.attach_gui(document.querySelector('.controls'));
viewer.add_keyframe();
window._draw = () => viewer._draw_background();
