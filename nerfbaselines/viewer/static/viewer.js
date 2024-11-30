import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { compute_camera_path } from './interpolation.js';
import { PivotControls, MouseInteractions, CameraFrustum, TrajectoryCurve } from './threejs_utils.js';


const theme_color = 0xffd369;
const trajectory_curve_color = 0xffd369;
const player_frustum_color = 0x20df80;
const keyframe_frustum_color = 0xff0000;
const dataset_frustum_color = 0xd3d3d3;

let _keyframe_counter = 0;
const viewport = document.querySelector(".viewport");
const renderers = [];
const state = {
  renderers: {},
};


class HTTPRemote {
  constructor(baseUrl) {
    this._baseUrl = baseUrl;
  }

  load_dataset_point_cloud(dataset) {
    const ply_url = `${this._baseUrl}/datasets/${encodeURIComponent(dataset)}/point_cloud.ply`;
  }
}

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
    // const response = await fetch(`${this._baseUrl}/video-feed?feedid=${this.state.feedid}`);
    // if (!response.body) {
    //   throw new Error("ReadableStream not yet supported in this browser");
    // }
    // const reader = response.body.getReader();


    // NOTE: We start the async loop here
    this._running = true;
    //this._run();
    // this._run(reader).catch((err) => {
    //   console.error("Error:", err);
    // });
    //this._run_polling().catch((err) => {
    //  console.error("Error:", err);
    //});
  }

  async _updateSingle() {
    const response = await fetch(`${this._baseUrl}/render?width=${viewport.clientWidth}&height=${viewport.clientHeight}`);
    // Read response as blob
    const blob = await response.blob();
    const image = new Image();
    image.src = URL.createObjectURL(blob);
    image.onload = () => {
      if (this.onUpdateFrame) {
        this.onUpdateFrame(image);
      }
    };
  }

  _run() {
    let lastUpdate = Date.now() - 1;

    const run = () => {
      this._updateSingle().then(() => {
        if (this._running) {
          const wait = Math.max(0, 1000 / 30 - (Date.now() - lastUpdate));
          lastUpdate = Date.now();
          setTimeout(() => run(), wait);
        }
      });
    };
    run();
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
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(viewport.clientWidth, viewport.clientHeight);
    viewport.appendChild(this.renderer.domElement);
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;

    this._camera = new THREE.PerspectiveCamera( 70, width / height, 0.01, 10 );
    this._camera.position.z = 1;

    this.scene = new THREE.Scene();

    const geometry = new THREE.BoxGeometry( 0.2, 0.2, 0.2 );
    const material = new THREE.MeshNormalMaterial();

    this._mesh = new THREE.Mesh( geometry, material );
    this.scene.add(this._mesh);

    this._controls = new OrbitControls(this._camera, viewport);
    this._controls.listenToKeyEvents(window);


    this._controls.enableDamping = true; // an animation loop is required when either damping or auto-rotation are enabled
   	this._controls.dampingFactor = 0.05;
		this._controls.screenSpacePanning = false;
		this._controls.maxPolarAngle = Math.PI / 2;

    this._enabled = true;
    this.renderer.setAnimationLoop((time) => this._animate(time));
    window.addEventListener("resize", () => this._resize());

    this._mouse_interactions = new MouseInteractions(this.renderer, this._camera, this.scene);
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
    this.scene.add(three_points);

    return {
      remove: () => {
        this.scene.remove(three_points);
        material.dispose();
        geometry.dispose();
      },
    };
  }

  add_trajectory_curve({ points, color }) {
    const geometry = new LineSegmentsGeometry();
    const material = new LineMaterial({
      color: color,
      linewidth: 4,
    });

    function setPositions(points) {
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
    setPositions(points);
    const segments = new LineSegments2(geometry, material);
    this.scene.add(segments);
    return {
      remove: () => {
        this.scene.remove(segments);
        geometry.dispose();
        material.dispose();
      },
      update: setPositions,
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
    const material = new LineMaterial({
      color: color,
      linewidth: 4,
      resolution: new THREE.Vector2(viewport.clientWidth, viewport.clientHeight),
    });
    // Attach material to geometry.
    const segments = new LineSegments2(geometry, material);

    // Return group
    const group = new THREE.Group();
    group.add(segments);
    group.position.copy(position);
    group.quaternion.copy(quaternion);
    group.remove = () => {
      this.scene.remove(group);
      geometry.dispose();
      material.dispose();
    }
    this.scene.add(group);
    return group;
  }

  _resize() {
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    this._camera.aspect = width / height;
    this._camera.updateProjectionMatrix();
    this._drawBackground();
    this.renderer.setSize(width, height);
  }

  _animate(time) {
    if (this._enabled) {
      this._mouse_interactions.update();
      if (!this._mouse_interactions.isCaptured())
        this._controls.update();
      this.renderer.render(this.scene, this._camera);
    }
  }

  _drawBackground() {
    if (this._backgroundTexture === null) {
      this._backgroundTexture = new THREE.Texture(lastFrames[0]);
      this.scene.background = this._backgroundTexture;
    } else if (this._backgroundTexture.image.width !== lastFrames[0].width || this._backgroundTexture.image.height !== lastFrames[0].height) {
      // Dispose the old texture
      this._backgroundTexture.dispose();
      this._backgroundTexture = new THREE.Texture(lastFrames[0]);
      this.scene.background = this._backgroundTexture;
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


function _attach_camera_path_selected_keyframe_pivot_controls(viewer) {
  let pivot_controls = undefined;
  let selected_keyframe = undefined;

  function update_keyframe_frustums({
    camera_path_keyframes,
    camera_path_selected_keyframe,
  }) {
    selected_keyframe = undefined;
    if (camera_path_selected_keyframe === undefined) {
      if (pivot_controls) {
        viewer.threejs_renderer.scene.remove(pivot_controls);
        pivot_controls.dispose();
        pivot_controls = undefined;
      }
      return;
    }
    selected_keyframe = camera_path_keyframes.find((keyframe) => keyframe.id === camera_path_selected_keyframe);
    if (!pivot_controls) {
      pivot_controls = new PivotControls({
        scale: 0.2,
      });
      viewer.threejs_renderer.scene.add(pivot_controls);
      pivot_controls.addEventListener("drag", (e) => {
        const matrix = e.matrix;
        // Decompose matrix into quaternion and position
        const quaternion = new THREE.Quaternion();
        const position = new THREE.Vector3();
        const scale = new THREE.Vector3();
        matrix.decompose(position, quaternion, scale);

        selected_keyframe.position.copy(position);
        selected_keyframe.quaternion.copy(quaternion);
        viewer.notifyChange({
          property: "camera_path_keyframes",
          trigger: "pivot_controls",
        });
      });
    }
    const matrix = new THREE.Matrix4().compose(
      selected_keyframe.position, 
      selected_keyframe.quaternion, 
      new THREE.Vector3(1, 1, 1));
    console.log("Updating pivot controls", matrix);
    pivot_controls.setMatrix(matrix);
  }

  viewer.addEventListener("change", ({ property, state, trigger }) => {
    if (property !== undefined &&
        property !== 'camera_path_keyframes' &&
        property !== 'camera_path_selected_keyframe') return;
    if (trigger === 'pivot_controls') return;
    update_keyframe_frustums(state);
  });
}


function _attach_camera_path_keyframes(viewer) {
  let keyframe_frustums = {};

  function update_keyframe_frustums({
    camera_path_keyframes,
    camera_path_selected_keyframe,
    render_fov,
    render_resolution_1,
    render_resolution_2,
  }) {
    const new_keyframe_frustums = {};
    for (const keyframe of camera_path_keyframes) {
      const fov = render_fov / 180 * Math.PI;
      const aspect = render_resolution_1 / render_resolution_2;
      let frustum = keyframe_frustums[keyframe.id];

      if (frustum === undefined) {
        frustum = new CameraFrustum({ 
          fov: fov,
          aspect: aspect,
          position: keyframe.position.clone(), 
          quaternion: keyframe.quaternion.clone(),
          scale: 0.1,
          color: keyframe_frustum_color,
          interactive: true,
          originSphereScale: 0.12,
        });
        frustum.addEventListener("click", () => {
          viewer._gui_state.camera_path_selected_keyframe = keyframe.id;
          viewer.notifyChange({ property: "camera_path_selected_keyframe" });
        });
        viewer.threejs_renderer.scene.add(frustum);
      } else {
        frustum.position.copy(keyframe.position);
        frustum.quaternion.copy(keyframe.quaternion);
        frustum.fov = fov;
        frustum.aspect = aspect;
      }
      new_keyframe_frustums[keyframe.id] = frustum;
      frustum.focused = keyframe.id === camera_path_selected_keyframe;
    }
    // Remove old keyframes
    for (const keyframe_id in keyframe_frustums) {
      if (new_keyframe_frustums[keyframe_id] === undefined) {
        viewer.threejs_renderer.scene.remove(keyframe_frustums[keyframe_id]);
        keyframe_frustums[keyframe_id].dispose();
      }
    }
    keyframe_frustums = new_keyframe_frustums;
  }

  viewer.addEventListener("change", ({ property, state }) => {
    if (property === undefined ||
        property === 'camera_path_keyframes' ||
        property === 'render_fov' ||
        property === 'render_resolution_1' ||
        property === 'render_resolution_2')
        update_keyframe_frustums(state);

    if (property === 'camera_path_selected_keyframe') {
      for (const keyframe of state.camera_path_keyframes) {
        const frustum = keyframe_frustums[keyframe.id];
        if (frustum === undefined) continue;
        frustum.focused = keyframe.id === state.camera_path_selected_keyframe;
      }
    }
  });
}


function _attach_camera_path(viewer) {
  viewer.addEventListener("change", ({ property, state }) => {
    if (property !== undefined &&
        property !== 'camera_path_keyframes' &&
        property !== 'camera_path_loop' &&
        property !== 'camera_path_interpolation' &&
        property !== 'camera_path_tension') return;
    const {
      camera_path_keyframes,
      camera_path_loop,
      camera_path_interpolation,
      camera_path_tension,
    } = state;
    state.camera_path_trajectory = undefined;
    if (camera_path_keyframes) {
      state.camera_path_trajectory = compute_camera_path({
        keyframes: camera_path_keyframes,
        loop: camera_path_loop,
        interpolation: camera_path_interpolation,
        tension: camera_path_tension,
      });
    }
    viewer.notifyChange({ property: "camera_path_trajectory", trigger: "camera_path" });
  });
}

function _attach_camera_path_curve(viewer) {
  let trajectory_curve = undefined;
  viewer.addEventListener("change", ({ property, state }) => {
    if (property !== undefined &&
        property !== 'camera_path_interpolation' &&
        property !== 'camera_path_trajectory' &&
        property !== 'camera_path_show_spline') return;
    const { 
      camera_path_trajectory, 
      camera_path_show_spline,
      camera_path_interpolation,
    } = state;

    // Remove trajectory
    if (camera_path_interpolation == 'none' || 
        !camera_path_trajectory || 
        camera_path_trajectory.positions.length === 0) {
      if (trajectory_curve) {
        viewer.threejs_renderer.scene.remove(trajectory_curve);
        trajectory_curve.dispose();
        trajectory_curve = undefined;
      }
      return;
    } else {
      // Create trajectory
      if (trajectory_curve === undefined) {
        trajectory_curve = new TrajectoryCurve({
          positions: camera_path_trajectory.positions,
          color: trajectory_curve_color,
        });
        viewer.threejs_renderer.scene.add(trajectory_curve);
      } else {
        // Update trajectory
        trajectory_curve.setPositions(camera_path_trajectory.positions);
      }
    }
  });
}


function _attach_player_frustum(viewer) {
  let player_frustum = undefined;

  viewer.addEventListener("change", ({ property, state }) => {
    if (property !== undefined &&
        property !== 'camera_path_trajectory' &&
        property !== 'preview_frame' &&
        property !== 'render_resolution_1' &&
        property !== 'render_resolution_2') return;
    const { 
      camera_path_trajectory,
      preview_frame,
      render_resolution_1,
      render_resolution_2,
    } = state;

    if (!camera_path_trajectory || camera_path_trajectory.positions.length === 0) {
      if (player_frustum !== undefined) {
        viewer.threejs_renderer.scene.remove(player_frustum);
        player_frustum.dispose();
        player_frustum = undefined;
      }
      return;
    }
    const { positions, quaternions, fovs } = camera_path_trajectory;
    const num_frames = positions.length;
    const frame = Math.min(Math.max(0, Math.floor(preview_frame)), num_frames - 1);
    const position = new THREE.Vector3().copy(positions[frame]);
    const quaternion = new THREE.Quaternion().copy(quaternions[frame]);
    const fov = fovs[frame];
    if (player_frustum) {
      viewer.threejs_renderer.scene.remove(player_frustum);
      player_frustum.dispose();
      player_frustum = undefined;
    }

    player_frustum = new CameraFrustum({ 
      fov: fov,
      aspect: render_resolution_1 / render_resolution_2,
      position,
      quaternion,
      scale: 0.1,
      color: player_frustum_color,
    });
    viewer.threejs_renderer.scene.add(player_frustum);
  });
}


class Viewer extends THREE.EventDispatcher {
  constructor(viewport) {
    super();
    this.threejs_renderer = new ThreeJSRenderer(viewport);
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

    this._gui_state = state;
    this._gui_state.camera_path_keyframes = [];

    this._camera_path = null;

    this._trajectory_curve = undefined;
    this._keyframe_frustums = {};
    this._player_frustum = undefined;

    this._on_camera_path_change_callbacks = [];
    this._on_gui_change_callbacks = [];

    this._attach_computed_properties();
    _attach_camera_path(this);
    this._attach_preview_is_playing();
    this._attach_update_preview_mode();
    _attach_camera_path_curve(this);
    _attach_camera_path_keyframes(this);
    _attach_camera_path_selected_keyframe_pivot_controls(this);
    _attach_player_frustum(this);
  }

  notifyChange(props) {
    this.dispatchEvent({ 
      ...props,
      type: "change", 
      state: this._gui_state 
    });
  }

  clear_keyframes() {
    this._gui_state.camera_path_selected_keyframe = undefined;
    this.notifyChange({ property: "camera_path_selected_keyframe" });
    this._gui_state.camera_path_keyframes = [];
    this.notifyChange({ property: "camera_path_keyframes" });
  }

  add_keyframe() {
    const { quaternion, position } = this.threejs_renderer.getCameraPose();
    const id = _keyframe_counter++;
    this._gui_state.camera_path_keyframes.push({
      id,
      quaternion,
      position,
    });
    this.notifyChange({ property: "camera_path_keyframes" });
    this._gui_state.camera_path_selected_keyframe = id;
    this.notifyChange({ property: "camera_path_selected_keyframe" });
  }

  clear_selected_keyframe() {
    this._gui_state.camera_path_selected_keyframe = undefined;
    this.notifyChange({ property: "camera_path_selected_keyframe" });
  }

  _attach_computed_properties() {
    this.addEventListener("change", ({ property, state }) => {
      if (property === undefined || property === 'camera_path_selected_keyframe') {
        state.camera_path_selected_keyframe_natural_index = 
          state.camera_path_keyframes.findIndex((keyframe) => keyframe.id === state.camera_path_selected_keyframe) + 1;
        this.notifyChange({ property: 'camera_path_selected_keyframe_natural_index' });
        state.camera_path_has_selected_keyframe = state.camera_path_selected_keyframe !== undefined;
        this.notifyChange({ property: 'camera_path_has_selected_keyframe' });
      }
    });
  }

  _attach_update_preview_mode() {
    this.addEventListener('change', ({ property, state }) => {
      if (property !== undefined && property !== 'preview_is_preview_mode') return;
      const { 
        preview_is_preview_mode,
      } = state;
      this.threejs_renderer.set_enabled(!preview_is_preview_mode);
      this.threejs_renderer.renderer.domElement.style.display = preview_is_preview_mode ? "none" : "block";
      this._preview_canvas.style.display = preview_is_preview_mode ? "block" : "none";
    });
  }

  _attach_preview_is_playing() {
    let preview_interval;
    this.addEventListener('change', ({ property, state }) => {
      if (property !== undefined &&
          property !== 'camera_path_trajectory' &&
          property !== 'camera_path_framerate' &&
          property !== 'camera_path_interpolation' &&
          property !== 'camera_path_default_transition_duration' &&
          property !== 'preview_is_playing') return;
      const {
        camera_path_trajectory,
        camera_path_framerate,
        camera_path_interpolation,
        camera_path_default_transition_duration,
        preview_is_playing,
      } = state;

      // Add preview timer
      if (preview_interval) {
        clearInterval(preview_interval);
        preview_interval = null;
      }

      if (preview_is_playing) {
        const fps = camera_path_interpolation === 'none' ? 1 / camera_path_default_transition_duration : camera_path_framerate;
        const n = camera_path_trajectory ? camera_path_trajectory.positions.length : 0;
        preview_interval = setInterval(() => {
          state.preview_frame = n > 0 ? (state.preview_frame + 1) % n : 0;
          this.notifyChange({ property: 'preview_frame', trigger: 'preview' });
        }, 1000 / fps);
      }
    });
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
        this.notifyChange({ property: name, trigger: 'gui_change' });
      });
      element.addEventListener("input", (event) => {
        state[name] = getValue(event.target);
        this.notifyChange({ property: name, trigger: 'gui_changing' });
        if (name === "preview_frame" && type === "range") {
          // Changing preview_frame stops the preview
          state.preview_is_playing = false;
          this.notifyChange({ property: "preview_is_playing", trigger: 'preview_frame_change' });
        }
      });
      const { name, value, type, checked } = element;
      state[name] = getValue(element);
      if (name === "preview_frame" && type === "range") {
        // Update maximum when the camera path changes
        this.addEventListener("change", ({ property, trigger, state }) => {
          if (property !== undefined && property !== 'camera_path_trajectory') return;
          const { camera_path_trajectory } = state;
          element.max = camera_path_trajectory ? (camera_path_trajectory.positions.length - 1) : 0;
          const old_value = state[name];
          const new_value = Math.min(element.value, element.max);
          if (old_value !== new_value) {
            element.value = new_value;
            element.dispatchEvent(new Event("input"));
          }
        });
      }
      this.addEventListener("change", ({ property, trigger }) => {
        if (property !== name) return;
        if (trigger === 'init' || trigger === 'gui_change' || trigger === 'gui_changing') return;
        setValue(element, state[name]);
      });
    });
    root.querySelectorAll("[data-bind]").forEach(element => {
      const name = element.getAttribute("data-bind");
      this.addEventListener("change", ({ property, trigger }) => {
        if (property !== name && property !== undefined) return;
        element.textContent = state[name];
      });
    });

    root.querySelectorAll("[data-enable-if]").forEach(element => {
      const name = element.getAttribute("data-enable-if");
      this.addEventListener("change", ({ property, state }) => {
        if (property !== name && property !== undefined) return;
        element.disabled = !state[name];
      });
    });

    root.querySelectorAll("[data-action]").forEach(element => {
      const action = element.getAttribute("data-action");
      element.addEventListener("click", () => {
        this[action]();
      });
    });

    // data-bind-class has the form "class1:property1"
    root.querySelectorAll("[data-bind-class]").forEach(element => {
      const [class_name, name] = element.getAttribute("data-bind-class").split(":");
      this.addEventListener("change", ({ property, trigger }) => {
        if (property !== name && property !== undefined) return;
        if (state[name]) {
          element.classList.add(class_name);
        } else {
          element.classList.remove(class_name);
        }
      });
    });
    this.notifyChange({ property: null, trigger: 'init' });
  }

  _draw_background() {
    if (!this._gui_state.preview_is_preview_mode) {
      this.threejs_renderer._drawBackground();
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
document.getElementById("button_clear_keyframes").addEventListener("click", () => viewer.clear_keyframes());
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


class DatasetManager {
  constructor({
    viewer,
    url,
  }) {
    this.viewer = viewer;
    this.scene = viewer.threejs_renderer.scene;
    this.url = url;

    this._train_cameras = new THREE.Group();
    this._train_cameras.visible = !!state.dataset_show_train_cameras;
    this._test_cameras = new THREE.Group();
    this._test_cameras.visible = !!state.dataset_show_test_cameras;
    this._pointcloud = new THREE.Group();
    this._pointcloud.visible = !!state.dataset_show_pointcloud;
    this.scene.add(this._train_cameras);
    this.scene.add(this._test_cameras);
    this.scene.add(this._pointcloud);
    viewer.addEventListener("change", ({ property, state }) => {
      if (property === undefined || property === 'dataset_show_pointcloud')
        this._pointcloud.visible = state.dataset_show_pointcloud;
      if (property === undefined || property === 'dataset_show_train_cameras')
        this._train_cameras.visible = state.dataset_show_train_cameras;
      if (property === undefined || property === 'dataset_show_test_cameras')
        this._test_cameras.visible = state.dataset_show_test_cameras;
    });

    this._load_cameras("test");
    this._load_cameras("train");
    this._load_pointcloud();
  }

  _load_cameras(split) {
    // Load dataset train/test frustums
    const trainCamerasLoader = new THREE.FileLoader();
    trainCamerasLoader.setResponseType('json'); // Ensures the result is parsed as JSON
    trainCamerasLoader.load(
      `${this.url}/${split}.json`,
      (result) => {
        const { cameras } = result;
        this.viewer._gui_state[`dataset_has_${split}_cameras`] = true;
        this.viewer.notifyChange({ property: `dataset_has_${split}_cameras` });
        let i = 0;
        for (const camera of cameras) {
          const pose = camera.pose; // Assuming pose is a flat array representing a 3x4 matrix
          if (pose.length !== 12) {
            console.error('Invalid pose array. Expected 12 elements for 3x4 matrix.');
            continue;
          }

          const poseMatrix = new THREE.Matrix4();
          poseMatrix.set(
            pose[0], pose[1], pose[2], pose[3],
            pose[4], pose[5], pose[6], pose[7],
            pose[8], pose[9], pose[10], pose[11],
            0, 0, 0, 1 // Add the last row to make it a full 4x4 matrix
          );

          // Optional: Decompose the pose matrix into position, quaternion, and scale
          const position = new THREE.Vector3();
          const quaternion = new THREE.Quaternion();
          const scale = new THREE.Vector3();
          poseMatrix.decompose(position, quaternion, scale);

          const [fx, fy, cx, cy] = camera.intrinsics;
          const [width, height] = camera.image_size;

          const frustum = new CameraFrustum({ 
            fov: Math.atan2(cy, fy) * 2,
            aspect: width / height,
            position,
            quaternion,
            scale: 0.1,
            color: dataset_frustum_color,
            hasImage: true,
          });

          // Replace image_path extension with .jpg
          new THREE.TextureLoader().load(`${this.url}/thumbnails/${split}/${i}.jpg`, (texture) => {
            frustum.setImageTexture(texture);
          });
          this[`_${split}_cameras`].add(frustum);
          i++;
        }
      },
      undefined,
      (error) => {
        console.error('An error occurred while loading the cameras:', error);
      }
    );
  }

  _load_pointcloud() {
    // Load PLY file
    const loader = new PLYLoader();
    loader.load(`${this.url}/pointcloud.ply`, (geometry) => {
      this.viewer._gui_state.dataset_has_pointcloud = true;
      this.viewer.notifyChange({ property: 'dataset_has_pointcloud' });
      geometry.setAttribute('color', geometry.getAttribute('color'));
      geometry.computeBoundingSphere();

      const material = new THREE.ShaderMaterial({ 
        uniforms: {
          scale: { value: 10.0 },
          point_ball_norm: { value: 2.0 },
        },
        vertexShader: _point_cloud_vertex_shader,
        fragmentShader: _point_cloud_fragment_shader,
      });
      material.vertexColors = geometry.hasAttribute('color');

      const points = new THREE.Points(geometry, material);
      this._pointcloud.add(points);
    }, undefined, (error) => {
        console.error('An error occurred while loading the PLY file:', error);
    });
  }
}

const dataset_manager = new DatasetManager({ viewer, url: "http://localhost:5001/dataset" });
