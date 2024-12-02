import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { compute_camera_path } from './interpolation.js';
import { PivotControls, MouseInteractions, CameraFrustum, TrajectoryCurve } from './threejs_utils.js';

import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
//import { FirstPersonControls } from 'three/addons/controls/FirstPersonControls.js';
import { FlyControls } from 'three/addons/controls/FlyControls.js';


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

const _R_threecam_cam = new THREE.Matrix4().makeRotationX(Math.PI);


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
        viewer.scene.remove(pivot_controls);
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
      viewer.scene.add(pivot_controls);
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
      const fov = render_fov;
      const aspect = render_resolution_1 / render_resolution_2;
      let frustum = keyframe_frustums[keyframe.id];

      if (frustum === undefined) {
        frustum = new CameraFrustum({ 
          fov,
          aspect,
          position: keyframe.position.clone(), 
          quaternion: keyframe.quaternion.clone(),
          scale: 0.1,
          color: keyframe_frustum_color,
          interactive: true,
          originSphereScale: 0.12,
        });
        frustum.addEventListener("click", () => {
          // If dataset frustum is selected, clear the selection
          if (viewer._gui_state.dataset_selected_image_id !== undefined) {
            viewer._gui_state.dataset_selected_image_id = undefined;
            viewer.notifyChange({ property: 'dataset_selected_image_id' });
          }

          frustum.focused = true;
          viewer._gui_state.camera_path_selected_keyframe = keyframe.id;
          viewer.notifyChange({ property: "camera_path_selected_keyframe" });
        });
        viewer.scene.add(frustum);
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
        viewer.scene.remove(keyframe_frustums[keyframe_id]);
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
        viewer.scene.remove(trajectory_curve);
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
        viewer.scene.add(trajectory_curve);
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
        viewer.scene.remove(player_frustum);
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
      viewer.scene.remove(player_frustum);
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
    viewer.scene.add(player_frustum);
  });
}


class HTTPRenderer extends THREE.EventDispatcher {
  constructor({ url, get_camera_params }) {
    super();
    this.url = url;
    this.get_camera_params = get_camera_params;
    this._num_errors = 0;
    this._running = true;
  }

  start() {
    let lastUpdate = Date.now() - 1;
    let lastParams = undefined;

    const _updateSingle = async (next) => {
      try {
        const width = viewport.clientWidth;
        const height = viewport.clientHeight;
        let {
          matrix,
          fov,
        } = this.get_camera_params();
        const round = (x) => Math.round(x * 100000) / 100000;
        const camera_pose = matrix4ToArray(matrix).map(round).join(",");
        // Fov is in radians
        const focal = height / (2 * Math.tan(THREE.MathUtils.degToRad(fov) / 2));
        const intrinsics = [focal, focal, width/2, height/2].map(round).join(",");
        const params = `pose=${camera_pose}&intrinsics=${intrinsics}&image_size=${width},${height}`
        if (params === lastParams) return;
        lastParams = params;
        const response = await fetch(`${this.url}?${params}`,{
          method: "POST",  // Disable caching
        });
        // Read response as blob
        const blob = await response.blob();
        const blobSrc = URL.createObjectURL(blob);
        const image = new Image();
        await new Promise((resolve, reject) => {
          image.onload = () => {
            resolve();
          };
          image.onerror = (error) => {
            URL.revokeObjectURL(blobSrc);
            reject(error);
          };
          image.src = blobSrc;
        });
        this._num_errors = 0;
        if (!this._running) return;
        this.dispatchEvent({ type: "frame", image });
        URL.revokeObjectURL(blobSrc);
      } catch (error) {
        this._num_errors++;
        console.error("Error fetching image:", error);
        if (!this._running) return;
        this.dispatchEvent({ type: "error", error });
      }
    };

    const run = () => {
      _updateSingle().then(() => {
        if (this._running && this._num_errors < 10) {
          const wait = Math.max(0, 1000 / 30 - (Date.now() - lastUpdate));
          lastUpdate = Date.now();
          setTimeout(() => run(), wait);
        }
      });
    };
    run();
  }

  dispose() {
    this._running = false;
  }
}


class DatasetManager {
  constructor({
    viewer,
    url,
  }) {
    this.viewer = viewer;
    this.scene = viewer.scene;
    this.url = url;
    this._disposed = true;

    const state = viewer._gui_state;
    this._train_cameras = new THREE.Group();
    this._test_cameras = new THREE.Group();
    this._pointcloud = new THREE.Group();
    this._update_gui(state);
    this.scene.add(this._train_cameras);
    this.scene.add(this._test_cameras);
    this.scene.add(this._pointcloud);
    this._disposed = false;
    this._on_viewer_change = this._on_viewer_change.bind(this);
    viewer.addEventListener("change", this._on_viewer_change);

    this._frustums = {};
    this._load_cameras("test");
    this._load_cameras("train");
    this._load_pointcloud();
  }

  _update_gui({ 
    dataset_show_pointcloud, 
    dataset_show_train_cameras, 
    dataset_show_test_cameras,
    dataset_selected_image_id,
  }) {
    this._pointcloud.visible = !!dataset_show_pointcloud;
    this._train_cameras.visible = !!dataset_show_train_cameras;
    this._test_cameras.visible = !!dataset_show_test_cameras;

    // Update frustums visibility
    for (const frustum_id in this._frustums) {
      const frustum = this._frustums[frustum_id];
      frustum.focused = frustum_id === state.dataset_selected_image_id;
    }

    // Clear selected camera if it is not visible
    if (!dataset_show_train_cameras && dataset_selected_image_id?.startsWith('train')) {
      this.viewer._gui_state.dataset_selected_image_id = undefined;
      this.viewer.notifyChange({ property: 'dataset_selected_image_id' });
    }
    if (!dataset_show_test_cameras && dataset_selected_image_id?.startsWith('test')) {
      this.viewer._gui_state.dataset_selected_image_id = undefined;
      this.viewer.notifyChange({ property: 'dataset_selected_image_id' });
    }
  }

  _on_viewer_change({ property, state }) {
    if (property === undefined || 
        property === 'dataset_show_pointcloud' ||
        property === 'dataset_show_train_cameras' ||
        property === 'dataset_show_test_cameras' ||
        property === 'dataset_selected_image_id') {
      this._update_gui(state);
    }
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this.viewer.removeEventListener("change", this._on_viewer_change);
    this.scene.remove(this._train_cameras);
    this.scene.remove(this._test_cameras);
    this.scene.remove(this._pointcloud);
    for (const frustum of this._frustums) {
      frustum.dispose();
    }
  }

  _load_cameras(split) {
    // Load dataset train/test frustums
    const trainCamerasLoader = new THREE.FileLoader();
    trainCamerasLoader.setResponseType('json'); // Ensures the result is parsed as JSON
    trainCamerasLoader.load(
      `${this.url}/${split}.json`,
      (result) => {
        if (this._disposed) return;
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
          const fov = THREE.MathUtils.radToDeg(2 * Math.atan2(cy, fy));
          const aspect = width / height;

          const frustum = new CameraFrustum({ 
            fov,
            aspect,
            position,
            quaternion,
            scale: 0.1,
            color: dataset_frustum_color,
            hasImage: true,
            interactive: true,
          });
          const id = `${split}/${i}`;
          this._frustums[id] = frustum;
          frustum.addEventListener("click", () => {
            // If camera path frustum is selected, clear the selection
            if (this.viewer._gui_state.camera_path_selected_keyframe !== undefined) {
              this.viewer._gui_state.camera_path_selected_keyframe = undefined;
              this.viewer.notifyChange({ property: 'camera_path_selected_keyframe' });
            }

            frustum.focused = true;
            this.viewer._gui_state.dataset_selected_image_id = id;
            this.viewer.notifyChange({ property: 'dataset_selected_image_id' });
          });

          // Replace image_path extension with .jpg
          new THREE.TextureLoader().load(`${this.url}/images/${split}/${i}.jpg?size=64`, (texture) => {
            texture.colorSpace = THREE.SRGBColorSpace;
            frustum.setImageTexture(texture);
          });
          this[`_${split}_cameras`].add(frustum);
          this.viewer._gui_state.dataset_images = this.viewer._gui_state.dataset_images || {};
          this.viewer._gui_state.dataset_images[id] = {
            id,
            index: i,
            split,
            image_name: camera.image_name,
            image_url: `${this.url}/images/${split}/${i}.jpg`,
          };
          i++;
        }
        this.viewer.notifyChange({ property: 'dataset_images' });
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
      if (this._disposed) return;
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


class Viewer extends THREE.EventDispatcher {
  constructor({ 
    viewport, 
    viewer_transform,
    viewer_initial_pose,
  }) {
    super();
    this._backgroundTexture = null;
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    viewport.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera( 70, width / height, 0.01, 10 );
    this.camera.position.z = 1;

    this.renderer_scene = new THREE.Scene();

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.listenToKeyEvents(window);
    this.controls.enableDamping = true; // an animation loop is required when either damping or auto-rotation are enabled
   	this.controls.dampingFactor = 0.05;
		this.controls.screenSpacePanning = false;
		this.controls.maxPolarAngle = Math.PI / 2;

    this._enabled = true;
    this.renderer.setAnimationLoop((time) => this._animate(time));
    window.addEventListener("resize", () => this._resize());

    this.mouse_interactions = new MouseInteractions(this.renderer, this.camera, this.renderer_scene);

    this.scene = new THREE.Group();
    this.renderer_scene.add(this.scene);
    if (viewer_transform) {
      this.scene.applyMatrix4(viewer_transform);
    }
    // Switch OpenCV to ThreeJS coordinate system
    this.scene.applyMatrix4(new THREE.Matrix4().makeRotationX(-Math.PI/2));
    // if (viewer_initial_pose) {
    //   this.camera.applyMatrix4(viewer_initial_pose);
    // }


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
    this._is_preview_mode = false;

    this._camera_path = null;

    this._trajectory_curve = undefined;
    this._keyframe_frustums = {};
    this._player_frustum = undefined;
    this._last_frames = {};

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

  clear_selected_dataset_image() {
    this._gui_state.dataset_selected_image_id = undefined;
    this.notifyChange({ property: 'dataset_selected_image_id' });
  }

  set_http_renderer(url) {
    if (this.http_renderer) {
      this.http_renderer.dispose();
      this.http_renderer = null;
    }

    // Connect HTTP renderer
    this.http_renderer = new HTTPRenderer({ 
      url,
      get_camera_params: () => this._get_camera_params(),
    });
    this.http_renderer.addEventListener("frame", ({ image }) => {
      this._last_frames[0] = image;
      this._draw_background();
    });
    this.http_renderer.start();

    // Update state
    this._gui_state.has_method = true;
    this.notifyChange({ property: 'has_method' });
  }

  set_dataset(url) {
    if (this.dataset_manager) {
      this.dataset_manager.dispose();
      this.dataset_manager = null;
    }
    // Add DatasetManager
    this.dataset_manager = new DatasetManager({ 
      viewer: this, 
      url,
    });
  }

  _resize() {
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this._draw_background();
    this.renderer.setSize(width, height);
  }

  _animate(time) {
    if (!this._is_preview_mode) {
      this.mouse_interactions.update();
      if (!this.mouse_interactions.isCaptured())
        this.controls.update();
      this.renderer.render(this.renderer_scene, this.camera);
    }
  }

  _get_camera_params() {
    const pose = this.camera.matrixWorld.clone();
    pose.multiply(_R_threecam_cam);
    pose.premultiply(this.scene.matrixWorld.clone().invert());
    // const fovRadians = THREE.MathUtils.degToRad(fov);
    const fov = this.camera.fov;
    return {
      matrix: pose,
      fov,
    };
  }

  notifyChange(props) {
    this.dispatchEvent({ 
      ...props,
      type: "change", 
      state: this._gui_state 
    });
  }

  addComputedProperty({ name, getter, dependencies }) {
    this._gui_state[name] = getter(this._gui_state);
    this.addEventListener("change", ({ property, state }) => {
      if (property !== undefined && !dependencies.includes(property)) return;
      state[name] = getter(state);
      this.notifyChange({ property: name });
    });
  }

  delete_all_keyframes() {
    this._gui_state.camera_path_selected_keyframe = undefined;
    this.notifyChange({ property: "camera_path_selected_keyframe" });
    this._gui_state.camera_path_keyframes = [];
    this.notifyChange({ property: "camera_path_keyframes" });
  }

  delete_keyframe(keyframe_id) {
    this._gui_state.camera_path_keyframes = this._gui_state.camera_path_keyframes.filter((keyframe) => keyframe.id !== keyframe_id);
    if (this._gui_state.camera_path_selected_keyframe === keyframe_id) {
      this._gui_state.camera_path_selected_keyframe = undefined;
      this.notifyChange({ property: "camera_path_selected_keyframe" });
    }
    this.notifyChange({ property: "camera_path_keyframes" });
  }

  delete_selected_keyframe() {
    if (this._gui_state.camera_path_selected_keyframe === undefined) return;
    this.delete_keyframe(this._gui_state.camera_path_selected_keyframe);
  }

  add_keyframe() {
    const { matrix } = this._get_camera_params();
    const quaternion = new THREE.Quaternion();
    const position = new THREE.Vector3();
    const scale = new THREE.Vector3();
    matrix.decompose(position, quaternion, scale);
    const id = _keyframe_counter++;
    this._gui_state.camera_path_keyframes.push({
      id,
      quaternion,
      position,
    });
    this.notifyChange({ property: "camera_path_keyframes" });
  }

  clear_selected_keyframe() {
    this._gui_state.camera_path_selected_keyframe = undefined;
    this.notifyChange({ property: "camera_path_selected_keyframe" });
  }

  _attach_computed_properties() {
    this.addComputedProperty({
      name: "camera_path_selected_keyframe_natural_index",
      dependencies: ["camera_path_keyframes", "camera_path_selected_keyframe"],
      getter: ({ camera_path_keyframes, camera_path_selected_keyframe }) => {
        return camera_path_keyframes.findIndex((keyframe) => keyframe.id === camera_path_selected_keyframe) + 1;
      }
    });
    this.addComputedProperty({
      name: "camera_path_has_selected_keyframe",
      dependencies: ["camera_path_selected_keyframe"],
      getter: ({ camera_path_selected_keyframe }) => camera_path_selected_keyframe !== undefined
    });
    this.addComputedProperty({
      name: "has_output_split",
      dependencies: ["has_method", "output_types"],
      getter: ({ has_method, output_types }) => has_method && output_types && output_types.length > 1
    });

    // Add dataset's computed properties
    this.addComputedProperty({
      name: "dataset_has_selected_camera",
      dependencies: ["dataset_selected_image_id"],
      getter: ({ dataset_selected_image_id }) => dataset_selected_image_id !== undefined,
    });

    this.addComputedProperty({
      name: "dataset_selected_image_index",
      dependencies: ["dataset_selected_image_id"],
      getter: ({ dataset_selected_image_id }) => dataset_selected_image_id?.split("/")[1],
    });

    this.addComputedProperty({
      name: "dataset_selected_image_split",
      dependencies: ["dataset_selected_image_id"],
      getter: ({ dataset_selected_image_id }) => dataset_selected_image_id?.split("/")[0],
    });

    this.addComputedProperty({
      name: "dataset_selected_image_name",
      dependencies: ["dataset_images", "dataset_selected_image_id"],
      getter: ({ dataset_images, dataset_selected_image_id }) => dataset_images?.[dataset_selected_image_id]?.image_name || "",
    });

    this.addComputedProperty({
      name: "dataset_selected_image_url",
      dependencies: ["dataset_images", "dataset_selected_image_id"],
      getter: ({ dataset_images, dataset_selected_image_id }) => dataset_images?.[dataset_selected_image_id]?.image_url || "",
    });
  }

  _attach_update_preview_mode() {
    this.addEventListener('change', ({ property, state }) => {
      if (property !== undefined && property !== 'preview_is_preview_mode') return;
      const { 
        preview_is_preview_mode,
      } = state;
      this._is_preview_mode = preview_is_preview_mode;
      this.controls.enabled = !this._is_preview_mode;
      this.renderer.domElement.style.display = this._is_preview_mode ? "none" : "block";
      this._preview_canvas.style.display = this._is_preview_mode ? "block" : "none";
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
      if (state[name] === undefined) {
        state[name] = getValue(element);
      } else {
        setValue(element, state[name]);
      }
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

    function bindOptionsOutputTypes(element) {
      function updateOptions() {
        const selectedValue = state[element.name];
        // Clear all options
        const oldOptions = {};
        Array.from(element.children).forEach((option) => {
          oldOptions[option.value] = option;
          element.removeChild(option);
        });

        // Add new options
        const output_types = state.output_types || ["not set"];
        for (const value of output_types) {
          const option = oldOptions[value] || document.createElement("option");
          option.value = value;
          option.textContent = value;
          if (value === selectedValue) option.selected = true;
          element.appendChild(option);
        }
      }

      this.addEventListener("change", ({ property, trigger, state }) => {
        if (property !== "output_types" && property !== undefined) return;
        updateOptions();
      });

      updateOptions();
    }
    document.getElementsByName("output_type").forEach(bindOptionsOutputTypes.bind(this));
    document.getElementsByName("split_output_type").forEach(bindOptionsOutputTypes.bind(this));

    root.querySelectorAll("select[data-bind-options]").forEach(element => {
      const name = element.getAttribute("data-bind-options");
      this.addEventListener("change", ({ property, trigger }) => {
        if (property !== name && property !== undefined) return;
        // Update options
        // Clear all options
        const oldOptions = {};
        element.childNodes.forEach((option) => {
          oldOptions[option.value] = option;
          element.removeChild(option);
        });

        // Add new options
        for (const value of state[name]) {
          const option = oldOptions[value] || document.createElement("option");
          option.value = value;
          option.textContent = value;
          element.appendChild(option);
        }
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

    // data-bind-attr has the form "attribute:property"
    root.querySelectorAll("[data-bind-attr]").forEach(element => {
      const [attr, name] = element.getAttribute("data-bind-attr").split(":");
      this.addEventListener("change", ({ property, trigger }) => {
        if (property !== name && property !== undefined) return;
        element.setAttribute(attr, state[name]);
      });
      if (state[name] !== undefined)
        element.setAttribute(attr, state[name]);
    });

    this.notifyChange({ property: null, trigger: 'init' });
  }

  _draw_background() {
    if (!this._gui_state.preview_is_preview_mode) {
      if (this._backgroundTexture === null) {
        this._backgroundTexture = new THREE.Texture(this._last_frames[0]);
        this._backgroundTexture.colorSpace = THREE.SRGBColorSpace;
        this.renderer_scene.background = this._backgroundTexture;
      } else if (this._backgroundTexture.image.width !== this._last_frames[0].width || this._backgroundTexture.image.height !== this._last_frames[0].height) {
        // Dispose the old texture
        this._backgroundTexture.dispose();
        this._backgroundTexture = new THREE.Texture(this._last_frames[0]);
        this._backgroundTexture.colorSpace = THREE.SRGBColorSpace;
        this.renderer_scene.background = this._backgroundTexture;
      } else {
        this._backgroundTexture.image = this._last_frames[0];
      }
      this._backgroundTexture.needsUpdate = true;
    } else {
      // Manually draw the background to the canvas
      const image = this._last_frames[0];
      if (image === undefined) return;
      const { width, height } = this._preview_canvas;
      this._preview_context.drawImage(image, 0, 0, width, height);
    }
  }
}


// Attach GUI
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


function makeMatrix4(elements) {
  if (!elements || elements.length !== 12) {
    raise("Invalid elements array. Expected 12 elements.");
  }
  return new THREE.Matrix4().set(
    elements[0], elements[1], elements[2], elements[3],
    elements[4], elements[5], elements[6], elements[7],
    elements[8], elements[9], elements[10], elements[11],
    0, 0, 0, 1,
  );
}

function matrix4ToArray(matrix) {
  const e = matrix.elements;
  return [e[0], e[4], e[8], e[12], e[1], e[5], e[9], e[13], e[2], e[6], e[10], e[14]];
}

fetch("http://localhost:5001/info")
  .then(response => response.json())
  .then(data => {
    let viewer_transform = undefined;
    let viewer_initial_pose = undefined;
    if (data.viewer_transform) {
      viewer_transform = makeMatrix4(data.viewer_transform);
    }
    if (data.viewer_initial_pose) {
      viewer_initial_pose = makeMatrix4(data.viewer_initial_pose);
    }

    data.output_types = ["color", "depth"];
    state.output_types = data.output_types || [];
    state.output_type = (state.output_types || state.output_types.length > 0) ? state.output_types[0] : "";
    state.split_output_type = (state.output_types || state.output_types.length > 1) ? state.output_types[1] : "";

    const viewer = new Viewer({
      viewport,
      viewer_transform,
      viewer_initial_pose,
    });
    viewer.attach_gui(document.querySelector('.controls'));
    viewer.set_http_renderer("http://localhost:5001/render");
    viewer.set_dataset("http://localhost:5001/dataset");
  });
