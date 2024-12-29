import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { compute_camera_path } from './interpolation.js';
import { PivotControls, MouseInteractions, CameraFrustum, TrajectoryCurve } from './threejs_utils.js';
import { ViewerControls } from './controls.js';
import * as Mp4Muxer from 'mp4-muxer';
import * as WebMMuxer from 'webm-muxer';
import { downloadZip } from 'client-zip';


const notification_autoclose = 5000;


class VideoWriter {
  constructor({ 
    width, height, 
    fps=30, 
    filename="rendering", 
    type="mp4",
    numFrames=undefined,
    mp4Codec='avc1.42001f',
    webmCodec='vp09.00.10.08',
    keyframeInterval='5s',
  }) {
    this.fps = fps;
    this.width = width;
    this.height = height;
    this.numFrames = numFrames;

    this.extension = type;
    this.type = type;
    this.filename = filename;
    this.mimeType = `video/${type}`;
    if (type === 'zip')
      this.mimeType = 'application/zip';
    this.codec = type === "mp4" ? mp4Codec : webmCodec;
    this.keyframeInterval = keyframeInterval;
  }

  static getCodecName(codec, type) {
    if (type === "mp4") {
      if (codec.startsWith('av01.')) return 'av1';
      if (codec.startsWith('avc1.')) return 'avc';
      if (codec.startsWith('avc3.')) return 'avc';
      if (codec.startsWith('hev1.')) return 'hevc';
      if (codec.startsWith('hvc1.')) return 'hevc';
      if (codec.startsWith('vp09.')) return 'vp9';
      throw `Unknown codec: ${codec}`;
    } else if (type === "webm") {
      if (codec.startsWith('av01.')) return 'V_AV1';
      if (codec === 'vp8') return 'V_VP8';
      if (codec.startsWith('vp09.')) return 'V_VP9';
      throw `Unknown codec: ${codec}`;
    }
  }

  _makeInMemoryEncoder(saveBlob) {
    if (this.type === "zip") {
      return this._makeZipEncoder({
        finalize: async (blob) => saveBlob(blob),
      });
    }
    const module = this.type === "mp4" ? Mp4Muxer : WebMMuxer;
    return this._makeVideoEncoder({
      module,
      target: new module.ArrayBufferTarget(),
      close: async () => undefined,
      finalize: async (muxer) => {
        saveBlob(new Blob([muxer.target.buffer], { type: this.mimeType }));
      },
    });
  }

  async _makeFileEncoder(fileHandle) {
    const stream = await fileHandle.createWritable();
    if (this.type === "zip") {
      return this._makeZipEncoder({
        finalize: async (blob) => {
          await stream.write(blob);
          await stream.close();
        }
      });
    }
    const module = this.type === "mp4" ? Mp4Muxer : WebMMuxer;
    return this._makeVideoEncoder({
      module,
      target: new module.FileSystemWritableFileStreamTarget(stream),
      close: () => stream.close(),
      finalize: () => stream.close(),
    });
  }

  async _makeZipEncoder({ finalize }) {
    const files = [];
    const canvas = new OffscreenCanvas(this.width, this.height);
    const ctx = canvas.getContext('bitmaprenderer');
    let keyframeCounter = 0;
    const pad = (num, size) => ('000000000' + num).substr(-size);
    this.addFrame = async (image) => {
      ctx.transferFromImageBitmap(image);
      const blob = await canvas.convertToBlob({ type: 'image/png' });
      files.push({
        name: `${pad(++keyframeCounter, 5)}.png`,
        lastModified: new Date(), 
        input: blob,
      });
      image.close();
    };

    this.close = async () => {}
    this.finalize = async () => {
      const blob = await downloadZip(files).blob()
      await finalize(blob);
    }
  }

  async saveAs() {
    const hasFSAccess = 'showOpenFilePicker' in window;
    const fallback = () => {
      const saveBlob = (blob) => {
        var a = document.createElementNS('http://www.w3.org/1999/xhtml', 'a')
        a.setAttribute('download', `${this.filename}.${this.extension}`);
        a.setAttribute('rel', 'noopener'); // tabnabbing
        a.setAttribute('href', URL.createObjectURL(blob));
        setTimeout(function () { URL.revokeObjectURL(a.href) }, 4E4); // 40s
        setTimeout(function () { a.click() }, 0);
      };
      this._makeInMemoryEncoder(saveBlob);
    };
    if (!hasFSAccess) {
      fallback();
      return
    }

    // Create file handle.
    let fileHandle;
    try {
      const opts = {
        startIn: 'downloads',
        suggestedName: `${this.filename}.${this.extension}`,
        types: [{
          accept: { [this.mimeType]: [`.${this.extension}`]},
        }],
      };
      fileHandle = await window.showSaveFilePicker(opts);
    } catch (ex) {
      throw ex;
      console.error(ex);
      console.log("Error saving file, falling back to download.");
      fallback();
      return;
    }

    await this._makeFileEncoder(fileHandle);
  }

  _makeVideoEncoder({ module, target, close, finalize }) {
    let errors = [];
    let framesGenerated = 0;
    const inMemory = target instanceof module.ArrayBufferTarget;
    const muxer = new module.Muxer({
      target,
      video: {
        codec: VideoWriter.getCodecName(this.codec, this.type),
        width: this.width,
        height: this.height,
        frameRate: this.fps,
      },
      audio: undefined,
      fastStart: (inMemory || !this.numFrames || this.numFrames < 1) ? 'in-memory' : {
        expectedVideoChunks: this.numFrames,
      },
      firstTimestampBehavior: 'offset'
    });
    const videoEncoder = new VideoEncoder({
      output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
      error: e => {
        console.error(e)
        errors.push(e);
      },
    });
    videoEncoder.configure({
      codec: this.codec,
      width: this.width,
      height: this.height,
      bitrate: 1e6,
    });
    let lastKeyframeTimestamp = -Infinity;
    this.addFrame = async (image) => {
      if (errors.length > 0) throw errors[0];
      const timestamp = framesGenerated * 1e6 / this.fps;
      let frame = new VideoFrame(image, {
        timestamp,
        duration: 1e6 / this.fps,
      });
      let keyFrame = false;
      // If keyframeInterval is string and ends with s, it is interpreted as seconds
      if (this.keyframeInterval.endsWith('s')) {
        const timeInterval = parseFloat(this.keyframeInterval.slice(0, -1));
        if (timestamp - lastKeyframeTimestamp >= timeInterval * 1e6) {
          keyFrame = true;
          lastKeyframeTimestamp = timestamp;
        }
      } else {
        if (framesGenerated % parseInt(this.keyframeInterval) === 0) {
          keyFrame = true;
        }
      }
      videoEncoder.encode(frame, { keyFrame });
      frame.close();
      framesGenerated++;
    };
    this.close = async () => {
      try {
        close?.();
      } catch (e) {
        console.error(e);
      }
    }
    this.finalize = async () => {
      let hasError = true;
      try {
        if (errors.length > 0) throw errors[0];
        await videoEncoder?.flush();
        muxer.finalize();

        await finalize?.(muxer);
        hasError = false;
      } finally {
        if (hasError)
          this.close();
      }
    };
  }
}


class SettingsManager {
  constructor({ viewer }) {
    this.viewer = viewer;
    this._populate_state();
    this.viewer.addEventListener("change", this._on_change.bind(this));
  }

  _on_change({ state, property }) {
    const defaultSettings = this._get_default_settings();
    if (property !== undefined && defaultSettings[property] !== undefined) {
      // We store the property to the local cache
      localStorage.setItem(`settings.${property}`, state[property]);
    }
  }

  _get_default_settings() {
    return {
      theme_color: "#ffd369",
      trajectory_curve_color: "#6bffe6",
      player_frustum_color: "#20df80",
      keyframe_frustum_color: "#ff0000",
      dataset_frustum_color: "#d3d3d3",
      notification_autoclose: 5000,
      dataset_show_train_cameras: false,
      dataset_show_test_cameras: true,
      dataset_show_pointcloud: true,

      camera_control_inertia: 0.6,
      camera_control_rotation_sensitivity: 1.0,
      camera_control_pan_sensitivity: 1.0,
      camera_control_zoom_sensitivity: 1.0,
      camera_control_key_speed: 1.0,

      camera_control_rotation_inverted: false,
      camera_control_pan_inverted: false,
      camera_control_zoom_inverted: false,

      camera_path_render_mp4_codec: 'avc1.42001f',
      camera_path_render_webm_codec: 'vp09.00.10.08',
      camera_path_render_keyframe_interval: '5s',

      viewer_theme: 'dark',
      viewer_font_size: 1,
    }
  }

  reset() {
    localStorage.clear();
    Object.assign(this.viewer.state, this._get_default_settings());
    this.viewer.notifyChange({ property: undefined });
  }

  _populate_state() {
    const state = this.viewer.state;
    const settings = this._get_default_settings();
    for (const k in settings) {
      let val = localStorage.getItem(`settings.${k}`);
      if (val === null || val === undefined)
        continue;
      if (typeof settings[k] === 'boolean') {
        val = val === 'true';
      }
      settings[k] = val;
    }
    Object.assign(state, settings);
    this.viewer.notifyChange({ property: undefined });
  }
}


function computeResolution(rendererResolution, maxResolution) {
  if (maxResolution === undefined) {
    return rendererResolution;
  }
  const [width, height] = rendererResolution;
  const aspect = width / height;
  let widthOut = Math.min(width, maxResolution);
  let heightOut = Math.min(height, maxResolution);
  if (widthOut / heightOut > aspect) {
    widthOut = heightOut * aspect;
  } else {
    heightOut = widthOut / aspect;
  }
  return [Math.round(widthOut), Math.round(heightOut)];
}


const hash_cyrb53 = (str, seed = 0) => {
  let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
  for(let i = 0, ch; i < str.length; i++) {
            ch = str.charCodeAt(i);
            h1 = Math.imul(h1 ^ ch, 2654435761);
            h2 = Math.imul(h2 ^ ch, 1597334677);
        }
  h1  = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2  = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);

  return 4294967296 * (2097151 & h2) + (h1 >>> 0);
};


async function saveAs(blob, opts) {
  const hasFSAccess = 'showOpenFilePicker' in window;
  const { type, filename, description, extension } = opts;
  const fallback = () => {
    var a = document.createElementNS('http://www.w3.org/1999/xhtml', 'a')
    a.setAttribute('download', filename);
    a.setAttribute('rel', 'noopener'); // tabnabbing
    a.setAttribute('href', URL.createObjectURL(blob));
    setTimeout(function () { URL.revokeObjectURL(a.href) }, 4E4); // 40s
    setTimeout(function () { a.click() }, 0);
  };
  if (!hasFSAccess) {
    fallback();
    return
  }

  // Create file handle.
  let writable = undefined;
  try {
    // For Chrome 86 and later...
    const opts = {
      startIn: 'downloads',
      suggestedName: filename,
      types: [{
        description,
        accept: { [type]: [`.${extension}`]},
      }],
    };
    writable = await (await window.showSaveFilePicker(opts)).createWritable();
  } catch (ex) {
    if (ex.name === 'AbortError') {
      return;
    }
    console.error(ex);
    console.log("Error saving file, falling back to download.");
    fallback();
    return;
  }

  // For Chrome 83 and later.
  await writable.write(blob);
  await writable.close();
  writable = undefined;
}

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
    keyframe_frustum_color,
    camera_path_default_fov,
    camera_path_resolution_1,
    camera_path_resolution_2,
  }) {
    const new_keyframe_frustums = {};
    for (const keyframe of camera_path_keyframes) {
      const fov = (keyframe?.fov !== undefined && keyframe?.fov !== null) ? keyframe.fov : camera_path_default_fov;
      const aspect = camera_path_resolution_1 / camera_path_resolution_2;
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
          if (viewer.state.dataset_selected_image_id !== undefined) {
            viewer.state.dataset_selected_image_id = undefined;
            viewer.notifyChange({ property: 'dataset_selected_image_id' });
          }

          frustum.focused = true;
          viewer.state.camera_path_selected_keyframe = keyframe.id;
          viewer.notifyChange({ property: "camera_path_selected_keyframe" });
        });
        viewer.scene.add(frustum);
      } else {
        frustum.position.copy(keyframe.position);
        frustum.quaternion.copy(keyframe.quaternion);
        frustum.fov = fov;
        frustum.aspect = aspect;
        frustum.color = keyframe_frustum_color;
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
        property === 'camera_path_default_fov' ||
        property === 'keyframe_frustum_color' ||
        property === 'camera_path_resolution_1' ||
        property === 'camera_path_resolution_2')
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
        property !== 'camera_path_tension' &&
        property !== 'camera_path_continuity' &&
        property !== 'camera_path_default_fov' &&
        property !== 'camera_path_framerate' &&
        property !== 'camera_path_duration' &&
        property !== 'camera_path_bias') return;
    const {
      camera_path_keyframes,
      camera_path_loop,
      camera_path_interpolation,
      camera_path_tension,
      camera_path_continuity,
      camera_path_bias,
      camera_path_default_fov,
      camera_path_duration,
      camera_path_framerate,
    } = state;
    state.camera_path_trajectory = undefined;
    if (camera_path_keyframes) {
      state.camera_path_trajectory = compute_camera_path({
        keyframes: camera_path_keyframes,
        loop: camera_path_loop,
        interpolation: camera_path_interpolation,
        tension: camera_path_tension || 0,
        continuity: camera_path_continuity || 0,
        bias: camera_path_bias || 0,
        default_fov: camera_path_default_fov,
        duration: camera_path_duration,
        framerate: camera_path_framerate,
      });
      if (state?.camera_path_trajectory?.positions?.length === 0)
        state.camera_path_trajectory = undefined;
    }
    viewer.notifyChange({ property: "camera_path_trajectory" });
  });
}

function _attach_camera_path_curve(viewer) {
  let trajectory_curve = undefined;
  viewer.addEventListener("change", ({ property, state }) => {
    if (property !== undefined &&
        property !== 'camera_path_interpolation' &&
        property !== 'camera_path_trajectory' &&
        property !== 'trajectory_curve_color' &&
        property !== 'camera_path_show_spline') return;
    const { 
      camera_path_trajectory, 
      camera_path_show_spline,
      camera_path_interpolation,
      trajectory_curve_color,
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
        trajectory_curve.color = trajectory_curve_color;
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
        property !== 'player_frustum_color' &&
        property !== 'camera_path_resolution_1' &&
        property !== 'camera_path_resolution_2') return;
    const { 
      camera_path_trajectory,
      preview_frame,
      player_frustum_color,
      camera_path_resolution_1,
      camera_path_resolution_2,
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
      aspect: camera_path_resolution_1 / camera_path_resolution_2,
      position,
      quaternion,
      scale: 0.1,
      color: player_frustum_color,
    });
    viewer.scene.add(player_frustum);
  });
}


function _attach_viewport_split_slider(viewer) {
  const viewport = viewer.viewport;
  const div = document.createElement("div");
  div.className = "viewport-slider";
  div.style.position = "absolute";
  div.style.top = "0";
  div.style.left = "50%";
  viewport.appendChild(div);

  let startPoint = undefined;
  div.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    div.setPointerCapture(e.pointerId);
    startPoint = { x: e.clientX, y: e.clientY, split_percentage: viewer.state.split_percentage };
  });
  div.addEventListener("pointermove", (e) => {
    if (!startPoint) return;
    const deltaX = e.clientX - startPoint.x;
    const deltaY = e.clientY - startPoint.y;
    const { split_percentage, split_tilt } = viewer.state;

    // Compute delta split percentage
    const tiltRadians = split_tilt * Math.PI / 180;
    const splitDir = [Math.cos(tiltRadians), Math.sin(tiltRadians)];
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    const splitDirLen = width / 2 * Math.abs(splitDir[0]) + height / 2 * Math.abs(splitDir[1]);
    const absDelta = deltaX * splitDir[0] + deltaY * splitDir[1];
    const delta = absDelta / splitDirLen / 2;

    viewer.state.split_percentage = Math.min(0.95, Math.max(0.05, startPoint.split_percentage + delta));
    viewer.notifyChange({ property: 'split_percentage' });
  });
  div.addEventListener("pointerup", (e) => {
    startPoint = undefined;
    e.preventDefault();
    div.releasePointerCapture(e.pointerId);
  });

  function update({
    split_enabled,
    split_percentage,
    split_tilt,
    preview_is_preview_mode,
  }) {
    div.style.display = (split_enabled && !preview_is_preview_mode) ? "block" : "none";

    // Compute position
    const tiltRadians = split_tilt * Math.PI / 180;
    const splitDir = [Math.cos(tiltRadians), Math.sin(tiltRadians)];
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    const splitDirLen = width / 2 * Math.abs(splitDir[0]) + height / 2 * Math.abs(splitDir[1]);
    const left = width / 2 + splitDir[0] * splitDirLen * (split_percentage*2-1);
    const top = height / 2 + splitDir[1] * splitDirLen * (split_percentage*2-1);
    div.style.left = `${left}px`;
    div.style.top = `${top}px`;
    div.style.transform = `translate(-50%, -50%) rotate(${split_tilt}deg)`;
    div.style.cursor = tiltRadians > Math.PI / 4 && tiltRadians < 3 * Math.PI / 4 ? "ns-resize" : "ew-resize";
  }

  viewer.addEventListener("change", ({ property, state }) => {
    if (property === undefined || 
        property === 'split_enabled' ||
        property === 'split_percentage' ||
        property === 'preview_is_preview_mode' ||
        property === 'split_tilt') {
      update(state);
    }
  });
  viewer.addEventListener("resize", () => update(viewer.state));
  update(viewer.state);
}

function _attach_fullscreen_mode(viewer) {
  function closeFullscreen() {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) { /* Safari */
      document.webkitExitFullscreen();
    } else if (document.msExitFullscreen) { /* IE11 */
      document.msExitFullscreen();
    }
  }
  let viewer_is_embedded = false;
  try {
    viewer_is_embedded = window.self !== window.top;
  } catch (e) {}
  viewer.state.viewer_is_embedded = viewer_is_embedded;
  viewer.notifyChange({ property: "viewer_is_embedded" });
  var elem = document.documentElement;
  function openFullscreen() {
    if (elem.requestFullscreen) {
      elem.requestFullscreen();
    } else if (elem.webkitRequestFullscreen) { /* Safari */
      elem.webkitRequestFullscreen();
    } else if (elem.msRequestFullscreen) { /* IE11 */
      elem.msRequestFullscreen();
    }
  }
  elem.addEventListener("fullscreenchange", () => {
    const newState = !!(document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement);
    if (viewer.state.viewer_fullscreen !== newState) {
      viewer.state.viewer_fullscreen = newState
      viewer.notifyChange({ property: "viewer_fullscreen", origin: elem });
    }
  });
  viewer.addEventListener("change", ({ property, state, origin }) => {
    if (origin === elem) return;
    if (property === "viewer_fullscreen") {
      if (state.viewer_fullscreen) {
        openFullscreen();
      } else {
        closeFullscreen();
      }
    }
  });
  viewer.state.viewer_fullscreen_enabled = document.fullscreenEnabled;
  viewer.notifyChange({ property: "viewer_fullscreen_enabled" });
  viewer.addEventListener("action", ({ action }) => {
    if (action === "viewer_toggle_fullscreen") {
      viewer.state.viewer_fullscreen = !viewer.state.viewer_fullscreen;
      viewer.notifyChange({ property: "viewer_fullscreen" });
    }
    if (action === "viewer_exit_fullscreen") {
      if (viewer.state.viewer_fullscreen) {
        viewer.state.viewer_fullscreen = false;
        viewer.notifyChange({ property: "viewer_fullscreen" });
      }
    }
    if (action === "viewer_enter_fullscreen") {
      if (!viewer.state.viewer_fullscreen) {
        viewer.state.viewer_fullscreen = true;
        viewer.notifyChange({ property: "viewer_fullscreen" });
      }
    }
  });
}


class HTTPFrameRenderer {
  constructor({ url }) {
    this.url = url;
  }

  async render(params, { flipY = false } = {}) {
    const response = await fetch(`${this.url}/render`,{
      method: "POST",  // Disable caching
      cache: "no-cache",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(params),
    });


    // Read response as blob
    const blob = await response.blob();
    return await createImageBitmap(blob, { imageOrientation: flipY ? "flipY" : undefined });
  }
}


export class WebSocketFrameRenderer {
  constructor({ url, update_notification }) {
    this.url = url;
    this._socket = undefined;
    this._subscriptions = {};
    this._thread_counter = 0;
  }

  _on_message(message) {
    const { thread, status } = message;
    const [ resolve, reject ] = this._subscriptions[thread];
    if (thread === undefined || this._subscriptions[thread] === undefined) {
      console.error("Invalid thread id:", thread);
      update_notification({
        header: "Invalid thread id",
        detail: `Received message with invalid thread id: ${thread}`,
        type: "error",
        closeable: true,
      });
      return;
    }
    delete this._subscriptions[thread];
    if (status === "error") {
      reject(message);
    } else {
      resolve(message);
    }
  }

  async _ensure_socket() {
    if (this._socket === undefined) {
      this._socket = await new Promise((resolve, reject) => {
        try {
          const socket = new WebSocket(this.url);
          socket.binaryType = "blob";
          socket.addEventListener("open", () => {
            console.log("WebSocket connection established");
            resolve(socket);
          });
        } catch (error) {
          reject(error);
        }
      });
      this._socket.addEventListener("close", () => {
        console.log("WebSocket connection closed");
        this._socket = undefined;
      });
      this._socket.addEventListener("error", (error) => {
        console.error("WebSocket error:", error);
        update_notification({
          id: "websocket",
          header: "WebSocket error",
          detail: error.message,
          type: "error",
          closeable: true,
        });
      });
      this._socket.addEventListener("message", async (event) => {
        let message;
        if (event.data instanceof Blob) {
          const blob = event.data;
          const headerLength = new DataView(await blob.slice(0, 4).arrayBuffer()).getUint32(0, false);
          message = JSON.parse(await blob.slice(4, 4 + headerLength).text());
          // Read binnary buffers
          if (blob.size > 4 + headerLength) {
            message.payload = blob.slice(4 + headerLength);
          }
        } else {
          message = JSON.parse(event.data);
        }
        this._on_message(message);
      });
    }
  }

  async render(params, { flipY = false } = {}) {
    await this._ensure_socket();
    this._thread_counter++;
    const message = await new Promise((resolve, reject) => {
      this._subscriptions[this._thread_counter] = [resolve, reject];
      this._socket.send(JSON.stringify({
        thread: this._thread_counter,
        ...params,
      }));
    });
    return await createImageBitmap(message.payload, { imageOrientation: flipY ? "flipY" : undefined });
  }
}

async function promise_parallel_n(tasks, num_parallel) {
  await new Promise((resolve, reject) => {
    const queue = [];
    let i = 0;
    let completed = 0;

    const after = (j) => {
      if (i < tasks.length)
        queue[j] = tasks[i++]().finally(() => {
          completed++;
          after(j);
        });
      if (completed === tasks.length) resolve();
    }

    for (let j = 0; j < Math.min(num_parallel, tasks.length); ++j) after(j);
  });
}


class DatasetManager {
  constructor({
    viewer,
    url,
    parts,
  }) {
    this.viewer = viewer;
    this.scene = viewer.scene;
    this.url = url;
    this._disposed = true;

    const state = viewer.state;
    this._train_cameras = new THREE.Group();
    this._test_cameras = new THREE.Group();
    this._pointcloud = new THREE.Group();
    this._update_gui(state);
    this.scene.add(this._train_cameras);
    this.scene.add(this._test_cameras);
    this.scene.add(this._pointcloud);
    this._disposed = false;
    this._on_viewer_change = this._on_viewer_change.bind(this);
    this._frustums = {};
    viewer.addEventListener("change", this._on_viewer_change);
    this._load(parts);
  }

  async _load(parts) {
    if (parts === undefined || parts.includes("pointcloud")) {
      this.viewer.state.dataset_has_pointcloud = true;
      this.viewer.notifyChange({ property: 'dataset_has_pointcloud' });
    }
    if (parts === undefined || parts.includes("test"))
      await this._load_cameras("test");
    if (parts === undefined || parts.includes("train"))
      await this._load_cameras("train");

    // Load images and pointcloud
    if (this.viewer.state.dataset_show_test_cameras && this.viewer.state.dataset_has_test_cameras)
      this._load_split_images("test");
    if (this.viewer.state.dataset_show_train_cameras && this.viewer.state.dataset_has_train_cameras)
      this._load_split_images("train");
    if (this.viewer.state.dataset_show_pointcloud && this.viewer.state.dataset_has_pointcloud)
      this._load_pointcloud();
  }

  _update_gui({ 
    dataset_show_pointcloud, 
    dataset_show_train_cameras, 
    dataset_show_test_cameras,
    dataset_selected_image_id,
    dataset_frustum_color,
  }) {
    this._pointcloud.visible = !!dataset_show_pointcloud;
    this._train_cameras.visible = !!dataset_show_train_cameras;
    this._test_cameras.visible = !!dataset_show_test_cameras;

    // Update frustums visibility
    for (const frustum_id in this._frustums) {
      const frustum = this._frustums[frustum_id];
      frustum.focused = frustum_id === this.viewer.state.dataset_selected_image_id;
      frustum.color = dataset_frustum_color;
    }

    // Clear selected camera if it is not visible
    if (!dataset_show_train_cameras && dataset_selected_image_id?.startsWith('train')) {
      this.viewer.state.dataset_selected_image_id = undefined;
      this.viewer.notifyChange({ property: 'dataset_selected_image_id' });
    }
    if (!dataset_show_test_cameras && dataset_selected_image_id?.startsWith('test')) {
      this.viewer.state.dataset_selected_image_id = undefined;
      this.viewer.notifyChange({ property: 'dataset_selected_image_id' });
    }
  }

  _on_viewer_change({ property, state }) {
    if ((property === undefined || property === 'pointcloud_scale') && this._pointcloud) {
      for (const point of this._pointcloud.children) {
        point.material.uniforms.scale.value = state.pointcloud_scale;
      }
    }
    if (property === undefined || 
        property === 'dataset_show_pointcloud' ||
        property === 'dataset_show_train_cameras' ||
        property === 'dataset_show_test_cameras' ||
        property === 'dataset_frustum_color' ||
        property === 'dataset_selected_image_id') {
      this._update_gui(state);
    }

    for (const split of ['train', 'test']) {
      if (property === `dataset_show_${split}_cameras`) {
        if (state[`dataset_show_${split}_cameras`]) {
          if (state[`dataset_has_${split}_cameras`])
            this._load_split_images(split);
        } else {
          this.viewer._update_notification({
            id: `dataset_${this.url}_${split}_images`, header: "", autoclose: 0,
          });
        }
      }
    }
    if (state[`dataset_has_pointcloud`] && property === 'dataset_show_pointcloud') {
      if (state.dataset_show_pointcloud)
        this._load_pointcloud();
      else if (this._cancel_load_pointcloud)
        this._cancel_load_pointcloud();
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
    this._frustums = {};
  }

  async _load_split_images(split) {
    if (!this.viewer.state[`dataset_has_${split}_cameras`]) return;
    if (!this.viewer.state[`dataset_show_${split}_cameras`]) return;

    try {
      if (this[`_${split}_loading_started`]) return;
      this[`_${split}_loading_started`] = true;
      let errors = [];
      let num_images = this[`_${split}_cameras`].children.length;
      const all_loaded = this[`_${split}_cameras`].children.every((frustum) => frustum._hasImage);
      if (all_loaded) return;
      let tasks = [];
      let num_loaded = 0;
      const cancel = () => {
        this.viewer.state[`dataset_show_${split}_cameras`] = false;
        this.viewer.notifyChange({ property: `dataset_show_${split}_cameras` });
      };
      this.viewer._update_notification({
        id: `dataset_${this.url}_${split}_images`,
        header: `Loading ${split} dataset images`,
        progress: 0,
        onclose: cancel,
        closeable: true,
      });
      for (let _i = 0; _i < num_images; ++_i) {
        const i = _i;
        tasks.push(async () => {
          if (!this.viewer.state[`dataset_show_${split}_cameras`]) return;
          try {
            const frustum = this[`_${split}_cameras`].children[i];
            if (frustum._hasImage) {
              ++num_loaded;
              return;
            }

            // Replace image_path extension with .jpg
            const image_url = `${this.url}/images/${split}/${i}.jpg?size=64`;
            const response = await fetch(image_url);
            if (!response.ok) {
              try {
                const error = await response.json();
                throw new Error(`Failed to load image: ${error.message}`);
              } catch (error) {
                throw new Error(`Failed to load image: ${response.statusText}`);
              }
            }
            const blob = await response.blob();
            const image = await createImageBitmap(blob, { imageOrientation: "flipY" });
            const texture = new THREE.Texture();
            texture.image = image;
            texture.needsUpdate = true;
            texture.colorSpace = THREE.SRGBColorSpace;
            if (!frustum._hasImage)
              frustum.setImageTexture(texture);
          } catch (error) {
            console.error('An error occurred while loading the images:', error);
            errors.push(error);
          }
          if (!this.viewer.state[`dataset_show_${split}_cameras`]) return;
          this.viewer._update_notification({
            id: `dataset_${this.url}_${split}_images`,
            header: `Loading ${split} dataset images`,
            progress: ++num_loaded / num_images,
            onclose: cancel,
            closeable: true,
          });
        });
      }

      await promise_parallel_n(tasks, 8);
      if (!this.viewer.state[`dataset_show_${split}_cameras`]) return;

      if (errors.length > 0) {
        this.viewer._update_notification({
          id: `dataset_${this.url}_${split}_images`,
          header: `Failed to load ${split} dataset images`,
          detail: errors[0].message,
          type: "error",
          closeable: true,
        });
      } else if (num_loaded === num_images) {
        this.viewer._update_notification({
          id: `dataset_${this.url}_${split}_images`,
          header: `Loaded ${split} dataset images`,
          autoclose: notification_autoclose,
          closeable: true,
        });
      }
    } finally {
      this[`_${split}_loading_started`] = false;
    }
  }

  async _load_cameras(split) {
    // Load dataset train/test frustums
    const response = await fetch(`${this.url}/${split}.json`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });
    // TODO: Handle errors
    const result = await response.json();
    // TODO: Handle errors
    try {
      const { cameras } = result;
      this.viewer.state[`dataset_has_${split}_cameras`] = true;
      this.viewer.notifyChange({ property: `dataset_has_${split}_cameras` });
      let i = 0;
      const appearance_options = [{label: "none", value: ""}];
      for (const camera of cameras) {
        const pose = camera.pose; // Assuming pose is a flat array representing a 3x4 matrix
        if (pose.length !== 12) {
          console.error('Invalid pose array. Expected 12 elements for 3x4 matrix.');
          continue;
        }

        const poseMatrix = new THREE.Matrix4();
        poseMatrix.set(...pose, 0, 0, 0, 1); // Add the last row to make it a full 4x4 matrix

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
          color: this.viewer.state.dataset_frustum_color,
          interactive: true,
        });
        const id = `${split}/${i}`;
        this._frustums[id] = frustum;
        frustum.addEventListener("click", () => {
          // If camera path frustum is selected, clear the selection
          if (this.viewer.state.camera_path_selected_keyframe !== undefined) {
            this.viewer.state.camera_path_selected_keyframe = undefined;
            this.viewer.notifyChange({ property: 'camera_path_selected_keyframe' });
          }

          frustum.focused = true;
          this.viewer.state.dataset_selected_image_id = id;
          this.viewer.notifyChange({ property: 'dataset_selected_image_id' });
        });
        this[`_${split}_cameras`].add(frustum);
        this.viewer.state.dataset_images = this.viewer.state.dataset_images || {};
        this.viewer.state.dataset_images[id] = {
          id,
          index: i,
          split,
          image_name: camera.image_name,
          image_url: `${this.url}/images/${split}/${i}.jpg`,
          matrix: poseMatrix,
          width, height,
          cx, cy, fx, fy,
        };
        appearance_options.push({
          value: i,
          label: `${i}: ${camera.image_name}`
        });
        i++;
      }
      this.viewer.notifyChange({ property: 'dataset_images' });
      this.viewer.state[`dataset_${split}_appearance_options`] = appearance_options;
      this.viewer.notifyChange({ property: `dataset_${split}_appearance_options` });
    } catch (error) {
      console.error('An error occurred while loading the cameras:', error);
      this.viewer._update_notification({
        id: this.url + "-" + split,
        header: `Error loading dataset ${split} cameras`,
        detail: error.message,
        type: "error",
      });
      this._update_notification();
    }
  }

  async _load_pointcloud() {
    // Load PLY file
    if (this._cancel_load_pointcloud) return; // Already loading
    if (this._pointcloud.children.length > 0) return; // Already loaded
    const notificationId = this.url + "-pointcloud";
    let cancelled = false;
    const controller = new AbortController();
    this._cancel_load_pointcloud = () => {
      if (!cancelled) {
        controller.abort();
      }
      cancelled = true;
      this.viewer._update_notification({ id: notificationId, autoclose: 0 });
    };
    try {
      // Update progress callback
      const updateProgress = (percentage) => {
        if (cancelled) return;
        this.viewer._update_notification({
          id: notificationId,
          header: "Loading dataset point cloud",
          progress: percentage,
          closeable: true,
          onclose: () => {
            if (this.viewer.state.dataset_show_pointcloud) {
              this.viewer.state.dataset_show_pointcloud = false;
              this.viewer.notifyChange({ property: 'dataset_show_pointcloud' });
            }
          },
        });
      };
      updateProgress(0);

      // Fetch
      let response = await fetch(`${this.url}/pointcloud.ply`, {
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(`Failed to load point cloud: ${response.statusText}`);
      }

      // Track progress
      const reader = response.body.getReader();
      const contentLength = response.headers.get( 'X-File-Size' ) || response.headers.get( 'Content-Length' );
      const total = contentLength ? parseInt( contentLength ) : 0;
      if (total !== 0) {
        let loaded = 0;
        const stream = new ReadableStream({
          start(controller) {
            function readData() {
              reader.read().then(({ done, value }) => {
                if (done) {
                  controller.close();
                } else {
                  loaded += value.byteLength;
                  updateProgress(loaded / total);
                  controller.enqueue(value);
                  readData();
                }
              }, (e) => {
                controller.error(e);
              });
            }
            readData();
          }
        });
        response = new Response(stream);
      }
      const arrayBuffer = await response.arrayBuffer();
      const geometry = new PLYLoader().parse(arrayBuffer);

      if (cancelled) return;
      geometry.setAttribute('color', geometry.getAttribute('color'));
      geometry.computeBoundingSphere();

      const material = new THREE.ShaderMaterial({ 
        uniforms: {
          scale: { value: this.viewer.state.pointcloud_scale },
          point_ball_norm: { value: 2.0 },
        },
        vertexShader: _point_cloud_vertex_shader,
        fragmentShader: _point_cloud_fragment_shader,
      });
      material.vertexColors = geometry.hasAttribute('color');

      const points = new THREE.Points(geometry, material);
      this._pointcloud.add(points);
      this.viewer._update_notification({
        id: notificationId,
        header: "Loaded dataset point cloud",
        closeable: true,
        autoclose: notification_autoclose,
      });
    } catch (error) {
      if (cancelled) return;
      console.error('An error occurred while loading the PLY file:', error);
      this.viewer._update_notification({
        id: notificationId,
        header: "Error loading dataset point cloud",
        detail: error.message,
        type: "error",
        closeable: true,
      });
    } finally {
      this._cancel_load_pointcloud = undefined;
    }
  }
}

function _attach_selected_keyframe_details(viewer) {
  const getKeyframes = ({ camera_path_keyframes, camera_path_selected_keyframe, camera_path_loop }) => {
    const keyframeIndex = camera_path_keyframes?.findIndex((keyframe) => keyframe.id === camera_path_selected_keyframe);
    const keyframe = keyframeIndex >= 0 ? camera_path_keyframes[keyframeIndex] : undefined;
    const nextKeyframe = (camera_path_keyframes?.length || 0) > 1 ?
      ((camera_path_loop || keyframeIndex < camera_path_keyframes.length - 1) ?
        camera_path_keyframes[(keyframeIndex + 1) % camera_path_keyframes.length] : undefined) : undefined;
    return { keyframe, nextKeyframe, keyframeIndex };
  };

  const updateSelectedKeyframe = (state) => {
    const { dataset_images, camera_path_keyframes, 
            camera_path_selected_keyframe, camera_path_loop } = state;
    const { keyframe, nextKeyframe, keyframeIndex } = getKeyframes(state);
    const def = x => x !== undefined && x !== null;
    const change = {
      camera_path_selected_keyframe_natural_index: keyframeIndex + 1,
      camera_path_has_selected_keyframe: def(camera_path_selected_keyframe),
      camera_path_selected_keyframe_appearance_train_index: def(keyframe?.appearance_train_index) ? 
        keyframe.appearance_train_index : "",
      camera_path_selected_keyframe_appearance_url: def(keyframe?.appearance_train_index) ?
        dataset_images?.[`train/${keyframe.appearance_train_index}`]?.image_url : "",
      camera_path_selected_keyframe_fov: def(keyframe?.fov) ? keyframe.fov : state.camera_path_default_fov,
      camera_path_selected_keyframe_override_fov: def(keyframe?.fov),
      camera_path_selected_keyframe_velocity_multiplier: def(keyframe?.velocity_multiplier) ? keyframe.velocity_multiplier : 1,
      camera_path_selected_keyframe_duration: def(keyframe?.duration) ? keyframe.duration : 2,
    };
    for (const property in change) {
      if (state[property] !== change[property]) {
        state[property] = change[property];
        viewer.notifyChange({ property });
      }
    }
  }
  viewer.addEventListener("change", ({property, state, trigger}) => {
    // Add setters
    const { keyframe, nextKeyframe } = getKeyframes(state);
    if (property === "camera_path_selected_keyframe_override_fov" && keyframe) {
      if (state.camera_path_selected_keyframe_override_fov) {
        if (keyframe?.fov === undefined && state.camera_path_default_fov !== undefined) {
          keyframe.fov = state.camera_path_default_fov;
          viewer.notifyChange({ property: "camera_path_keyframes" });
        }
      } else {
        keyframe.fov = undefined;
        viewer.notifyChange({ property: "camera_path_keyframes" });
      }
      return
    }

    if (property === "camera_path_selected_keyframe_velocity_multiplier" && keyframe) {
      if (state.camera_path_selected_keyframe_velocity_multiplier && keyframe.velocity_multiplier !== state.camera_path_selected_keyframe_velocity_multiplier) {
        keyframe.velocity_multiplier = state.camera_path_selected_keyframe_velocity_multiplier;
        viewer.notifyChange({ property: "camera_path_keyframes" });
      }
      return;
    }

    if (property === "camera_path_selected_keyframe_duration" && keyframe) {
      if (state.camera_path_selected_keyframe_duration && keyframe.duration !== state.camera_path_selected_keyframe_duration) {
        keyframe.duration = state.camera_path_selected_keyframe_duration;
        viewer.notifyChange({ property: "camera_path_keyframes" });
      }
      return;
    }

    if (property === "camera_path_selected_keyframe_fov" && keyframe) {
      if (state.camera_path_selected_keyframe_override_fov && keyframe.fov !== state.camera_path_selected_keyframe_fov) {
        keyframe.fov = state.camera_path_selected_keyframe_fov;
        viewer.notifyChange({ property: "camera_path_keyframes" });
      }
      return;
    }

    if (property === "camera_path_selected_keyframe_appearance_train_index" && keyframe) {
      const index = (
        state.camera_path_selected_keyframe_appearance_train_index === ""
      ) ?  undefined : parseInt(state.camera_path_selected_keyframe_appearance_train_index);
      keyframe.appearance_train_index = index;
      viewer.notifyChange({ property: "camera_path_keyframes" });
      return;
    }
  });

  viewer.addEventListener("change", ({property, state}) => {
    if (property === undefined ||
        property === "camera_path_keyframes" ||
        property === "camera_path_selected_keyframe" ||
        property === "camera_path_loop" ||
        property === "camera_path_duration" ||
        property === "dataset_images")
      updateSelectedKeyframe(state);
  });
  updateSelectedKeyframe(viewer.state);
}


function _attach_camera_control(viewer) {
  viewer.addEventListener("change", ({ property, state }) => {
    if (property === undefined || property === 'camera_control_mode') {
      viewer.controls.mode = state.camera_control_mode;
      viewer.controls.resetDamping();
    }

    if (property === undefined || property === 'camera_control_inertia' || property === 'prerender_enabled') {
      // If pre-render is enabled, disable damping
      viewer.controls.enableDamping = !state.prerender_enabled && state.camera_control_inertia > 0;
      viewer.controls.dampingFactor = 1 - state.camera_control_inertia ** (1/2);
      viewer.controls.resetDamping();
    }
    const trans = x => Math.exp((x-1)*2.5);
    if (property === undefined || property === 'camera_control_rotation_sensitivity' || property === 'camera_control_rotation_inverted')
      viewer.controls.rotateSpeed = trans(state.camera_control_rotation_sensitivity) * (state.camera_control_rotation_inverted ? -1 : 1);
    if (property === undefined || property === 'camera_control_pan_sensitivity' || property === 'camera_control_pan_inverted')
      viewer.controls.panSpeed = trans(state.camera_control_pan_sensitivity) * (state.camera_control_pan_inverted ? -1 : 1);
    if (property === undefined || property === 'camera_control_zoom_sensitivity' || property === 'camera_control_zoom_inverted')
      viewer.controls.zoomSpeed = trans((state.camera_control_zoom_sensitivity-1)*2+0.5) * (state.camera_control_zoom_inverted ? -1 : 1);
    if (property === undefined || property === 'camera_control_key_speed')
      viewer.controls.keyPanSpeed = viewer.controls.keyRotateSpeed = 7 * trans(state.camera_control_key_speed);
  });
}


function parseBinding(expr) {
  if (expr.indexOf("&&") > 0) {
    const names = [];
    const callbacks = [];
    const parts = expr.split("&&").map(x => x.trim()).filter(x => x.length > 0).map(parseBinding);
    parts.forEach(([callback, partNames]) => {
      names.push(...partNames);
      callbacks.push(callback);
    });
    return [(state) => callbacks.every(callback => callback(state)), names];
  }
  if (expr.indexOf("==") > 0) {
    const [name, value] = expr.split("==");
    return [(state) => state[name] == value, [name]];
  }
  if (expr.indexOf("!=") > 0) {
    const [name, value] = expr.split("!=");
    return [(state) => state[name] != value, [name]];
  }
  if (expr.startsWith("!")) {
    const name = expr.slice(1);
    return [(state) => !state[name], [name]];
  }
  return [(state) => state[expr], [expr]];
}


export class Viewer extends THREE.EventDispatcher {
  constructor({ 
    viewport, 
    viewer_transform,
    viewer_initial_pose,
    url,
    state,
  }) {
    super();
    this._backgroundTexture = undefined;
    this.viewport = viewport || document.querySelector(".viewport");
    const width = this.viewport.clientWidth;
    const height = this.viewport.clientHeight;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.viewport.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera( 70, width / height, 0.01, 10 );
    this.camera.up = new THREE.Vector3(0, 0, 1);
    this.renderer_scene = new THREE.Scene();
    this.mouse_interactions = new MouseInteractions(this.renderer, this.camera, this.renderer_scene, this.viewport);

    this.controls = new ViewerControls(this.camera, this.renderer.domElement);

    this._enabled = true;
    this.renderer.setAnimationLoop((time) => this._animate(time));
    window.addEventListener("resize", () => this._resize());

    this.scene = new THREE.Group();
    this.renderer_scene.add(this.scene);
    if (viewer_transform) {
      if (Array.isArray(viewer_transform))
        viewer_transform = makeMatrix4(viewer_transform);
      this.scene.applyMatrix4(viewer_transform);
    }
    if (viewer_initial_pose) {
      if (Array.isArray(viewer_initial_pose))
        viewer_initial_pose = makeMatrix4(viewer_initial_pose);
      const matrix = viewer_initial_pose.clone();
      this.set_camera({ matrix });
      const target = new THREE.Vector3(0, 0, 0);
      
      this.controls.target?.copy(target);
      this.controls.update();
    }

    this._preview_canvas = document.createElement("canvas");
    this._preview_canvas.style.width = "100%";
    this._preview_canvas.style.height = "100%";
    this._preview_canvas.style.display = "none";
    this._preview_canvas.width = this.viewport.clientWidth;
    this._preview_canvas.height = this.viewport.clientHeight;
    this.viewport.appendChild(this._preview_canvas);
    this._preview_context = this._preview_canvas.getContext("2d");
    this._preview_context.fillStyle = "black";
    this._preview_context.fillRect(0, 0, this._preview_canvas.width, this._preview_canvas.height);

    this._state = state || {};
    Object.defineProperty(this, "state", { get: () => this._state });

    this.settings_manager = new SettingsManager({ viewer: this });
    this.state.camera_path_keyframes = [];
    this._is_preview_mode = false;

    this._camera_path = undefined;

    this._trajectory_curve = undefined;
    this._keyframe_frustums = {};
    this._player_frustum = undefined;
    this._last_frames = {};

    this._attach_computed_properties();
    this._attach_actions();
    _attach_camera_control(this);
    _attach_camera_path(this);
    _attach_fullscreen_mode(this);
    this._attach_preview_is_playing();
    this._attach_update_preview_mode();
    _attach_camera_path_curve(this);
    _attach_camera_path_keyframes(this);
    _attach_camera_path_selected_keyframe_pivot_controls(this);
    _attach_player_frustum(this);
    _attach_viewport_split_slider(this, this.viewport);
    _attach_selected_keyframe_details(this);

    this.addEventListener("change", ({ property, state }) => {
      if (property === undefined || property === 'render_fov') {
        this.camera.fov = state.render_fov;
        this.camera.updateProjectionMatrix();
        this._draw_background();
      }
    });
  }

  reset_settings() {
    this.settings_manager.reset();
  }

  clear_selected_dataset_image() {
    this.state.dataset_selected_image_id = undefined;
    this.notifyChange({ property: 'dataset_selected_image_id' });
  }

  set_camera({ matrix, fov }) {
    if (matrix !== undefined) {
      if (typeof matrix === "string") {
        try {
          // Parse camera
          const camNumbers = matrix.matchAll(/([-0-9]+(?:\.[0-9]+|)(?:e[-0-9]+|))/g).map(x => parseFloat(x)).toArray();
          matrix = makeMatrix4(camNumbers);
        } catch (e) {
          // TODO:
          this._update_notification({
            header: "Error setting camera matrix",
            detail: "Failed to parse camera format. Ensure it is 12 numbers separated by comma or newline.",
            type: "error",
          });
        }
      }

      matrix = matrix.clone();
      matrix.multiply(_R_threecam_cam.clone().invert());
      matrix.premultiply(this.scene.matrixWorld);
      const position = new THREE.Vector3();
      const quaternion = new THREE.Quaternion();
      const scale = new THREE.Vector3();
      matrix.decompose(position, quaternion, scale);
      this.camera.position.set(...position);
      this.camera.quaternion.set(...quaternion);

      // Reset damping
      this.controls?._sphericalDelta?.set( 0, 0, 0 );
			this.controls?._panOffset?.set( 0, 0, 0 );
    }
    if (fov !== undefined) {
      this.camera.fov = fov;
    }
  }

  set_camera_to_selected_dataset_image() {
    const { dataset_images, dataset_selected_image_id } = this.state;
    if (dataset_selected_image_id === undefined) return;
    const dataset_image = dataset_images[dataset_selected_image_id];
    if (!dataset_image) return;
    this.set_camera({ matrix: dataset_image.matrix });
  }

  set_camera_to_selected_keyframe() {
    const { camera_path_keyframes, camera_path_selected_keyframe } = this.state;
    if (camera_path_selected_keyframe === undefined) return;
    const keyframe = camera_path_keyframes.find((keyframe) => keyframe.id === camera_path_selected_keyframe);
    if (!keyframe) return;
    const matrix = new THREE.Matrix4();
    matrix.compose(keyframe.position, keyframe.quaternion, new THREE.Vector3(1, 1, 1));
    this.set_camera({ matrix });
  }

  _get_render_params() {
    const state = this.state;

    const _get_params = ({ resolution, width, height, fov, matrix, ...rest }) => {
      [width, height] = computeResolution([width, height], resolution);
      const round = (x) => Math.round(x * 100000) / 100000;
      const focal = height / (2 * Math.tan(THREE.MathUtils.degToRad(fov) / 2));
      const request = {
        pose: matrix4ToArray(matrix).map(round).join(","),
        intrinsics: [focal, focal, width/2, height/2].map(round).join(","),
        image_size: `${width},${height}`,
        ...rest
      };
      if (this.state.preview_camera !== undefined) {
        request.output_type = state.camera_path_render_output_type;
      } else {
        request.output_type = state.output_type;
        if (state.split_enabled && state.split_output_type) {
          request.split_output_type = state.split_output_type;
          request.split_percentage = "" + round(state.split_percentage === undefined ? 0.5 : state.split_percentage);
          request.split_tilt = "" + round(state.split_tilt || 0.0);
        }
      }
      return request;
    }

    let cameraParams = this._get_camera_params();
    let height = this.viewport.clientHeight;
    let width = Math.round(height * cameraParams.aspect);
    let params, paramsJSON;
    if (state.prerender_enabled) {
      if (this._was_full_render) {
        params = _get_params({ 
          resolution: state.render_resolution, 
          width, height, ...cameraParams });
        paramsJSON = JSON.stringify(params);
        if (paramsJSON !== this._last_render_params) {
          params = _get_params({ 
            resolution: state.prerender_resolution, 
            width, height, ...cameraParams });
          paramsJSON = JSON.stringify(params);
          this._was_full_render = false;
        }
      } else {
        params = _get_params({ 
          resolution: state.prerender_resolution, 
          width, height, ...cameraParams });
        paramsJSON = JSON.stringify(params);
        if (paramsJSON === this._last_render_params) {
          params = _get_params({ 
            resolution: state.render_resolution, 
            width, height, ...cameraParams });
          paramsJSON = JSON.stringify(params);
          this._was_full_render = true;
        }
      }
    } else {
      params = _get_params({ 
        resolution: state.render_resolution, 
        width, height, ...cameraParams });
      paramsJSON = JSON.stringify(params);
    }
    if (paramsJSON === this._last_render_params) return;
    this._last_render_params = paramsJSON;
    return params;
  }

  set_frame_renderer(frame_renderer) {
    // Connect HTTP renderer
    if (this.frame_renderer)
      throw new Error("Frame renderer is already set.")
    this.frame_renderer = frame_renderer;
    let lastUpdate = Date.now() - 1;
    const run = async () => {
      while (true) {
        try {
          const params = this._get_render_params();
          if (params !== undefined) {
            const image = await this.frame_renderer.render(params, { flipY: true });
            this.dispatchEvent({ type: "frame", image });
            this._last_frames[0] = image;
            this._draw_background();
          }

          // Wait
          const wait = Math.max(0, 1000 / 30 - (Date.now() - lastUpdate));
          lastUpdate = Date.now();
          if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
        } catch (error) {
          console.error("Error updating single frame:", error.message);
          this._update_notification({
            header: "Error rendering frame",
            detail: error.message,
            type: "error",
            id: "renderer",
            closeable: true,
          });
        }
      }
    };

    run();

    // Update state
    this.state.has_method = true;
    this.notifyChange({ property: 'has_method' });
  }

  set_websocket_renderer({ url }) {
    this.set_frame_renderer(new WebSocketFrameRenderer({ 
      url,
      update_notification: (notification) => {
        viewer._update_notification(notification);
      }
    }));
    let absoluteUrl = new URL(url, window.location.href).href;
    if (absoluteUrl.startsWith("http://")) absoluteUrl = "ws://" + absoluteUrl.slice(7);
    if (absoluteUrl.startsWith("https://")) absoluteUrl = "wss://" + absoluteUrl.slice(8);
    this.state.frame_renderer_url = absoluteUrl;
    this.notifyChange({ property: 'frame_renderer_url' });
  }

  set_dataset({ url, parts }) {
    if (this.dataset_manager) {
      this.dataset_manager.dispose();
      this.dataset_manager = undefined;
    }
    // Add DatasetManager
    this.dataset_manager = new DatasetManager({ 
      viewer: this, 
      url,
      parts,
    });
  }

  _resize() {
    const width = this.viewport.clientWidth;
    const height = this.viewport.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this._preview_canvas.width = width;
    this._preview_canvas.height = height;
    this._draw_background();
    this.dispatchEvent({ 
      type: "resize", 
      width,
      height,
    });
  }

  _animate(time) {
    if (this._lastTime === undefined) this._lastTime = time;
    const delta = time - this._lastTime;
    this._lastTime = time;
    if (!this._is_preview_mode) {
      this.mouse_interactions.update();
      if (!this.mouse_interactions.isCaptured()) {
        this.controls?.update(delta);
      }
      this.renderer.render(this.renderer_scene, this.camera);
    }
  }

  _get_camera_params() {
    if (this.state.preview_camera !== undefined) {
      return this.state.preview_camera;
    }
    const pose = this.camera.matrixWorld.clone();
    pose.multiply(_R_threecam_cam);
    pose.premultiply(this.scene.matrixWorld.clone().invert());
    // const fovRadians = THREE.MathUtils.degToRad(fov);
    const fov = this.camera.fov;
    let appearance_train_indices = undefined;
    let appearance_weights = undefined;
    if (this.state.camera_path_selected_keyframe_appearance_train_index !== undefined &&
        this.state.camera_path_selected_keyframe_appearance_train_index !== ""
    ) {
      appearance_train_indices = [parseInt(this.state.camera_path_selected_keyframe_appearance_train_index)];
      appearance_weights = [1]
    } else if (this.state.render_appearance_train_index !== undefined &&  
               this.state.render_appearance_train_index !== "") {
      appearance_train_indices = [parseInt(this.state.render_appearance_train_index)];
      appearance_weights = [1]
    }
    return {
      matrix: pose,
      fov,
      aspect: this.camera.aspect,
      appearance_train_indices,
      appearance_weights,
    };
  }

  notifyChange(props) {
    this.dispatchEvent({ 
      ...props,
      type: "change", 
      state: this.state, 
    });
  }

  dispatchAction(action, payload) {
    this.dispatchEvent({
      ...(payload || {}),
      type: "action",
      action,
      state: this.state,
    });
  }

  addComputedProperty({ name, getter, dependencies }) {
    this.state[name] = getter(this.state);
    this.addEventListener("change", ({ property, state }) => {
      if (property !== undefined && !dependencies.includes(property)) return;
      state[name] = getter(state);
      this.notifyChange({ property: name });
    });
  }

  copy_public_url() {
    const el = document.createElement('textarea');
    el.value = this.state.viewer_public_url;
    el.setAttribute('readonly', '');
    el.style.position = 'absolute';
    el.style.left = '-9999px';
    el.style.opacity = 0;
    document.body.appendChild(el);
    el.select();
    document.execCommand('copy');
    document.body.removeChild(el);
  }

  async create_public_url() {
    if (this.state.viewer_public_url) return;
    this._update_notification({
      id: "public-url",
      header: "Creating public URL",
      closeable: false,
    });
    let publicUrl = undefined;
    this.state.viewer_requesting_public_url = true;
    this.notifyChange({ property: "viewer_requesting_public_url" });
    try {
      const response = await fetch("./create-public-url?accept_license_terms=yes", { method: "POST" });
      let result;
      try {
        result = await response.json();
      } catch (error) {
        throw new Error(`Failed to create public URL: ${response.statusText}`);
      }
      if (result.status === "error") {
        throw new Error(result.message);
      }
      publicUrl = result.public_url;

      console.log("Public URL:", publicUrl);
      this.state.viewer_public_url = publicUrl;
      this.notifyChange({ property: "viewer_public_url" });
      this._update_notification({
        id: "public-url",
        header: "Public URL created",
        detailHTML: `<a href="${publicUrl}" target="_blank" style="word-wrap:break-word;word-break:break-all">${publicUrl}</a>`,
        closeable: true,
      });
    } catch (error) {
      console.error("Failed to create public URL:", error.message);
      this._update_notification({
        id: "public-url",
        header: "Failed to create public URL",
        detail: error.message,
        type: "error",
      });
    } finally {
      this.state.viewer_requesting_public_url = false;
      this.notifyChange({ property: "viewer_requesting_public_url" });
    }
  }

  delete_all_keyframes() {
    this.state.camera_path_selected_keyframe = undefined;
    this.notifyChange({ property: "camera_path_selected_keyframe" });
    this.state.camera_path_keyframes = [];
    this.notifyChange({ property: "camera_path_keyframes" });
    this.state.camera_path_duration = 0;
    this.notifyChange({ property: "camera_path_duration" });
  }

  delete_keyframe(keyframe_id) {
    this.state.camera_path_keyframes = this.state.camera_path_keyframes.filter((keyframe) => keyframe.id !== keyframe_id);
    if (this.state.camera_path_selected_keyframe === keyframe_id) {
      this.state.camera_path_selected_keyframe = undefined;
      this.notifyChange({ property: "camera_path_selected_keyframe" });
    }
    const n = this.state.camera_path_keyframes.length;
    this.state.camera_path_duration = n < 1 ? 0 : this.state.camera_path_duration * n / (n+1);
    this.notifyChange({ property: "camera_path_duration" });
    this.notifyChange({ property: "camera_path_keyframes" });
  }

  delete_selected_keyframe() {
    if (this.state.camera_path_selected_keyframe === undefined) return;
    this.delete_keyframe(this.state.camera_path_selected_keyframe);
  }

  add_keyframe() {
    const { matrix } = this._get_camera_params();
    const quaternion = new THREE.Quaternion();
    const position = new THREE.Vector3();
    const scale = new THREE.Vector3();
    matrix.decompose(position, quaternion, scale);
    const id = this._keyframeCounter = (this._keyframeCounter || 0) + 1;
    this.state.camera_path_keyframes.push({
      id,
      quaternion,
      position,
      fov: undefined,
    });
    const duration = this.state.camera_path_duration || 0;
    const n = this.state.camera_path_keyframes.length;
    this.state.camera_path_duration = (n < 2 && duration == 0) ? 2 : duration * n / (n - 1);
    this.notifyChange({ property: "camera_path_duration" });
    this.notifyChange({ property: "camera_path_keyframes" });
  }

  clear_selected_keyframe() {
    this.state.camera_path_selected_keyframe = undefined;
    this.notifyChange({ property: "camera_path_selected_keyframe" });
  }

  _attach_computed_properties() {
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
      name: "dataset_selected_image_pose",
      dependencies: ["dataset_images", "dataset_selected_image_id"],
      getter: ({ dataset_images, dataset_selected_image_id }) => {
        const matrix = dataset_images?.[dataset_selected_image_id]?.matrix;
        if (!matrix) return "";
        return formatMatrix4(matrix, 5);
      },
    });

    for (const prop of ["image_name", "image_url", "width", "height", "fx", "fy", "cx", "cy"]) {
      this.addComputedProperty({
        name: `dataset_selected_${prop}`,
        dependencies: ["dataset_images", "dataset_selected_image_id"],
        getter: ({ dataset_images, dataset_selected_image_id }) => dataset_images?.[dataset_selected_image_id]?.[prop] || "",
      });
    }

    this.addComputedProperty({
      name: "render_appearance_image_url",
      dependencies: ["dataset_images", "render_appearance_train_index"],
      getter: ({ dataset_images, render_appearance_train_index }) => {
        if (render_appearance_train_index === undefined || 
            render_appearance_train_index === "") 
          return "";
        const image = dataset_images?.[`train/${render_appearance_train_index}`];
        return image?.image_url || "";
      }
    });

    this.addComputedProperty({
      name: "camera_path_computed_duration",
      dependencies: ["camera_path_keyframes", "camera_path_duration", "camera_path_interpolation"],
      getter: ({ camera_path_keyframes, camera_path_duration, camera_path_interpolation }) => {
        if (camera_path_interpolation === "none") {
          return camera_path_keyframes.map(x => x.duration === undefined ? 2 : x.duration).reduce((a, b) => a + b, 0);
        }
        return camera_path_duration;
      }
    });
  }

  _attach_actions() {
    this.addEventListener("action", ({ action, state }) => {
      if (action === "set_camera_to_selected_dataset_image")
        this.set_camera_to_selected_dataset_image();
      if (action === "set_camera_to_selected_keyframe")
        this.set_camera_to_selected_keyframe();
      if (action === "delete_all_keyframes")
        this.delete_all_keyframes();
      if (action === "delete_selected_keyframe")
        this.delete_selected_keyframe();
      if (action === "add_keyframe")
        this.add_keyframe();
      if (action === "clear_selected_keyframe")
        this.clear_selected_keyframe();
      if (action === "reset_settings")
        this.reset_settings();
      if (action === "clear_selected_dataset_image")
        this.clear_selected_dataset_image();
      if (action === "render_video")
        this.render_video();
      if (action === "save_camera_path")
        this.save_camera_path();
      if (action === "viewer_create_public_url")
        this.create_public_url();
      if (action === "viewer_copy_public_url")
        this.copy_public_url();
    });
  }

  _attach_update_preview_mode() {
    this.addEventListener('change', ({ property, state }) => {
      // Update preview mode
      if (property === undefined || 
          property !== 'preview_is_preview_mode') {
        const { 
          preview_is_preview_mode,
        } = state;
        this._is_preview_mode = preview_is_preview_mode;
        if (this.controls)
          this.controls.enabled = !this._is_preview_mode;
        this._draw_background();
        this.renderer.domElement.style.display = this._is_preview_mode ? "none" : "block";
        this._preview_canvas.style.display = this._is_preview_mode ? "block" : "none";
      }

      // Update preview camera
      if (property === undefined ||
          property === 'camera_path_trajectory' ||
          property === 'preview_frame' ||
          property === 'preview_is_preview_mode' ||
          property === 'render_appearance_train_index' ||
          property === 'camera_path_resolution_1' ||
          property === 'camera_path_resolution_2') {
        const { 
          camera_path_keyframes,
          camera_path_trajectory,
          preview_frame,
          preview_is_preview_mode,
          render_appearance_train_index,
          camera_path_resolution_1,
          camera_path_resolution_2,
        } = state;

        let preview_camera = undefined;
        if (preview_is_preview_mode && camera_path_trajectory && camera_path_trajectory.positions.length > 0) {
          const { positions, quaternions, fovs, weights } = camera_path_trajectory;
          const num_frames = positions.length;
          const frame = Math.min(Math.max(0, Math.floor(preview_frame)), num_frames - 1);
          const position = new THREE.Vector3().copy(positions[frame]);
          const quaternion = new THREE.Quaternion().copy(quaternions[frame]);
          const fov = fovs[frame];
          const pose = new THREE.Matrix4();
          let appearance_weights = weights[frame];
          let appearance_train_indices = camera_path_trajectory.appearanceTrainIndices;
          if (appearance_weights?.length === 0) {
            appearance_weights = undefined;
            appearance_train_indices = undefined;
          }
          pose.compose(position, quaternion, new THREE.Vector3(1, 1, 1));
          preview_camera = {
            matrix: pose,
            fov,
            aspect: camera_path_resolution_1 / camera_path_resolution_2,
            appearance_weights,
            appearance_train_indices,
          };
        }
        if (preview_camera !== state.preview_camera) {
          state.preview_camera = preview_camera;
          this.notifyChange({ property: 'preview_camera' });
        }
      }
    });
  }

  _attach_preview_is_playing() {
    let preview_interval;
    let isCancelled = [false];
    this.addEventListener('change', ({ property, state }) => {
      if (property !== undefined &&
          property !== 'camera_path_trajectory' &&
          property !== 'camera_path_framerate' &&
          property !== 'camera_path_keyframes' &&
          property !== 'camera_path_interpolation' &&
          property !== 'camera_path_duration' &&
          property !== 'preview_is_playing') return;
      const {
        camera_path_trajectory,
        camera_path_framerate,
        camera_path_interpolation,
        camera_path_duration,
        camera_path_keyframes,
        preview_is_playing,
      } = state;

      // Add preview timer
      if (preview_interval) {
        isCancelled[0] = true;
        isCancelled = [false];
        clearInterval(preview_interval);
        preview_interval = undefined;
      }

      if (preview_is_playing) {
        let delays;
        const isCancelledLocal = isCancelled;
        if (camera_path_interpolation === 'none') {
          delays = camera_path_keyframes.map(x => 1000 * (x.duration === undefined ? 2 : x.duration));
        } else {
          delays = Array.from({ length: camera_path_trajectory?.positions?.length || 0 }, (_, i) => 1000 / camera_path_framerate);
        }
        const stepCallback = () => {
          if (isCancelledLocal[0]) return;
          state.preview_frame = delays.length > 0 ? (state.preview_frame + 1) % delays.length : 0;
          preview_interval = setTimeout(stepCallback, delays[state.preview_frame]);
          if (isCancelledLocal[0]) return;
          this.notifyChange({ property: 'preview_frame' });
        };
        preview_interval = setTimeout(stepCallback, delays[state.preview_frame] || 0);
      }
    });
  }

  export_camera_path() {
    const state = this.state;
    const w = state.camera_path_resolution_1;
    const h = state.camera_path_resolution_2;
    const appearances = [];
    const keyframes = [];
    for (const keyframe of state.camera_path_keyframes) {
      const pose = new THREE.Matrix4();
      pose.compose(keyframe.position, keyframe.quaternion, new THREE.Vector3(1, 1, 1));
      let appearance = undefined;
      const keyframe_dict = {
        pose: matrix4ToArray(pose),
        fov: keyframe.fov,
        velocity_multiplier: keyframe.velocity_multiplier,
        duration: keyframe.duration,
      };
      if (keyframe.appearance_train_index !== undefined) {
        keyframe_dict.appearance = appearance = {
          embedding_train_index: keyframe.appearance_train_index,
        };
      }
      keyframes.push(keyframe_dict);
      if (appearance) {
        appearances.push(appearance);
      }
    }

    // now populate the camera path:
    const trajectory_frames = this.state.camera_path_trajectory;
    if (!trajectory_frames) {
      throw new Error("No trajectory frames found");
    }

    const frames = [];
    if (trajectory_frames) {
      for (let i = 0; i < trajectory_frames.positions.length; i++) {
        const pos = trajectory_frames.positions[i];
        const wxyz = trajectory_frames.quaternions[i];
        const fov = trajectory_frames.fovs[i];
        const weights = trajectory_frames.weights[i];
        const pose = new THREE.Matrix4();
        pose.compose(pos, wxyz, new THREE.Vector3(1, 1, 1));
        const focal = h / (2 * Math.tan(THREE.MathUtils.degToRad(fov) / 2));
        frames.push({
          pose: matrix4ToArray(pose),
          intrinsics: [focal, focal, w / 2, h / 2],
          appearance_weights: weights,
        });
      }
    }

    const source = {
      type: "interpolation",
      interpolation: state.camera_path_interpolation,
      keyframes,
      default_fov: state.camera_path_default_fov,
      duration: state.camera_path_duration,
    };
    let fps = state.camera_path_framerate;
    if (source.interpolation === "kochanek-bartels") {
      source.is_cycle = state.camera_path_loop;
      source.tension = state.camera_path_tension;
      source.continuity = state.camera_path_continuity;
      source.bias = state.camera_path_bias;
    } else if (source.interpolation === "linear" || source.interpolation === "circle") {
      source.is_cycle = state.camera_path_loop;
    }
    const data = {
      version: 'nerfbaselines-v2',
      camera_model: "pinhole",
      image_size: [w, h],
      fps,
      source,
      frames,
    };
    if (appearances.length != 0) {
      data.appearances = appearances;
    }
    if (source.interpolation === "none") {
      // Add frame repeats to match target FPS
      const frame_repeats = [];
      let time = 0;
      let nFrames = 0;
      keyframes.forEach((keyframe, i) => {
        const duration = keyframe.duration || 2;
        const numFrames = Math.max(0, Math.round((time + duration) * fps) - nFrames);
        frame_repeats.push(numFrames);
        nFrames += numFrames;
        time = nFrames / fps;
      });
      data.frame_repeats = frame_repeats;
    }
    return data
  }

  async save_camera_path() {
    try {
      const data = this.export_camera_path();
      await saveAs(new Blob([JSON.stringify(data, null, '  ')]), { 
        type: "application/json",
        filename: "camera_path.json",
        extension: "json",
        description: "Camera trajectory JSON",
      });
    } catch (error) {
      console.error("Error saving camera path:", error);
      this._update_notification({
        header: "Error saving camera path",
        detail: error.message,
        type: "error",
      });
    }
  }

  async render_video() {
    const width = this.state.camera_path_resolution_1;
    const height = this.state.camera_path_resolution_2;
    const fps = this.state.camera_path_framerate;
    const { positions, quaternions, fovs, weights, appearanceTrainIndices } = this.state.camera_path_trajectory;
    const writer = new VideoWriter({ 
      width, height, fps, 
      type: this.state.camera_path_render_format,
      mp4Codec: this.state.camera_path_render_mp4_codec,
      webmCodec: this.state.camera_path_render_webm_codec,
      keyframeInterval: this.state.camera_path_render_keyframe_interval,
      numFrames: positions.length });
    try {
      await writer.saveAs();
    } catch (error) {
      if (error.name === "AbortError") {
        return;
      }
      console.error("Error saving video:", error);
      this._update_notification({
        header: "Error saving video",
        detail: error.message,
        type: "error",
      });
    }

    const renderId = this._renderVideoId = (this._renderVideoId || 0) + 1;
    let closed = false;
    this._update_notification({
      header: "Rendering video",
      id: renderId,
      progress: 0,
      onclose: () => { closed = true; },
    });

    try {
      for (let i=0; i < positions.length; i++) {
        if (closed) break;
        let appearance_weights = weights[i];
        let appearance_train_indices = appearanceTrainIndices;
        if (appearance_weights?.length === 0) {
          appearance_weights = undefined;
          appearance_train_indices = undefined;
        }
        const round = (x) => Math.round(x * 100000) / 100000;
        const focal = height / (2 * Math.tan(THREE.MathUtils.degToRad(fovs[i]) / 2));
        const matrix = new THREE.Matrix4().compose(positions[i], quaternions[i], new THREE.Vector3(1, 1, 1));
        const renderRequest = {
          pose: matrix4ToArray(matrix).map(round).join(","),
          intrinsics: [focal, focal, width/2, height/2].map(round).join(","),
          image_size: `${width},${height}`,
          output_type: this.state.camera_path_render_output_type,
          appearance_weights,
          appearance_train_indices,
          lossless: true,
        };
        const frame = await this.frame_renderer.render(renderRequest);
        if (closed) break;
        try {
          await writer.addFrame(frame);
        } finally {
          frame?.close?.();
        }
        this._update_notification({
          header: "Rendering video",
          id: renderId,
          progress: i / positions.length,
          onclose: () => { closed = true; },
        });
      }

      if (closed) {
        await writer.close?.();
      } else {
        await writer.finalize();
        this._update_notification({
          id: renderId,
          header: "Rendering finished",
          autoclose: notification_autoclose,
          closeable: true,
        });
      }
    } catch (error) {
      this._update_notification({
        id: renderId,
        header: "Rendering failed",
        detail: error.message,
        type: "error",
        closeable: true,
      });
    }
  }

  load_camera_path(data) {
    try {
      if (!data) {
        throw new Error("No data provided");
      }
      if (data.version !== "nerfbaselines-v2") {
        throw new Error(`Unsupported version ${data.version}. Only 'nerfbaselines-v2' is supported`);
      }
      const state = this.state;
      if (data.camera_model !== "pinhole") {
        throw new Error("Only pinhole camera model is supported");
      }
      const source = data.source;
      if (!source) {
        throw new Error("Trajectory does not contain 'source'. It is not editable.");
      }
      const {
        interpolation,
      } = source;
      if (source.type !== "interpolation" || ["none", "kochanek-bartels", "circle", "linear"].indexOf(interpolation) === -1) {
        throw new Error("Trajectory does not contain 'source' with 'type' set to 'interpolation' and 'interpolation' set to 'none', 'linear', 'kochanek-bartels', or 'circle'. It is not editable.");
      }
      function validate_appearance(appearance) {
        if (appearance && appearance.embedding_train_index === undefined) {
          throw new Error("Setting appearance is only supported through embedding_train_index");
        }
        return appearance;
      }
      state.camera_path_interpolation = interpolation;
      [state.camera_path_resolution_1, state.camera_path_resolution_2] = data.image_size;
      if (interpolation === "kochanek-bartels") {
        state.camera_path_tension = source.tension || 0;
        state.camera_path_continuity = source.continuity || 0;
        state.camera_path_bias = source.bias || 0;
        state.camera_path_loop = source.is_cycle;
      } else if (interpolation === "linear" || interpolation === "circle") {
        state.camera_path_loop = source.is_cycle;
      }
      state.camera_path_framerate = data.fps;
      const correctnull = (x) => (x === null) ? undefined : x;
      if (correctnull(data.fps) !== undefined)
        state.camera_path_framerate = data.fps;
      const {
        default_fov,
        duration,
      } = source;
      if (correctnull(default_fov) !== undefined)
        state.camera_path_default_fov = default_fov;
      state.camera_path_duration = correctnull(duration) === undefined ? (2*source["keyframes"].length) : duration;
      const keyframes = [];
      for (let k of source["keyframes"]) {
        const matrix = makeMatrix4(k.pose);
        const position = new THREE.Vector3();
        const quaternion = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        matrix.decompose(position, quaternion, scale);
        const appearance = validate_appearance(correctnull(k.appearance));
        const appearance_train_index = appearance ? appearance.embedding_train_index : undefined;
        keyframes.push({
          id: this._keyframeCounter = (this._keyframeCounter || 0) + 1,
          quaternion,
          position,
          fov: correctnull(k.fov),
          appearance_train_index,
          velocity_multiplier: correctnull(k.velocity_multiplier),
          duration: correctnull(k.duration),
        });
      }
      state.camera_path_keyframes = keyframes;
      state.camera_path_selected_keyframe = undefined;
      this.notifyChange({ property: undefined });
    } catch (error) {
      console.error("Error loading camera path:", error);
      this._update_notification({
        header: "Error loading camera path",
        detail: error.message,
        type: "error",
      });
    }
  }

  attach_gui(root) {
    root = root || document.body;
    const state = this.state;
    // Handle state change
    function getValue(element) {
      const { name, value, type, checked } = element;
      if (type === "checkbox") return checked;
      if (type === "radio") return value;
      if (type === "number" || type === "range") return parseFloat(value);
      return value;
    }
    function setValue(element, value) {
      const { type } = element;
      if (type === "checkbox") {
        element.checked = value;
      } else if (type === "radio") {
        element.checked = element.value === value;
      } else {
        element.value = value;
        const e = new Event("input");
        e.simulated = true;
        element.dispatchEvent(e);
      }
    }
    const querySelectorAll = (selector) => {
      let out = root.querySelectorAll(selector);
      if (root.matches(selector)) out = Array.from(out).concat(root);
      return out;
    }
    querySelectorAll("[data-set-viewer-ref]").forEach(element => { element.viewer = this });
    querySelectorAll("input[name],select[name]").forEach(element => {
      element.addEventListener("change", (event) => {
        if (event.simulated) return;
        state[name] = getValue(event.target);
        this.notifyChange({ 
          property: name, 
          origin: element,
        });
      });
      element.addEventListener("input", (event) => {
        if (event.simulated) return;
        state[name] = getValue(event.target);
        this.notifyChange({ 
          property: name, 
          origin: element,
        });
        if (name === "preview_frame" && type === "range") {
          // Changing preview_frame stops the preview
          state.preview_is_playing = false;
          this.notifyChange({ 
            property: "preview_is_playing", 
            origin: element,
          });
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
        this.addEventListener("change", ({ property, state }) => {
          if (property !== undefined && property !== 'camera_path_trajectory') return;
          if (origin === element) return;
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
      this.addEventListener("change", ({ property, origin }) => {
        if (property !== name && property !== undefined) return;
        if (origin === element) return;
        setValue(element, state[name]);
      });
    });
    root.querySelectorAll("[data-bind]").forEach(element => {
      const name = element.getAttribute("data-bind");
      this.addEventListener("change", ({ property }) => {
        if (property !== name && property !== undefined) return;
        element.innerText = state[name];
      });
    });

    querySelectorAll("[data-options]").forEach(element => {
      const name = element.getAttribute("data-options");
      function updateOptions() {
        const selectedValue = state[element.name];
        // Clear all options
        const oldOptions = {};
        Array.from(element.children).forEach((option) => {
          oldOptions[option.value] = option;
          element.removeChild(option);
        });

        // Add new options
        if (state[name]) {
          for (let value of state[name]) {
            let text;
            if (typeof value === "string")
              text = value;
            else {
              text = value.label;
              value = value.value;
            }

            const option = oldOptions[value] || document.createElement("option");
            option.value = value;
            option.textContent = text;
            if (value === selectedValue) option.setAttribute("selected", "");
            element.appendChild(option);
          }
          element.value = selectedValue;
        }
      }
      this.addEventListener("change", ({ property }) => {
        if (property !== name && property !== undefined) return;
        updateOptions();
      });
      updateOptions();
    });

    querySelectorAll("[data-enable-if]").forEach(element => {
      const [evalFn, dependencies] = parseBinding(element.getAttribute("data-enable-if"));
      this.addEventListener("change", ({ property, state }) => {
        if (property !== undefined && !dependencies.includes(property)) return;
        let value = evalFn(state);
        if (Array.isArray(value)) value = value.length > 0;
        if (typeof value === "object") value = Object.entries(value).length > 0;
        if (element.tagName.toLowerCase() === "a")
          element.toggleAttribute("data-disabled", !value);
        else
          element.disabled = !value;
      });
    });

    querySelectorAll("[data-visible-if]").forEach(element => {
      const [evalFn, dependencies] = parseBinding(element.getAttribute("data-visible-if"));
      let display = element.style.display;
      if (display === "none") display = null;
      this.addEventListener("change", ({ property, state }) => {
        if (property !== undefined && !dependencies.includes(property)) return;
        element.style.display = evalFn(state) ? display : "none";
      });
    });

    querySelectorAll("[data-action]").forEach(element => {
      const action = element.getAttribute("data-action");
      element.addEventListener("click", (e) => {
        if (e.simulated) return;
        if (!element.getAttribute("disabled"))
          this.dispatchAction(action);
      });
    });

    // data-bind-class has the form "class1:property1"
    querySelectorAll("[data-bind-class]").forEach(element => {
      const [class_name, expr] = element.getAttribute("data-bind-class").split(":");
      const [evalFn, dependencies] = parseBinding(expr);
      this.addEventListener("change", ({ property, state }) => {
        if (property !== undefined && !dependencies.includes(property)) return;
        if (evalFn(state)) {
          element.classList.add(class_name);
        } else {
          element.classList.remove(class_name);
        }
      });
    });

    // data-bind-attr has the form "attribute:property"
    querySelectorAll("[data-bind-attr]").forEach(element => {
      const [attr, name] = element.getAttribute("data-bind-attr").split(":");
      this.addEventListener("change", ({ property, state }) => {
        if (property !== name && property !== undefined) return;
        element.setAttribute(attr, state[name]);
      });
      if (state[name] !== undefined)
        element.setAttribute(attr, state[name]);
    });

    querySelectorAll('#input_camera_path').forEach((input) => {
      input.addEventListener('change', (event) => {
        if (event.simulated) return;
        const file = event.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
          const data = JSON.parse(event.target.result);
          // Empty the input value to allow loading the same file again
          input.value = '';
          // Load camera path
          this.load_camera_path(data);
        };
        reader.readAsText(file);
      });
    });

    // Camera pose textarea
    const camera_pose_elements = root.querySelectorAll("[name=camera_pose]");
    const updateCameraElements = () => {
      const { matrix } = this._get_camera_params();
      const value = formatMatrix4(matrix, 5);
      camera_pose_elements.forEach((element) => {
        if (!element.hasFocus) 
          element.value = value
      });
    }
    camera_pose_elements.forEach((element) => {
      element.addEventListener("focus", () => element.hasFocus = true);
      element.addEventListener("input", () => element.hasFocus = true);
      element.addEventListener("blur", () => {
        element.hasFocus = false
        this.set_camera(element.value);
      });
    });
    if (camera_pose_elements.length > 0) {
      // Set interval to update camera pose
      this._updateCameraPoseInterval = setInterval(updateCameraElements, 300);
      updateCameraElements();
    }

    // Method info
    const method_info_element = document.getElementById("method_info");
    if (method_info_element) {
      const update_method_info = ({ method_info }) => {
        const newHtml = method_info ? buildMethodInfo(method_info) : "";
        method_info_element.innerHTML = newHtml;
      }
      this.addEventListener("change", ({ property, state }) => {
        if (property === "method_info" || property === undefined)
          update_method_info(state);
      });
    }
    
    // Dataset info
    const dataset_info_element = document.getElementById("dataset_info");
    if (dataset_info_element) {
      const update_dataset_info = ({ dataset_info }) => {
        const newHtml = dataset_info ? buildDatasetInfo(dataset_info) : "";
        dataset_info_element.innerHTML = newHtml;
      }
      this.addEventListener("change", ({ property, state }) => {
        if (property === "dataset_info" || property === undefined)
          update_dataset_info(state);
      });
    }

    // Method hparams
    const method_hparams_element = document.getElementById("method_hparams");
    if (method_hparams_element) {
      const update_method_hparams = (method_hparams) => {
        // Remove display none
        let newHtml = ""
        if (method_hparams) {
          for (const k in method_hparams) {
            newHtml += `<strong>${k}:</strong><span>${method_hparams[k]}</span>`
          }
        }
        method_hparams_element.innerHTML = newHtml;
      }
      this.addEventListener("change", ({ property, state }) => {
        if (property === "method_info" || property === undefined)
          update_method_hparams(state.method_info?.hparams || {});
      });
    }

    this.addEventListener("change", ({ property, state }) => {
      if (property === "theme_color" || property === undefined) {
        document.documentElement.style.setProperty("--theme-color", state.theme_color);
      }
      if (property === "viewer_font_size" || property === undefined) {
        document.documentElement.style.fontSize = `${state.viewer_font_size}rem`;
      }
    });

    this.notifyChange({ property: undefined });
  }

  _update_notification({ id, header, progress, autoclose=undefined, detail="", detailHTML=undefined, type="info", onclose, closeable=true }) {
    this._notifications = this._notifications || {};
    this._notification_id_counter = this._notification_id_counter || 0;
    if (id === undefined) {
      id = this._notification_id_counter++;
    }
    let notification = this._notifications[id];
    const close = () => {
      notification.remove();
      delete this._notifications[id];
    }
    if (autoclose === 0) {
      if (notification) close();
      return;
    }
    if (!notification) {
      const notifications = document.querySelector(".notifications");
      notification = this._notifications[id] = document.createElement("div");
      notification.innerHTML = `<div class="notification-header"><div></div><i class="ti ti-x"></i></div>
        <span class="detail"></span>
        <div class="progress"></div>`;
      notifications.appendChild(notification);
      notification.querySelector(".notification-header > i").addEventListener("click", () => {
        const event = new Event("close");
        event.id = id;
        event.close = close;
        if (!event.defaultPrevented && notification._onclose) notification._onclose(event);
        if (!event.defaultPrevented) notification.dispatchEvent(event);
        if (!event.defaultPrevented) close();
      });
    }
    notification._onclose = onclose;
    notification.className = `notification notification-${type}`;
    notification.style.setProperty("--progress", `${progress * 100}%`);
    notification.querySelector(".notification-header div").textContent = header;
    if (detailHTML !== undefined)
      notification.querySelector(".detail").innerHTML = detailHTML;
    else
      notification.querySelector(".detail").textContent = detail;
    notification.querySelector(".progress").style.display = progress !== undefined ? "block" : "none";
    notification.querySelector(".notification-header > i").style.display = closeable ? "block" : "none";
    if (notification.autocloseInterval) {
      clearInterval(notification.autocloseInterval);
      notification.autocloseInterval = undefined;
    }
    if (autoclose !== undefined) {
      notification.autocloseInterval = setTimeout(() => {
        notification.remove();
        delete this._notifications[id];
      }, autoclose);
    }
    return notification;
  }

  _draw_background() {
    if (!this._last_frames[0]) return;
    if (!this.state.preview_is_preview_mode) {
      if (this._backgroundTexture === undefined) {
        this._backgroundTexture = new THREE.Texture(this._last_frames[0]);
        this._backgroundTexture.colorSpace = THREE.SRGBColorSpace;
        this._backgroundTexture.flipY = true;
        this.renderer_scene.background = this._backgroundTexture;
      } else if (this._backgroundTexture.image.width !== this._last_frames[0].width || this._backgroundTexture.image.height !== this._last_frames[0].height) {
        // Dispose the old texture
        this._backgroundTexture.dispose();
        this._backgroundTexture = new THREE.Texture(this._last_frames[0]);
        this._backgroundTexture.colorSpace = THREE.SRGBColorSpace;
        this._backgroundTexture.flipY = true;
        this.renderer_scene.background = this._backgroundTexture;
      } else {
        this._backgroundTexture.image = this._last_frames[0];
      }
      this._backgroundTexture.needsUpdate = true;
    } else {
      // Manually draw the background to the canvas
      // The image will be drawn in the center of the canvas, scaled to fit the canvas
      const image = this._last_frames[0];
      if (image === undefined) return;
      const { width, height } = this._preview_canvas;
      const imageAspect = image.width / image.height;
      const canvasAspect = width / height;
      const scale = imageAspect > canvasAspect ? width / image.width : height / image.height;
      const scaledWidth = image.width * scale;
      const scaledHeight = image.height * scale;
      const x = (width - scaledWidth) / 2;
      const y = (height - scaledHeight) / 2;
      this._preview_context.fillStyle = "black";
      this._preview_context.fillRect(0, 0, width, height);
      this._preview_context.save();
      this._preview_context.scale(1, -1);
      this._preview_context.drawImage(image, x, -(y + scaledHeight), scaledWidth, scaledHeight);
      this._preview_context.restore();
    }
  }
}

function buildMethodInfo(info) {
  let out = "";
  if (info.method_id !== undefined)
    out += `<strong>Method ID:</strong><span>${info.method_id}</span>\n`;
  if (info.name !== undefined)
    out += `<strong>Name:</strong><span>${info.name}</span>\n`;
  if (info.link !== undefined)
    out += `<strong>Web:</strong><a href="${info.link}">${info.link}</a>\n`;
  if (info.description !== undefined)
    out += `<strong>Description:</strong><span>${info.description}</span>\n`;
  if (info.paper_title !== undefined) {
    if (info.paper_link !== undefined)
      out += `<strong>Paper:</strong><a href=${info.paper_link}>${info.paper_title}</a>\n`;
    else
      out += `<strong>Paper:</strong><span>${info.paper_title}</span>\n`;
  }
  if (info.paper_authors !== undefined)
    out += `<strong>Paper authors:</strong><span>${info.paper_authors.join(', ')}</span>\n`;
  if (method_info.nb_version !== undefined)
    out += `<strong>NB version:</strong><span>${method_info.nb_version}</span>\n`;
  if (method_info.presets !== undefined)
    out += `<strong>Presets:</strong><span>${method_info.presets.join(', ')}</span>\n`;
  if (method_info.config_overrides !== undefined) {
    const config_overrides = ""
    for (const k in method_info.config_overrides) {
      const v = method_info.config_overrides[k];
      config_overrides += `${k} = ${v}<br/>\n`
    }
    out += `<strong>Config overrides:</strong><span>${config_overrides}</span>\n`;
  }
  return out;
}

function buildDatasetInfo(info) {
  let out = "";
  if (info.id !== undefined)
    out += `<strong>Dataset ID:</strong><span>${info.id}</span>\n`;
  if (info.name !== undefined)
    out += `<strong>Name:</strong><span>${info.name}</span>\n`;
  if (info.link !== undefined)
    out += `<strong>Web:</strong><a href="${info.link}">${info.link}</a>\n`;
  if (info.description !== undefined)
    out += `<strong>Description:</strong><span>${info.description}</span>\n`;
  if (info.evaluation_protocol !== undefined)
    out += `<strong>Eval. protocol:</strong><span>${info.evaluation_protocol}</span>\n`;
  if (info.paper_title !== undefined) {
    if (info.paper_link !== undefined)
      out += `<strong>Paper:</strong><a href=${info.paper_link}>${info.paper_title}</a>\n`;
    else
      out += `<strong>Paper:</strong><span>${info.paper_title}</span>\n`;
  }
  if (info.paper_authors !== undefined)
    out += `<strong>Paper authors:</strong><span>${info.paper_authors.join(', ')}</span>\n`;
  return out;
}


export function makeMatrix4(elements) {
  if (!elements || elements.length !== 12) {
    throw new Error("Invalid elements array. Expected 12 elements.");
  }
  return new THREE.Matrix4().set(...elements, 0, 0, 0, 1);
}

function matrix4ToArray(matrix) {
  const e = matrix.elements;
  return [e[0], e[4], e[8], e[12], e[1], e[5], e[9], e[13], e[2], e[6], e[10], e[14]];
}

export function formatMatrix4(matrix, round) {
  let a = matrix4ToArray(matrix);
  if (round !== undefined) {
    // For each element, round to the nearest `round` decimal places
    a = a.map(x => x.toFixed(round)).map(x => x.startsWith("-") ? x : ` ${x}`);
    // Add trailing zeros to ensure the number of decimal places is `round`
  }
  return `${a[0]}, ${a[1]}, ${a[2]}, ${a[3]},\n${a[4]}, ${a[5]}, ${a[6]}, ${a[7]},\n${a[8]}, ${a[9]}, ${a[10]}, ${a[11]}`
}
