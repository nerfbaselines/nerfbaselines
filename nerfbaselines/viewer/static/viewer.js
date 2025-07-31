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
import palettes from './palettes.js';


const notification_autoclose = 5000;

const notempty = (x) => x !== undefined && x !== null && x !== "";


function drawChart({ svg, ...data }) {
  // Clear the SVG
  while (svg.firstChild) {
    svg.removeChild(svg.firstChild);
  }
  svg.remoteSvgListeners?.();
  const computedStyles = getComputedStyle(svg);
  const aspectRatio = computedStyles.aspectRatio;
  const aspect = aspectRatio.split('/')[0] / aspectRatio.split('/')[1];
  if (!aspect) {
    throw new Error("Aspect ratio must be defined for the SVG element.");
  }

  const svgNs = "http://www.w3.org/2000/svg"
  const width = 287;
  const height = width / aspect;
  const axiso = 15;
  const axisStroke = 1;
  console.log(computedStyles);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.style.width = "100%";
  svg.style['aspect-ratio'] = `${aspect}`;
  svg.style.height = "auto";

  const bottom = height - 12;
  const left = 20;
  const right = width - 15;
  const top = 15;

  const polyline = document.createElementNS(svgNs, 'polyline');
  polyline.setAttribute('fill', 'none');
  polyline.setAttribute('stroke-width', '2');
  polyline.style.stroke = 'var(--theme-color)';
  svg.appendChild(polyline);

  // Add x-axis
  const xAxis = document.createElementNS(svgNs, 'line');
  xAxis.setAttribute('x1', left-axisStroke/2);
  xAxis.setAttribute('y1', bottom);
  xAxis.setAttribute('x2', right+axiso-5);
  xAxis.setAttribute('y2', bottom);
  xAxis.style.stroke = 'var(--primary-color)';
  xAxis.setAttribute('stroke-width', axisStroke);
  svg.appendChild(xAxis);
  // Add x-axis arrowhead
  const xArrowhead = document.createElementNS(svgNs, 'polygon');
  xArrowhead.setAttribute('points', `${right+axiso},${bottom} ${right+axiso-8},${bottom-4} ${right+axiso-8},${bottom+4}`);
  xArrowhead.style.fill = 'var(--primary-color)';
  svg.appendChild(xArrowhead);

  // Add y-axis
  const yAxis = document.createElementNS(svgNs, 'line');
  yAxis.setAttribute('x1', left);
  yAxis.setAttribute('y1', bottom+axisStroke/2);
  yAxis.setAttribute('x2', left);
  yAxis.setAttribute('y2', top-10);
  yAxis.style.stroke = 'var(--primary-color)';
  yAxis.setAttribute('stroke-width', axisStroke);
  svg.appendChild(yAxis);
  // Add y-axis arrowhead
  const yArrowhead = document.createElementNS(svgNs, 'polygon');
  yArrowhead.setAttribute('points', `${left},${top-15} ${left-4},${top-5} ${left+4},${top-5}`);
  yArrowhead.style.fill = 'var(--primary-color)';
  svg.appendChild(yArrowhead);

  // Add mouse focus line
  const focusLine = document.createElementNS(svgNs, 'line');
  focusLine.setAttribute('x1', -100);
  focusLine.setAttribute('y1', bottom+axisStroke/2);
  focusLine.setAttribute('x2', -100);
  focusLine.setAttribute('y2', top-10);
  focusLine.style.stroke = 'var(--secondary-color)';
  focusLine.setAttribute('stroke-width', axisStroke);
  svg.appendChild(focusLine);

  // Add mouse focus circle
  const focusCircle = document.createElementNS(svgNs, 'circle');
  focusCircle.setAttribute('cx', -100);
  focusCircle.setAttribute('cy', -100);
  focusCircle.setAttribute('r', 3);
  focusCircle.style.fill = 'var(--theme-color)';
  svg.appendChild(focusCircle);
  
  // Add mouse focus text
  const focusText = document.createElementNS(svgNs, 'text');
  focusText.setAttribute('x', -100);
  focusText.setAttribute('y', -100);
  focusText.setAttribute('text-anchor', 'middle');
  focusText.setAttribute('font-size', '12');
  focusText.setAttribute('font-weight', 'bold');
  focusText.style.fill = 'var(--highlight-color)';
  focusText.textContent = '';
  svg.appendChild(focusText);

  // Add axis labels
  const xLabelEl = document.createElementNS(svgNs, 'text');
  xLabelEl.setAttribute('x', width- 10);
  xLabelEl.setAttribute('y', bottom-5);
  xLabelEl.setAttribute('text-anchor', 'end');
  xLabelEl.setAttribute('font-size', '12');
  xLabelEl.style.fill = 'var(--primary-color)';
  xLabelEl.style.visibility = 'hidden';
  svg.appendChild(xLabelEl);
  // Add y-axis label
  const yLabelEl = document.createElementNS(svgNs, 'text');
  yLabelEl.setAttribute('x', left+8);
  yLabelEl.setAttribute('y', top);
  yLabelEl.setAttribute('text-anchor', 'begin');
  yLabelEl.setAttribute('font-size', '12');
  yLabelEl.style.fill = 'var(--primary-color)';
  yLabelEl.style.visibility = 'hidden';
  svg.appendChild(yLabelEl);

  // Prepare x-axis ticks
  const xtickElements = Array.from({ length: 3 }, (_, i) => {
    const tick = document.createElementNS(svgNs, 'line');
    tick.setAttribute('y1', bottom-2);
    tick.setAttribute('y2', bottom+2);
    tick.style.stroke = 'var(--primary-color)';
    tick.setAttribute('stroke-width', axisStroke);
    svg.appendChild(tick);

    // Add x-axis labels
    const label = document.createElementNS(svgNs, 'text');
    label.setAttribute('y', bottom+10);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('font-size', '10');
    label.style.fill = 'var(--primary-color)';
    svg.appendChild(label);
    return { tick, label };
  });

  // Prepare y-axis ticks
  const ytickElements = Array.from({ length: 3 }, (_, i) => {
    const tick = document.createElementNS(svgNs, 'line');
    tick.setAttribute('x1', left-2);
    tick.setAttribute('x2', left+2);
    tick.style.stroke = 'var(--primary-color)';
    tick.setAttribute('stroke-width', axisStroke);
    svg.appendChild(tick);

    // Add y-axis labels
    const label = document.createElementNS(svgNs, 'text');
    label.setAttribute('x', left-3);
    label.setAttribute('text-anchor', 'end');
    label.setAttribute('dominant-baseline', 'middle');
    label.setAttribute('font-size', '10');
    label.style.fill = 'var(--primary-color)';
    svg.appendChild(label);
    return { tick, label };
  });

  let getYValue = null;

  // Update with the data
  const setData = ({ x, y, xLabel, yLabel }) => {
    let xmax = parseFloat(Math.max(...x).toPrecision(2));
    let xmin = parseFloat(Math.min(...x).toPrecision(2));
    let ymax = parseFloat(Math.max(...y).toPrecision(2));
    let ymin = parseFloat(Math.min(...y).toPrecision(2));
    const xScale = (right-left) / (x[x.length - 1] - x[0]);
    const yScale = (bottom-top) / (y[y.length - 1] - y[0]);

    xtickElements.forEach(({ tick, label }, i) => {
      const xValue = parseFloat(xmin+(xmax-xmin)*(i+1)/xtickElements.length).toPrecision(2);
      tick.setAttribute('x1', left+xValue*xScale);
      tick.setAttribute('x2', left+xValue*xScale);
      label.setAttribute('x', left+xValue*xScale);
      label.textContent = (parseFloat(xValue)).toPrecision(2);
    });
    ytickElements.forEach(({ tick, label }, i) => {
      const yValue = parseFloat(ymin+(ymax-ymin)*i/(ytickElements.length-1)).toPrecision(2);
      tick.setAttribute('y1', bottom-yValue*yScale);
      tick.setAttribute('y2', bottom-yValue*yScale);
      label.setAttribute('y', bottom-yValue*yScale+1);
      label.textContent = (parseFloat(yValue)).toPrecision(2);
    });

    const points = x.map((x, i) => `${left+x * xScale},${bottom-y[i] * yScale}`).join(' ');
    polyline.setAttribute('points', points);

    // Set axis labels
    if (xLabel) {
      xLabelEl.style.visibility = 'visible';
      xLabelEl.textContent = xLabel;
    } else { xLabelEl.style.visibility = 'hidden'; }
    if (yLabel) {
      yLabelEl.style.visibility = 'visible';
      yLabelEl.textContent = yLabel;
    } else { yLabelEl.style.visibility = 'hidden'; }
    
    // Set getYValue function
    getYValue = (vOrig) => {
      const v = (vOrig - left) / xScale;
      let yValue;
      if (v < x[0]) yValue = y[0];
      else if (v > x[x.length - 1]) yValue = y[y.length - 1];
      else {
        for (let i = 0; i < x.length - 1; i++) {
          if (x[i] <= v && v <= x[i+1]) {
            const t = (v - x[i]) / (x[i+1] - x[i]);
            yValue = y[i] * (1-t) + y[i+1] * t;
            break;
          }
        }
      }
      const out = bottom-yValue*yScale;
      return { y: out, yValue };
    };
  };

  const hideMouseFocus = () => {
    focusLine.setAttribute('x1', -100);
    focusLine.setAttribute('x2', -100);
    focusCircle.setAttribute('cx', -100);
    focusCircle.setAttribute('cy', -100);
    focusText.setAttribute('x', -100);
    focusText.setAttribute('y', -100);
  };

  const mousemove = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const svgLeft = svg.getBoundingClientRect().left;
    const x = (e.clientX - svgLeft) / svg.clientWidth * width;
    if (x < left || x > right || !getYValue) {
      // Make invisible
      return;
    }
    const {y, yValue } = getYValue(x);
    focusLine.setAttribute('x1', x);
    focusLine.setAttribute('x2', x);
    focusCircle.setAttribute('cx', x);
    focusCircle.setAttribute('cy', y);
    focusText.setAttribute('x', x);
    focusText.setAttribute('y', y-8);
    focusText.textContent = yValue.toPrecision(5);
  };
  const mouseleave = (e) => {
    if (e.target !== svg) return;
    e.preventDefault();
    e.stopPropagation();
    hideMouseFocus();
  };

  svg.addEventListener('mousemove', mousemove, { passive: false, capture: true });
  svg.addEventListener('mouseleave', mouseleave, { passive: false, capture: true });
  svg.removeSvgListeners = () => {
    svg.removeEventListener('mousemove', mousemove, { passive: false, capture: true });
    svg.removeEventListener('mouseleave', mouseleave, { passive: false, capture: true });
  }
  svg.setData = setData;
  setData(data);
}


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
      if (ex.name === 'AbortError') {
        throw ex;
      }
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
      firstTimestampBehavior: 'offset',
      options: {},
    });
    const videoEncoder = new VideoEncoder({
      output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
      error: e => {
        console.error(e)
        errors.push(e);
      },
    });
    let bitrate = Math.round(
      0.1 * this.width * this.height * this.fps
    );
    let quantizer;
    let bitrateMode = "variable";
    let [codec, ...codecParams] = this.codec.split(";").map(s => s.trim());
    codecParams.forEach(param => {
      try {
        let [key, value] = param.split("=");
        if (key === "bppf") {
          const bppf = parseFloat(value);
          bitrate = Math.round(bppf * this.width * this.height * this.fps);
          return;
        }
        if (key === "bps") {
          bitrate = parseInt(value); return;
        }
        if (key === "crf") {
          bitrateMode = "quantizer";
          quantizer = parseInt(value); return;
        }
      } catch (e) {
        throw new Error(`Invalid codec parameter: ${param}`);
      }
      throw new Error(`Unknown codec parameter: ${key}`);
    });

    videoEncoder.configure({
      codec: codec,
      width: this.width,
      height: this.height,
      framerate: Math.round(this.fps),
      bitrate,
      bitrateMode,
      latencyMode: "quality",
    });
    let lastKeyframeTimestamp = -Infinity;
    this.addFrame = async (image, { repeats } = {}) => {
      repeats = repeats || 1;
      for (let i = 0; i < repeats; i++) {
        repeats = repeats || 1;
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
        videoEncoder.encode(frame, { 
          keyFrame,
          vp9: { quantizer },
          av1: { quantizer },
          avc: { quantizer },
          hevc: { quantizer },
        });
        frame.close();
        framesGenerated++;
      }
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

  _on_change({ state, property, trigger }) {
    const defaultSettings = SettingsManager.get_default_settings();
    if (trigger !== "gui_input" && trigger !== "gui_change") return;
    if (property !== undefined && defaultSettings[property] !== undefined) {
      let settings = {};
      const settingsJSON = localStorage.getItem(`settings`);
      if (settingsJSON && settingsJSON !== "") {
        settings = JSON.parse(settingsJSON);
      }
      settings[property] = state[property];
      localStorage.setItem(`settings`, JSON.stringify(settings));
    }
  }

  static get_default_settings() {
    return {
      theme_color: "#ffd369",
      trajectory_curve_color: "#6bffe6",
      player_frustum_color: "#20df80",
      keyframe_frustum_color: "#ff0000",
      dataset_frustum_color: "#d3d3d3",
      notification_autoclose: 5000,
      dataset_show_train_cameras: false,
      dataset_show_test_cameras: false,
      dataset_show_pointcloud: false,

      camera_control_inertia: 0.6,
      camera_control_rotation_sensitivity: 1.0,
      camera_control_pan_sensitivity: 1.0,
      camera_control_zoom_sensitivity: 1.0,
      camera_control_key_speed: 1.0,

      camera_control_rotation_inverted: false,
      camera_control_pan_inverted: false,
      camera_control_zoom_inverted: false,

      camera_path_render_mp4_codec: 'avc1.640028;bppf=0.15',
      camera_path_render_webm_codec: 'vp09.02.40.10;bppf=0.12',
      camera_path_render_keyframe_interval: '5s',

      viewer_theme: 'dark',
      viewer_font_size: 1,

      output_palette: 'viridis',
      split_palette: 'viridis',
    }
  }

  reset() {
    localStorage.clear();
    Object.assign(this.viewer.state, SettingsManager.get_default_settings());
    this.viewer.notifyChange({ property: undefined });
  }

  _populate_state() {
    const state = this.viewer.state;
    const settings = SettingsManager.get_default_settings();
    try {
      const json = localStorage.getItem(`settings`);
      const settingsUpdate = JSON.parse(json);
      Object.assign(settings, settingsUpdate);
    } catch (e) {
      console.error(e);
      localStorage.clear();
    }
    Object.assign(state, settings);
    this.viewer.notifyChange({ property: undefined });
  }
}


function _attach_persistent_state(viewer) {
  let changed = true;
  const sessionId = window.location.search;
  viewer.addEventListener("change", ({ property, state, trigger }) => {
    changed = true;
  });
  viewer.addEventListener("start", ({ state }) => {
    if (sessionStorage.getItem('viewer_state'+sessionId) === null) {
      return;
    }
    const { 
      state: savedState, cameraMatrix: cameraMatrixArray, cameraUpVector: cameraUpVectorArray,
    } = JSON.parse(sessionStorage.getItem('viewer_state'+sessionId));
    // Fix types in the saved state
    if (savedState.camera_path_keyframes) {
      for (const keyframe of savedState.camera_path_keyframes) {
        keyframe.position = new THREE.Vector3(
          keyframe.position.x, 
          keyframe.position.y, 
          keyframe.position.z
        );
        keyframe.quaternion = new THREE.Quaternion().fromArray(keyframe.quaternion);
      }
    }
    const cameraMatrix = cameraMatrixArray ? makeMatrix4(cameraMatrixArray) : undefined;
    const cameraUpVector = cameraUpVectorArray ? new THREE.Vector3().fromArray(cameraUpVectorArray) : undefined;
    viewer.dispatchEvent("loading_state", { 
      state: savedState,
      cameraMatrix,
      cameraUpVector,
    });
    Object.assign(state, savedState);
    if (cameraMatrix) {
      viewer.scene.updateMatrixWorld();
      viewer.set_camera({ matrix: cameraMatrix });
    }
    if (cameraUpVector) {
      viewer.camera.up.copy(cameraUpVector);
      viewer.controls.updateUp();
    }
    viewer.notifyChange({ property: undefined, trigger: "restore_state" });
  });
  setInterval(() => {
    if (changed) {
      const state = {...viewer.state};
      delete state.dataset_has_pointcloud;
      delete state.dataset_has_train_cameras;
      delete state.dataset_has_test_cameras;
      delete state.dataset_images;
      delete state.viewer_is_embedded;
      delete state.viewer_public_url;
      delete state.output_types;

      delete state.dataset_info;
      delete state.method_info;

      // Delete computed properties
      for (let k of viewer._computed_property_names) {
        delete state[k];
      }

      // Delete default settings
      for (let k in SettingsManager.get_default_settings()) {
        delete state[k];
      }
      let cameraMatrix = viewer.get_camera_params().matrix;
      viewer.dispatchEvent("saving_state", { state, cameraMatrix, cameraUpVector: viewer.camera.up });
      cameraMatrix = matrix4ToArray(cameraMatrix);
      sessionStorage.setItem('viewer_state'+sessionId, JSON.stringify({ 
        state, 
        cameraMatrix, 
        cameraUpVector: viewer.camera.up.toArray() }));
    }
  }, 300);
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


async function evaluateES6(code) {
  const objectURL = URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
  try {
    return (await import(objectURL)).default;
  } finally {
    URL.revokeObjectURL(objectURL);
  }
}


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
      throw ex;
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
        property !== 'camera_path_time_interpolation' &&
        property !== 'camera_path_distance_alpha' &&
        property !== 'camera_path_default_transition_duration' &&
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
      camera_path_distance_alpha,
      camera_path_time_interpolation,
      camera_path_default_transition_duration,
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
        distance_alpha: camera_path_distance_alpha,
        time_interpolation: camera_path_time_interpolation,
        default_transition_duration: camera_path_default_transition_duration,
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


function _attach_output_configuration(viewer) {
  const self_trigger = "selected_output_configuration";
  viewer.addEventListener("change", ({ property, state, trigger }) => {
    if (property === undefined ||
        property === 'output_type' ||
        property === 'outputs_configuration') {
      const output_configuration = state.outputs_configuration?.[state.output_type] || {};
      state.output_range_min = notempty(output_configuration.range_min) ? output_configuration.range_min : "";
      viewer.notifyChange({ property: "output_range_min", trigger: self_trigger });
      state.output_range_max = notempty(output_configuration.range_max) ? output_configuration.range_max : "";
      viewer.notifyChange({ property: "output_range_max", trigger: self_trigger });
      state.output_palette_enabled = output_configuration.palette_enabled || false;
      viewer.notifyChange({ property: "output_palette_enabled", trigger: self_trigger });
    }

    if (property === undefined ||
        property === 'split_output_type' ||
        property === 'outputs_configuration') {
      const output_configuration = state.outputs_configuration?.[state.split_output_type] || {};
      let value = output_configuration.split_range_min;
      if (value === undefined) {
        value = output_configuration.range_min || "";
      }
      state.split_range_min = value;
      viewer.notifyChange({ property: "split_range_min", trigger: self_trigger });
      value = output_configuration.split_range_max;
      if (value === undefined) {
        value = output_configuration.range_max || "";
      }
      state.split_range_max = value;
      viewer.notifyChange({ property: "split_range_max", trigger: self_trigger });
      state.split_palette_enabled = output_configuration.palette_enabled || false;
      viewer.notifyChange({ property: "split_palette_enabled", trigger: self_trigger });
    }

    if (property === 'output_range_min' && trigger !== self_trigger) {
      let value = state.output_range_min;
      const output_configuration = state.outputs_configuration?.[state.output_type] || {};
      if (output_configuration.range_min !== value) {
        state.outputs_configuration = {
          ...state.outputs_configuration,
          [state.output_type]: {
            ...output_configuration,
            range_min: value,
          },
        };
      }
      viewer.notifyChange({ property: "outputs_configuration", trigger: self_trigger });
    }

    if (property === 'output_range_max' && trigger !== self_trigger) {
      let value = state.output_range_max;
      const output_configuration = state.outputs_configuration?.[state.output_type] || {};
      if (output_configuration.range_max !== value) {
        state.outputs_configuration = {
          ...state.outputs_configuration,
          [state.output_type]: {
            ...output_configuration,
            range_max: value,
          },
        };
      }
      viewer.notifyChange({ property: "outputs_configuration", trigger: self_trigger });
    }

    if (property === 'split_range_min' && trigger !== self_trigger) {
      let value = state.split_range_min;
      const output_configuration = state.outputs_configuration?.[state.split_output_type] || {};
      if (output_configuration.split_range_min !== value) {
        state.outputs_configuration = {
          ...state.outputs_configuration,
          [state.split_output_type]: {
            ...output_configuration,
            split_range_min: value,
          },
        };
      }
      viewer.notifyChange({ property: "outputs_configuration", trigger: self_trigger });
    }

    if (property === 'split_range_max' && trigger !== self_trigger) {
      let value = state.split_range_max;
      const output_configuration = state.outputs_configuration?.[state.split_output_type] || {};
      if (output_configuration.split_range_max !== value) {
        state.outputs_configuration = {
          ...state.outputs_configuration,
          [state.split_output_type]: {
            ...output_configuration,
            split_range_max: value,
          },
        };
      }
      viewer.notifyChange({ property: "outputs_configuration", trigger: self_trigger });
    }
  });
}


class HTTPFrameRenderer {
  constructor({ url }) {
    this.url = url;
  }

  async render(params, { flipY = false } = {}) {
    if (!params.output_type) return;
    const response = await fetch(`${this.url}`,{
      method: "POST",  // Disable caching
      cache: "no-cache",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      let errorMessage = "Connection to HTTP renderer failed";
      try {
        const error = await response.json();
        errorMessage = error.message;
      } catch (e) {}
      throw new Error(errorMessage);
    }

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
          const onopen = () => {
            socket.removeEventListener("open", onopen);
            socket.removeEventListener("close", onclose);
            socket.removeEventListener("error", onclose);
            console.log("WebSocket connection established");
            resolve(socket);
          };
          const onclose = (e) => {
            console.log("WebSocket connection closed");
            socket.removeEventListener("open", onopen);
            socket.removeEventListener("close", onclose);
            socket.removeEventListener("error", onclose);
            reject(new Error("WebSocket connection closed"));
          };
          const socket = new WebSocket(this.url);
          socket.binaryType = "blob";
          socket.addEventListener("open", onopen);
          socket.addEventListener("close", onclose);
          socket.addEventListener("error", onclose);
        } catch (error) {
          reject(error);
        }
      });
      const closeAll = (message) => {
        const sub = this._subscriptions;
        this._subscriptions = {};
        for (const [thread, [resolve, reject]] of Object.entries(sub)) {
          reject({
            thread,
            status: "error",
            error: message,
          });
        }
      }
      this._socket.addEventListener("close", () => {
        console.log("WebSocket connection closed");
        this._socket = undefined;
        closeAll("WebSocket connection closed");
      });
      this._socket.addEventListener("error", (error) => {
        console.error("WebSocket error:", error);
        closeAll("WebSocket error");
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
    if (!params.output_type) return;
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


function getWorkerScriptURL(aURL) {
  const esModuleShimsURL = new URL("./third-party/es-module-shims.wasm.js", import.meta.url);
  const baseURL = new URL("./", import.meta.url);
  const importMap = {
    imports: {
      "three": new URL("./third-party/three.module.js", import.meta.url),
      "three/addons/": new URL("./third-party/", import.meta.url),
    }
  };
  return URL.createObjectURL(new Blob(
    [
      `importScripts('${new URL(esModuleShimsURL, baseURL).href}');
      importShim.addImportMap(${JSON.stringify(importMap)});
      importShim('${new URL(aURL, baseURL).href}').catch(e => setTimeout(() => { throw e; }))`
    ],
    { type: 'application/javascript' }));
}


export class WorkerFrameRenderer {
  constructor({ 
    update_notification,
    onready,
    ...options
  }) {
    this._options = options;
    this._onready = onready;
    this._notificationId = "WorkerFrameRenderer-" + (
      WorkerFrameRenderer._notificationIdCounter = (WorkerFrameRenderer._notificationIdCounter || 0) + 1);
    this.update_notification = update_notification;
    this._renderPromises = {};
    this._requestCounter = 0;
    this._isReady = false;
    if (window.Worker === undefined) {
      console.error("WorkerFrameRenderer: Web Workers are not supported, falling back to main thread");
      this.fallback();
    } else {
      this._setup_worker(options);
    }
  }

  _setup_worker(options) {
    this._worker = this.setup_worker(options);
    this._worker.onmessage = (e) => {
      const { type, ...data } = e.data;
      if (type === "loaded") {
        // Worker loaded, we can handle ther rest
        this._worker.postMessage({ 
          type: "init", 
          options, 
        });
      }
      if (type === "notification") {
        data.notification.id = this._notificationId;
        this.update_notification?.(data.notification);
      }
      if (type === "ready") {
        this._isReady = true;
        this._onready?.(data);
      }
      if (type === "rendered") {
        if (this._renderPromises[data.requestId]) {
          const [ resolve, reject ] = this._renderPromises[data.requestId];
          if (data.error) reject(data.error);
          else resolve(data.imageBitmap);
        }
      }
    };
    this._worker.onerror = (e) => {
      // This is a global error, not a message error
      // The error event is fired when the worker could not be loaded
      // We will fallback to the main thread
      console.error("Worker: Error received from worker");
      this.fallback();
    };
    this._worker.onmessageerror = (e) => {
      console.error("Worker: Message error received from worker");
    };
  }

  async fallback() {
    // Kill worker if running
    this._worker?.terminate();
    this._worker = undefined;

    // Fallback to main thread
    try {
      this._renderer = await this.setup_renderer({
        ...this._options,
        update_notification: (notification) => {
          notification.id = this._notificationId;
          this.update_notification?.(notification);
        },
        onready: () => {
          this._isReady = true;
          this._onready?.();
        }
      });
    } catch (error) {
      console.error("Failed to fallback to main thread");
      console.error(error);
      this.update_notification?.({
        id: this._notificationId,
        header: "Failed to fallback to main thread",
        detail: error.message,
        type: "error",
        closeable: true,
      });
    }
  }

  render(params, options) {
    if (!this._isReady) return undefined;
    if (this._renderer) {
      return this._renderer.render(params, options);
    }

    const requestId = this._requestCounter++;
    return new Promise((resolve, reject) => {
      try {
        this._renderPromises[requestId] = [resolve, reject];
        this._worker.postMessage({ 
          type: "render", 
          params, 
          options,
          requestId,
        })
      } catch (error) {
        reject(error);
      }
    });
  }

  setup_worker(options) { throw new Error("Not implemented"); }
  setup_renderer(options) { throw new Error("Not implemented"); }
}


export class MeshFrameRenderer extends WorkerFrameRenderer {
  setup_worker() {
    return new Worker(getWorkerScriptURL("./mesh.js"));
  }

  async setup_renderer(options) {
    const mesh = await import("./mesh.js");
    return new mesh.MeshFrameRenderer(options);
  }
}


export class GaussianSplattingFrameRenderer extends WorkerFrameRenderer {
  setup_worker() {
    return new Worker(getWorkerScriptURL("./3dgs.js"));
  }

  async setup_renderer(options) {
    const mesh = await import("./3dgs.js");
    return new mesh.GaussianSplattingFrameRenderer(options);
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
    pointcloud_url,
  }) {
    this.notificationId = "DatasetManager-" + (
      DatasetManager._notificationIdCounter = (DatasetManager._notificationIdCounter || 0) + 1);
    this.viewer = viewer;
    this.scene = viewer.scene;
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

    // Load parts
    this._pointcloud_url = pointcloud_url;
    if (pointcloud_url) {
      this.viewer.state.dataset_has_pointcloud = true;
      this.viewer.notifyChange({ property: 'dataset_has_pointcloud' });
    } else {
      this.viewer.state.dataset_has_pointcloud = false;
      this.viewer.notifyChange({ property: 'dataset_has_pointcloud' });
    }
    this._train_load_images_tasks = null;
    this._test_load_images_tasks = null;
    if (url) {
      const absUrl = new URL(url, window.location.href).href;
      this._load_cameras(absUrl).then(() => {
        if (state.dataset_show_train_cameras && this._train_load_images_tasks !== null)
          this._load_split_images("train");
        if (state.dataset_show_test_cameras && this._test_load_images_tasks !== null)
          this._load_split_images("test");
        if (state.dataset_show_pointcloud && this._pointcloud_url) {
          this._load_pointcloud(this._pointcloud_url);
        }
      });
    }
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
          this.viewer.update_notification({
            id: `${this.notificationId}-${split}`, header: "", autoclose: 0,
          });
        }
      }
    }
    if (state[`dataset_has_pointcloud`] && property === 'dataset_show_pointcloud') {
      if (state.dataset_show_pointcloud)
        this._load_pointcloud(this._pointcloud_url);
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
    const taskDefs = this[`_${split}_load_images_tasks`] || [];
    if (this[`_${split}_loading_started`]) return;

    try {
      if (this[`_${split}_loading_started`]) return;
      this[`_${split}_loading_started`] = true;
      let errors = [];
      const all_loaded = this[`_${split}_cameras`].children.every((frustum) => frustum._hasImage);
      if (all_loaded) return;
      let tasks = [];
      let num_loaded = 0;
      const cancel = () => {
        this[`_${split}_loading_started`] = false;
        this.viewer.state[`dataset_show_${split}_cameras`] = false;
        this.viewer.notifyChange({ property: `dataset_show_${split}_cameras` });
      };
      this.viewer.update_notification({
        id: `${this.notificationId}-${split}`,
        header: `Loading ${split} dataset images`,
        progress: 0,
        onclose: cancel,
        closeable: true,
      });
      for (let _i = 0; _i < taskDefs.length; ++_i) {
        const { id, thumbnail_url } = taskDefs[_i];
        tasks.push(async () => {
          if (!this.viewer.state[`dataset_show_${split}_cameras`]) return;
          try {
            const frustum = this._frustums[id];
            if (frustum._hasImage) {
              ++num_loaded;
              return;
            }

            // Replace image_path extension with .jpg
            const response = await fetch(thumbnail_url);
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
          this.viewer.update_notification({
            id: `${this.notificationId}-${split}`,
            header: `Loading ${split} dataset images`,
            progress: ++num_loaded / taskDefs.length,
            onclose: cancel,
            closeable: true,
          });
        });
      }

      await promise_parallel_n(tasks, 8);
      if (!this.viewer.state[`dataset_show_${split}_cameras`]) return;

      if (errors.length > 0) {
        this.viewer.update_notification({
          id: `${this.notificationId}-${split}`,
          header: `Failed to load ${split} dataset images`,
          detail: errors[0].message,
          type: "error",
          closeable: true,
        });
      } else if (num_loaded === taskDefs.length) {
        this.viewer.update_notification({
          id: `${this.notificationId}-${split}`,
          header: `Loaded ${split} dataset images`,
          autoclose: notification_autoclose,
          closeable: true,
        });
      }
    } finally {
      this[`_${split}_loading_started`] = false;
    }
  }

  async _load_cameras(url) {
    // Load dataset train/test frustums
    let result;
    try {
      const response = await fetch(url);
      if (!response.ok) {
        let error = `Failed to load dataset: ${response.statusText}`;
        try {
          const error = await response.json();
          error = `Failed to load dataset: ${error.message}`;
        } catch (error) {}
        throw new Error(error);
      }
      result = await response.json();
    } catch (error) {
      console.error('An error occurred while loading the cameras:', error);
      this.viewer.update_notification({
        id: `${this.notificationId}-cameras`,
        header: "Error loading dataset cameras",
        detail: error.message,
        type: "error",
      });
      return null;
    }
    if (result.pointcloud_url && !this._pointcloud_url) {
      // Make pointcloud_url relative to the dataset url
      url = new URL(url, window.location.href);
      const pointcloud_url = new URL(result.pointcloud_url, url);
      this._pointcloud_url = pointcloud_url.href;
      this.viewer.state.dataset_has_pointcloud = true;
      this.viewer.notifyChange({ property: 'dataset_has_pointcloud' });
    }
    if (result.metadata) {
      this.viewer.state.dataset_info = (
        Object.assign(result.metadata, this.viewer.state.dataset_info || {})
      );
      this.viewer.notifyChange({ property: 'dataset_info' });
    }
    for (const split of ['train', 'test']) {
      if (!result[split]) continue;
      const tasks = this[`_${split}_load_images_tasks`] = [];
      try {
        const { cameras } = result[split];
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
            image_url: camera.image_url,
            thumbnail_url: camera.thumbnail_url,
            matrix: poseMatrix,
            width, height,
            cx, cy, fx, fy,
          };
          tasks.push({
            id,
            thumbnail_url: camera.thumbnail_url,
          });
          appearance_options.push({
            value: `${i}`,
            label: `${i}: ${camera.image_name}`
          });
          i++;
        }
        this.viewer.notifyChange({ property: 'dataset_images' });
        this.viewer.state[`dataset_${split}_appearance_options`] = appearance_options;
        this.viewer.notifyChange({ property: `dataset_${split}_appearance_options` });
      } catch (error) {
        console.error('An error occurred while loading the cameras:', error);
        this.viewer.update_notification({
          id: `${this.notificationId}-${split}`,
          header: `Error loading dataset ${split} cameras`,
          detail: error.message,
          type: "error",
        });
        this.update_notification();
        return null;
      }
    }
  }

  async _load_pointcloud(url) {
    // Load PLY file
    if (this._cancel_load_pointcloud) return; // Already loading
    if (this._pointcloud.children.length > 0) return; // Already loaded
    const notificationId = this.notificationId + "-pointcloud";
    let cancelled = false;
    const controller = new AbortController();
    this._cancel_load_pointcloud = () => {
      if (!cancelled) {
        controller.abort();
      }
      cancelled = true;
      this.viewer.update_notification({ id: notificationId, autoclose: 0 });
    };
    try {
      // Update progress callback
      const updateProgress = (percentage) => {
        if (cancelled) return;
        this.viewer.update_notification({
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
      let response = await fetch(url, {
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
      this.viewer.update_notification({
        id: notificationId,
        header: "Loaded dataset point cloud",
        closeable: true,
        autoclose: notification_autoclose,
      });
    } catch (error) {
      if (cancelled) return;
      console.error('An error occurred while loading the PLY file:', error);
      this.viewer.update_notification({
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
  const self_trigger = "selected_keyframe_details";
  const getKeyframes = ({ camera_path_keyframes, camera_path_selected_keyframe, camera_path_loop }) => {
    const keyframeIndex = camera_path_keyframes?.findIndex((keyframe) => keyframe.id === camera_path_selected_keyframe);
    const keyframe = keyframeIndex >= 0 ? camera_path_keyframes[keyframeIndex] : undefined;
    const prevKeyframe = (camera_path_keyframes?.length || 0) > 1 ?
      ((camera_path_loop || keyframeIndex > 0) ?
        camera_path_keyframes[(keyframeIndex + camera_path_keyframes.length - 1) % camera_path_keyframes.length] : undefined) : undefined;
    return { keyframe, prevKeyframe, keyframeIndex };
  };

  viewer._computed_property_names.add("camera_path_selected_keyframe_natural_index");
  viewer._computed_property_names.add("camera_path_has_selected_keyframe");
  viewer._computed_property_names.add("camera_path_selected_keyframe_appearance_train_index");
  viewer._computed_property_names.add("camera_path_selected_keyframe_appearance_url");
  viewer._computed_property_names.add("camera_path_selected_keyframe_fov");
  viewer._computed_property_names.add("camera_path_selected_keyframe_override_fov");
  viewer._computed_property_names.add("camera_path_selected_keyframe_velocity_multiplier");
  viewer._computed_property_names.add("camera_path_selected_keyframe_show_in_duration");
  viewer._computed_property_names.add("camera_path_selected_keyframe_show_duration");
  viewer._computed_property_names.add("camera_path_selected_keyframe_override_duration");
  viewer._computed_property_names.add("camera_path_selected_keyframe_override_in_duration");
  viewer._computed_property_names.add("camera_path_selected_keyframe_duration");
  viewer._computed_property_names.add("camera_path_selected_keyframe_in_duration");

  const updateSelectedKeyframe = (state) => {
    const { dataset_images, camera_path_keyframes, 
            camera_path_selected_keyframe, camera_path_loop,
            camera_path_interpolation, camera_path_time_interpolation,
            camera_path_default_transition_duration } = state;
    const { keyframe, prevKeyframe, keyframeIndex } = getKeyframes(state);
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
      camera_path_selected_keyframe_show_in_duration: (
        camera_path_time_interpolation === "time" && (camera_path_loop || keyframeIndex > 0)
      ),
      camera_path_selected_keyframe_show_duration: (
        camera_path_interpolation === "none" || 
        (camera_path_time_interpolation === "time" && (camera_path_loop || keyframeIndex < camera_path_keyframes.length - 1))
      ),
      camera_path_selected_keyframe_override_duration: def(keyframe?.duration),
      camera_path_selected_keyframe_override_in_duration: def(prevKeyframe?.duration),
      camera_path_selected_keyframe_duration: def(keyframe?.duration) ? keyframe.duration : camera_path_default_transition_duration,
      camera_path_selected_keyframe_in_duration: def(prevKeyframe?.duration) ? prevKeyframe.duration : camera_path_default_transition_duration,
    };
    for (const property in change) {
      if (state[property] !== change[property]) {
        state[property] = change[property];
        viewer.notifyChange({ property, trigger: self_trigger });
      }
    }
  }
  viewer.addEventListener("change", ({property, state, trigger}) => {
    if (trigger === self_trigger) return;
    // Add setters
    const { keyframe, prevKeyframe } = getKeyframes(state);
    if (property === "camera_path_selected_keyframe_override_fov" && keyframe) {
      if (state.camera_path_selected_keyframe_override_fov) {
        if (keyframe?.fov === undefined && state.camera_path_default_fov !== undefined) {
          keyframe.fov = state.camera_path_default_fov;
          viewer.notifyChange({ property: "camera_path_keyframes", trigger: self_trigger });
        }
      } else {
        keyframe.fov = undefined;
        viewer.notifyChange({ property: "camera_path_keyframes", trigger: self_trigger });
      }
      return
    }

    if (property === "camera_path_selected_keyframe_override_in_duration" && keyframe) {
      if (state.camera_path_selected_keyframe_override_in_duration) {
        if (prevKeyframe?.duration === undefined && state.camera_path_default_transition_duration !== undefined) {
          prevKeyframe.duration = state.camera_path_default_transition_duration;
          viewer.notifyChange({ property: "camera_path_keyframes", trigger: self_trigger });
        }
      } else {
        prevKeyframe.duration = undefined;
        viewer.notifyChange({ property: "camera_path_keyframes", trigger: self_trigger });
      }
      return
    }

    if (property === "camera_path_selected_keyframe_override_duration" && keyframe) {
      if (state.camera_path_selected_keyframe_override_duration) {
        if (keyframe?.duration === undefined && state.camera_path_default_transition_duration !== undefined) {
          keyframe.duration = state.camera_path_default_transition_duration;
          viewer.notifyChange({ property: "camera_path_keyframes", trigger: self_trigger });
        }
      } else {
        keyframe.duration = undefined;
        viewer.notifyChange({ property: "camera_path_keyframes", trigger: self_trigger });
      }
      return
    }

    if (property === "camera_path_selected_keyframe_velocity_multiplier" && keyframe) {
      if (state.camera_path_selected_keyframe_velocity_multiplier && keyframe.velocity_multiplier !== state.camera_path_selected_keyframe_velocity_multiplier) {
        keyframe.velocity_multiplier = state.camera_path_selected_keyframe_velocity_multiplier;
        viewer.notifyChange({ property: "camera_path_keyframes", trigger: self_trigger });
      }
      return;
    }

    if (property === "camera_path_selected_keyframe_duration" && keyframe) {
      if (state.camera_path_selected_keyframe_override_duration && keyframe.duration !== state.camera_path_selected_keyframe_duration) {
        keyframe.duration = state.camera_path_selected_keyframe_duration;
        viewer.notifyChange({ property: "camera_path_keyframes", trigger: self_trigger });
      }
      return;
    }

    if (property === "camera_path_selected_keyframe_in_duration" && prevKeyframe) {
      if (state.camera_path_selected_keyframe_override_in_duration && 
          prevKeyframe.duration !== state.camera_path_selected_keyframe_in_duration) {
        prevKeyframe.duration = state.camera_path_selected_keyframe_in_duration;
        viewer.notifyChange({ property: "camera_path_keyframes", trigger: self_trigger });
      }
      return;
    }

    if (property === "camera_path_selected_keyframe_fov" && keyframe) {
      if (state.camera_path_selected_keyframe_override_fov && keyframe.fov !== state.camera_path_selected_keyframe_fov) {
        keyframe.fov = state.camera_path_selected_keyframe_fov;
        viewer.notifyChange({ property: "camera_path_keyframes", trigger: self_trigger });
      }
      return;
    }

    if (property === "camera_path_selected_keyframe_appearance_train_index" && keyframe) {
      const index = (
        state.camera_path_selected_keyframe_appearance_train_index === ""
      ) ?  undefined : parseInt(state.camera_path_selected_keyframe_appearance_train_index);
      keyframe.appearance_train_index = index;
      viewer.notifyChange({ property: "camera_path_keyframes", trigger: self_trigger });
      return;
    }
  });

  viewer.addEventListener("change", ({property, state, trigger}) => {
    if (property === undefined ||
        property === "camera_path_keyframes" ||
        property === "camera_path_selected_keyframe" ||
        property === "camera_path_loop" ||
        property === "camera_path_duration" ||
        property === "camera_path_default_transition_duration" ||
        property === "dataset_images") {
      if (trigger === self_trigger) return;
      updateSelectedKeyframe(state);
    }
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
      viewer.controls.keyPanSpeed = viewer.controls.keyRotateSpeed = 10 * trans(state.camera_control_key_speed);
  });
}


function querySelectorAll(elements, selector) {
  const out = [];
  for (const element of elements) {
    out.push(...element.querySelectorAll(selector));
    if (element.matches(selector)) out.push(element);
  }
  return out;
}


function _attach_init_show_in_new_window(viewer) {
  const isInframe = window.location !== window.parent.location;
  viewer.state.viewer_show_in_new_window_visible = isInframe;
  viewer.notifyChange({ property: "viewer_show_in_new_window_visible" });
}


function _attach_draggable_keyframe_panel(viewer) {
  viewer.addEventListener("gui_attached", ({ elements }) => {
    // Attach draggable keyframe panel
    querySelectorAll(elements, ".keyframe-panel").forEach((panel) => {
      let dragCounter = 0;
      let originalIndex;
      let dragged, originalAfterElement;
      let offsetY, offsetX, ghostEl;
      let lastState = [];

      const keyframeAddEvents = (element) => {
        element.draggable = true;
        element.addEventListener("dragstart", (event) => {
          dragged = element;
          originalAfterElement = dragged.nextElementSibling;
          originalIndex = Array.from(panel.children).indexOf(dragged);
          const { top, left } = element.getBoundingClientRect();
          offsetY = event.clientY - top;
          offsetX = event.clientX - left;
          event.dataTransfer.effectAllowed = "move";
          event.dataTransfer.setDragImage(dragged, offsetX, offsetY);
          // Fix panel height
          panel.style.height = `${panel.clientHeight}px`;
          setTimeout(() => {
            dragged.classList.add("dragging")
            dragged.remove();
          });
        });
        element.addEventListener("dragend", () => {
          if (!dragged) return;
          // Revert changes if not dropped
          element.classList.remove("dragging");
          panel.insertBefore(dragged, originalAfterElement);
          // Remove height fix
          panel.style.height = "";
        });

        // Delete keyframe
        element.querySelectorAll(".ti-trash").forEach((trash) => {
          trash.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            const key = element.getAttribute("data-key");
            viewer.delete_keyframe({ keyframe_id: key });
          });
        });

        // Delete keyframe
        element.querySelectorAll(".ti-copy-plus").forEach((trash) => {
          trash.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            const key = element.getAttribute("data-key");
            viewer.duplicate_keyframe({ 
              keyframe_id: key,
            });
          });
        });

        // Move keyframe up
        element.querySelectorAll(".ti-arrow-narrow-up").forEach((up) => {
          up.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            const key = element.getAttribute("data-key");
            const keyframes = viewer.state.camera_path_keyframes;
            const index = keyframes.findIndex((x) => x.id === key);
            if (index > 0) {
              const lastElem = keyframes.splice(index, 1);
              keyframes.splice(index - 1, 0, lastElem[0]);
              viewer.notifyChange({ property: "camera_path_keyframes" });
            }
          });
        });

        // Move keyframe down
        element.querySelectorAll(".ti-arrow-narrow-down").forEach((down) => {
          down.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            const key = element.getAttribute("data-key");
            const keyframes = viewer.state.camera_path_keyframes;
            const index = keyframes.findIndex((x) => x.id === key);
            if (index < keyframes.length - 1) {
              const lastElem = keyframes.splice(index, 1);
              keyframes.splice(index + 1, 0, lastElem[0]);
              viewer.notifyChange({ property: "camera_path_keyframes" });
            }
          });
        });

        element.addEventListener("click", () => {
          viewer.state.camera_path_selected_keyframe = element.getAttribute("data-key");
          viewer.notifyChange({ property: "camera_path_selected_keyframe" });
        });
      };

      panel.addEventListener("dragenter", (event) => {
        dragCounter++;
      });

      panel.addEventListener("drop", (event) => {
        if (!dragged) return;
        dragCounter = 0;
        const newIndex = Array.from(panel.children).indexOf(dragged);
        dragged.classList.remove("dragging");
        dragged = undefined;
        panel.style.height = "";

        // Here we need to commit the change to keyframes
        const keyframes = viewer.state.camera_path_keyframes;
        const keyframe = keyframes[originalIndex];
        keyframes.splice(originalIndex, 1);
        keyframes.splice(newIndex, 0, keyframe);
        // Commit change to last state
        const lastElem = lastState.splice(originalIndex, 1);
        lastState.splice(newIndex, 0, lastElem[0]);
        viewer.notifyChange({ property: "camera_path_keyframes" });
      });

      panel.addEventListener("dragover", (event) => {
        if (!dragged) return;
        let afterElement;
        for (const child of panel.children) {
          if (child.classList.contains("dragging")) continue;
          const { top, bottom } = child.getBoundingClientRect();
          const height = bottom - top;
          if (event.clientY - offsetY < top + height / 2) {
            if (!afterElement) { afterElement = child; }
          }
        }
        panel.insertBefore(dragged, afterElement);
        event.preventDefault();
      });

      panel.addEventListener("dragleave", (event) => {
        if (!dragged) return;
        dragCounter--;
        if (dragCounter === 0) {
          dragged.remove();
        }
      });

      const createKeyframeElement = (keyframe) => {
        const element = document.createElement("div");
        element.classList.add("keyframe");
        element.setAttribute("data-key", keyframe.id);
        element.draggable = true;
        element.innerHTML = `
        <span>
          <i class="ti ti-arrow-narrow-up"></i>
          <i class="ti ti-arrow-narrow-down"></i>
          <i class="ti ti-copy-plus"></i>
          <i class="ti ti-trash"></i>
        </span>
        <span></span>
        <span>${keyframe.duration}</span>
        <span>${keyframe.velocity_multiplier}</span>
        `;
        keyframeAddEvents(element);
        return element;
      };

      // Handle changes to keyframes
      viewer.addEventListener("change", ({ property, state }) => {
        if (property === undefined || property === "camera_path_keyframes" || property === "camera_path_trajectory") {
          const { camera_path_keyframes } = state;
          const elementMap = {};
          // First, we fix order and missing elements
          lastState.forEach((x, i) => { elementMap[x.id] = x; });
          let i;
          for (i=0; i < camera_path_keyframes.length; ++i) {
            const keyframe = camera_path_keyframes[i];
            // 1) Element exists and is at the correct position
            if (lastState[i]?.id === keyframe.id) continue;
            // 2) Element exists, but is at the wrong position
            if (elementMap[keyframe.id]) {
              const node = elementMap[keyframe.id];
              const originalIndex = lastState.indexOf(node);
              lastState.splice(originalIndex, 1);
              lastState.splice(i, 0, node);
              node.element.remove();
              panel.insertBefore(node.element, panel.children[i]);
              continue
            }
            // 3) Element does not exist
            const element = createKeyframeElement(keyframe);
            lastState.splice(i, 0, {
              id: keyframe.id,
              element,
            });
            panel.insertBefore(element, panel.children[i]);
          }
          // 2) Remove extra elements
          for (;i < lastState.length; i++) {
            lastState[i].element.remove();
          }
          lastState = lastState.slice(0, camera_path_keyframes.length);

          // 3) Update last state index value
          for (i=0; i < lastState.length; i++) {
            if (lastState[i].index !== i) {
              lastState[i].index = i;
              lastState[i].element.children[1].innerText = i + 1;
            }
          }

          // 4) Update keyframe timings
          if (state.camera_path_trajectory) {
            const { keyframeStarts, keyframeDurations } = state.camera_path_trajectory;
            for (let i=0; i < lastState.length; i++) {
              const start = keyframeStarts[i]?.toFixed(2) || "-"
              const duration = keyframeDurations[i]?.toFixed(2) || "-";
              if (lastState[i].start !== start) {
                lastState[i].start = start;
                lastState[i].element.children[2].innerText = start;
              }
              if (lastState[i].duration !== duration) {
                lastState[i].duration = duration;
                lastState[i].element.children[3].innerText = duration;
              }
            }
          }
        }
      });
    });
  });
}


function parseBinding(expr, type=undefined) {
  if (expr.indexOf("||") > 0) {
    const names = [];
    const callbacks = [];
    const parts = expr.split("||").map(x => x.trim()).filter(x => x.length > 0).map(x => parseBinding(x, type));
    parts.forEach(([callback, partNames]) => {
      names.push(...partNames);
      callbacks.push(callback);
    });
    return [(state) => callbacks.any(callback => callback(state)), names];
  }
  if (expr.indexOf("&&") > 0) {
    const names = [];
    const callbacks = [];
    const parts = expr.split("&&").map(x => x.trim()).filter(x => x.length > 0).map(x => parseBinding(x, type));
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
  const getter = (state) => {
    let value = state[expr];
    if (type === "bool" && Array.isArray(value)) value = value.length > 0;
    return value;
  };
  return [getter, [expr]];
}


const isNarrowScreen = () => window.innerWidth < 1000;
const defaultState = {
  menu_visible: !isNarrowScreen(),
  output_palettes: Object.keys(palettes),
  outputs_configuration: {
    depth: { palette_enabled: true },
    accumulation: { palette_enabled: true },
  },
};


export class Viewer extends THREE.EventDispatcher {
  constructor({ 
    viewport, 
    viewer_transform,
    viewer_initial_pose,
    url,
    state,
    plugins,
  }) {
    super();
    state = Object.assign({}, defaultState, state || {});
    for (const key in defaultState.outputs_configuration) {
      state.outputs_configuration[key] = Object.assign({}, 
        defaultState.outputs_configuration[key], 
        state.outputs_configuration[key]);
    }
    this._backgroundTexture = undefined;
    this.viewport = viewport || document.querySelector(".viewport");
    const width = this.viewport.clientWidth;
    const height = this.viewport.clientHeight;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(width, height);
    this.viewport.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera( 70, width / height, 0.01, 100 );
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
      this.scene.updateMatrixWorld();
    }
    if (viewer_initial_pose) {
      if (Array.isArray(viewer_initial_pose))
        viewer_initial_pose = makeMatrix4(viewer_initial_pose);
      viewer_initial_pose = viewer_initial_pose.clone();
      viewer_initial_pose.premultiply(this.scene.matrixWorld.clone().invert());
      // Normalize matrix
      const scale = new THREE.Vector3();
      const quaternion = new THREE.Quaternion();
      const position = new THREE.Vector3();
      viewer_initial_pose.decompose(position, quaternion, scale);
      viewer_initial_pose.compose(position, quaternion, new THREE.Vector3(1, 1, 1));
      this._viewer_initial_pose = viewer_initial_pose;
      this.reset_camera();
    } else {
      this._viewer_initial_pose = new THREE.Matrix4().copy(this.camera.matrixWorld);
    }

    this._computed_property_names = new Set();
    this._computed_property_names.add("camera_path_trajectory");


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
    _attach_camera_control(this);
    _attach_camera_path(this);
    this._attach_preview_is_playing();
    this._attach_update_preview_mode();
    _attach_camera_path_curve(this);
    _attach_camera_path_keyframes(this);
    _attach_camera_path_selected_keyframe_pivot_controls(this);
    _attach_player_frustum(this);
    _attach_viewport_split_slider(this, this.viewport);
    _attach_selected_keyframe_details(this);
    _attach_draggable_keyframe_panel(this);
    _attach_persistent_state(this);
    _attach_init_show_in_new_window(this);
    _attach_output_configuration(this);

    this.addEventListener("change", ({ property, state }) => {
      if (property === undefined || property === 'render_fov') {
        this.camera.fov = state.render_fov;
        this.camera.updateProjectionMatrix();
        this._draw_background();
      }
      if (property === undefined || property === 'menu_visible') {
        setTimeout(() => this._resize(), 0);
      }
    });

    this.dispatchEvent({ type: "start", state: this.state });
    (plugins || []).map((plugin) => this.load_plugin(plugin));
  }

  force_render() {
    this._force_render = true;
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
          this.update_notification({
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

  _get_render_params({ force=false } = {}) {
    const state = this.state;
    if (this._force_render) {
      this._force_render = false;
      this._last_render_params = undefined;
      this._was_full_render = false;
    }

    const _get_params = ({ resolution, width, height, fov, matrix, ...rest }) => {
      [width, height] = computeResolution([width, height], resolution);
      const round = (x) => Math.round(x * 100000) / 100000;
      const focal = height / (2 * Math.tan(THREE.MathUtils.degToRad(fov) / 2));
      const request = {
        pose: matrix4ToArray(matrix).map(round),
        intrinsics: [focal, focal, width/2, height/2].map(round),
        image_size: [width, height],
        palette: state.output_palette,
        output_range: [
          notempty(state.output_range_min) ? 1*state.output_range_min : null,
          notempty(state.output_range_max) ? 1*state.output_range_max : null,
        ],
        ...rest
      };
      request.output_type = state.output_type === "" ? undefined : state.output_type;
      if (state.split_enabled && state.split_output_type) {
        request.split_output_type = state.split_output_type === "" ? undefined : state.split_output_type;
        request.split_percentage = round(state.split_percentage === undefined ? 0.5 : state.split_percentage);
        request.split_tilt = round(state.split_tilt || 0.0);
        request.split_palette = state.split_palette;
        request.split_range = [
          notempty(state.split_range_min) ? 1*state.split_range_min : null,
          notempty(state.split_range_max) ? 1*state.split_range_max : null,
        ];
      }
      return request;
    }

    let cameraParams = this.get_camera_params();
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
    if (paramsJSON === this._last_render_params && !force) return;
    this._last_render_params = paramsJSON;
    return params;
  }

  _set_frame_renderer(frame_renderer) {
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
            if (image) {
              this.dispatchEvent({ type: "frame", image });
              this._last_frames[0] = image;
              this._draw_background();
            }
          }

          // Wait
          const wait = Math.max(0, 1000 / 30 - (Date.now() - lastUpdate));
          lastUpdate = Date.now();
          if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
        } catch (error) {
          console.error("Error updating single frame:", error.message);
          this.update_notification({
            header: "Error rendering frame",
            detail: error.message,
            type: "error",
            id: "renderer",
            closeable: true,
          });
          this._last_frames[0] = undefined;
          this._draw_background();
        }
      }
    };

    run();
  }

  _on_renderer_ready({ output_types, supported_appearance_train_indices }) {
    const old_output_type = this.state.output_type;
    const old_split_output_type = this.state.split_output_type;
    this.state.output_types = output_types;
    this.state.supported_appearance_train_indices = 
      supported_appearance_train_indices === "all" ? "all" : supported_appearance_train_indices?.map(x => x.toString());
    this.notifyChange({ property: 'output_types' });
    this.notifyChange({ property: 'supported_appearance_train_indices' });
    if (!output_types.includes(old_output_type)) {
      this.state.output_type = output_types[0];
      this.notifyChange({ property: 'output_type' });
    }
    if (!output_types.includes(old_split_output_type)) {
      this.state.split_output_type = output_types[0];
      this.notifyChange({ property: 'split_output_type' });
    }
    if (this.state.supported_appearance_train_indices !== "all" && 
        this.state.render_appearance_train_index !== '' && 
        !supported_appearance_train_indices?.includes(this.state.render_appearance_train_index)) {
      this.state.render_appearance_train_index = '';
      this.notifyChange({ property: 'render_appearance_train_index' });
    }
    this.force_render()
  }

  set_mesh_renderer(params) {
    params.mesh_url = new URL(params.mesh_url, window.location.href).href;
    if (params.mesh_url_per_appearance) {
      for (const key in params.mesh_url_per_appearance) {
        params.mesh_url_per_appearance[key] = new URL(params.mesh_url_per_appearance[key], window.location.href).href;
      }
    }
    try {
      this._set_frame_renderer(new MeshFrameRenderer({
        ...params,
        update_notification: (notification) => this.update_notification(notification),
        onready: (e) => this._on_renderer_ready(e),
      }));
    } catch (error) {
      this.update_notification({
        header: "Error starting mesh renderer",
        detail: error.message,
        type: "error",
        closeable: true,
      });
    }
  }

  set_3dgs_renderer(params) {
    params.scene_url = new URL(params.scene_url, window.location.href).href;
    if (params.scene_url_per_appearance) {
      for (const key in params.scene_url_per_appearance) {
        params.scene_url_per_appearance[key] = new URL(params.scene_url_per_appearance[key], window.location.href).href;
      }
    }
    try {
      this._set_frame_renderer(new GaussianSplattingFrameRenderer({
        ...params,
        update_notification: (notification) => this.update_notification(notification),
        onready: (e) => this._on_renderer_ready(e),
      }));
    } catch (error) {
      this.update_notification({
        header: "Error starting Gaussian Splatting renderer",
        detail: error.message,
        type: "error",
        closeable: true,
      });
    }
  }

  set_remote_renderer({
    websocket_url,
    http_url,
    output_types,
  }) {
    try {
      if (websocket_url)
        websocket_url = new URL(websocket_url, window.location.href).href;
      if (http_url)
        http_url = new URL(http_url, window.location.href).href;
      // Google Colab does not support websocket connections
      const supportsWebsocket = !window.location.host.endsWith(".googleusercontent.com");
      if (!supportsWebsocket)
        websocket_url = null;

      if (websocket_url) {
        if (websocket_url.startsWith("http://")) websocket_url = "ws://" + websocket_url.slice(7);
        if (websocket_url.startsWith("https://")) websocket_url = "wss://" + websocket_url.slice(8);

        this._set_frame_renderer(new WebSocketFrameRenderer({ 
          url: websocket_url,
          update_notification: (notification) => {
            viewer.update_notification(notification);
          }
        }));
        this.state.frame_renderer_url = websocket_url;
        this.notifyChange({ property: 'frame_renderer_url' });
      } else {
        this._set_frame_renderer(new HTTPFrameRenderer({ 
          url: http_url,
          update_notification: (notification) => {
            viewer.update_notification(notification);
          }
        }));
        this.state.frame_renderer_url = http_url;
        this.notifyChange({ property: 'frame_renderer_url' });
      }
      const supported_appearance_train_indices = "all";
      this._on_renderer_ready({ output_types, supported_appearance_train_indices });
    } catch (error) {
      this.update_notification({
        header: "Error starting remote renderer",
        detail: error.message,
        type: "error",
        closeable: true,
      });
    }
  }

  set_renderer(params) {
    const { type, ...rendererState } = params;
    this.dispatchAction(`set_${type}_renderer`, rendererState);
  }

  set_dataset(options) {
    if (this.dataset_manager) {
      this.dataset_manager.dispose();
      this.dataset_manager = undefined;
    }
    // Add DatasetManager
    this.dataset_manager = new DatasetManager({ 
      ...options,
      viewer: this
    });
  }

  _resize() {
    const width = Math.max(1, this.viewport.clientWidth);
    const height = Math.max(1, this.viewport.clientHeight);

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

  get_camera_params() {
    if (this.state.preview_camera !== undefined) {
      return this.state.preview_camera;
    }

    // Cache variables for performance
    const v1 = this._v1 = this._v1 || new THREE.Vector3();
    const q1 = this._q1 = this._q1 || new THREE.Quaternion();
    const s1 = this._s1 = this._s1 || new THREE.Vector3();

    const pose = this.camera.matrixWorld.clone();
    pose.multiply(_R_threecam_cam);
    pose.premultiply(this.scene.matrixWorld.clone().invert());

    // Normalize pose
    pose.decompose(v1, q1, s1);
    s1.set(1, 1, 1);
    pose.compose(v1, q1, s1);

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

  dispatchAction(action, ...args) {
    if (this[action]) {
      return this[action](...args);
    }
    if (this._actionHandlers && this._actionHandlers[action]) {
      return this._actionHandlers[action](...args);
    }
    throw new Error(`Action handler not found: ${action}`);
  }

  setActionHandler(action, handler) {
    this._actionHandlers = this._actionHandlers || {};
    this._actionHandlers[action] = handler;
  }

  addComputedProperty({ name, getter, dependencies }) {
    this._computed_property_names.add(name);
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

  create_public_url_accept() {
    return this.create_public_url({ accept_license_terms: true });
  }

  async create_public_url({ accept_license_terms=false } = {}) {
    this.update_notification({
      id: "public-url",
      header: "Creating public URL",
      closeable: false,
    });
    let publicUrl = undefined;
    this.state.viewer_requesting_public_url = true;
    this.notifyChange({ property: "viewer_requesting_public_url" });
    try {
      const response = await fetch(`./create-public-url?accept_license_terms=${accept_license_terms?'yes':'no'}`, { method: "POST" });
      let result;
      try {
        result = await response.json();
      } catch (error) {
        throw new Error(`Failed to create public URL: ${response.statusText}`);
      }
      if (result.status === "error") {
        const error = new Error(result.message);
        error.license_terms_url = result.license_terms_url;
        throw error;
      }
      publicUrl = result.public_url;

      console.log("Public URL:", publicUrl);
      this.state.viewer_public_url = publicUrl;
      this.notifyChange({ property: "viewer_public_url" });
      this.update_notification({
        id: "public-url",
        header: "Public URL created",
        detailHTML: `<a href="${publicUrl}" target="_blank" style="word-wrap:break-word;word-break:break-all">${publicUrl}</a>`,
        closeable: true,
      });
    } catch (error) {
      if (error.license_terms_url === "https://www.cloudflare.com/website-terms/") {
        // User needs to accept Cloudflare's terms
        this.update_notification({
          id: "public-url",
          autoclose: 0,
        });
        this.dispatchAction("open_dialog_confirm_cloudflare_create_public_url");
        return;
      }
      console.error("Failed to create public URL:", error.message);
      this.update_notification({
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

  delete_keyframe({ keyframe_id } = {}) {
    if (keyframe_id === undefined) {
      keyframe_id = this.state.camera_path_selected_keyframe
      if (keyframe_id === undefined) return
    }
    this.state.camera_path_keyframes = this.state.camera_path_keyframes.filter((keyframe) => keyframe.id !== keyframe_id);
    if (this.state.camera_path_selected_keyframe === keyframe_id) {
      this.state.camera_path_selected_keyframe = undefined;
      this.notifyChange({ property: "camera_path_selected_keyframe" });
    }
    const n = this.state.camera_path_keyframes.length;
    // We decrease the total duration, but we allow at least two digits of precision
    this.state.camera_path_duration = n < 1 ? 0 : 
      this.state.camera_path_duration = modifyNumberKeepPrecision(
        [this.state.camera_path_duration, 0.01],
        (x) => x*n / (n + 1));
    this.notifyChange({ property: "camera_path_duration" });
    this.notifyChange({ property: "camera_path_keyframes" });
  }

  duplicate_keyframe({ keyframe_id } = {}) {
    if (keyframe_id === undefined) {
      keyframe_id = this.state.camera_path_selected_keyframe
      if (keyframe_id === undefined) return
    }

    const keyframeIndex = this.state.camera_path_keyframes.findIndex((keyframe) => keyframe.id === keyframe_id);
    const keyframe = this.state.camera_path_keyframes[keyframeIndex];
    const copiedKeyframe = {
      ...keyframe,
      id: this._assign_keyframe_id(),
      quaternion: keyframe.quaternion.clone(),
      position: keyframe.position.clone(),
    };
    this.state.camera_path_keyframes.splice(keyframeIndex + 1, 0, copiedKeyframe);
    const duration = this.state.camera_path_duration || 0;
    const n = this.state.camera_path_keyframes.length;
    if (n !== 2 || this.state.camera_path_loop) {
      this.state.camera_path_duration = modifyNumberKeepPrecision(
        [this.state.camera_path_duration, this.state.camera_path_default_transition_duration],
        (a, b) => a + b);
        this.state.camera_path_default_transition_duration;
    }
    this.notifyChange({ property: "camera_path_duration" });
    this.notifyChange({ property: "camera_path_keyframes" });
  }



  add_keyframe() {
    const { matrix } = this.get_camera_params();
    const quaternion = new THREE.Quaternion();
    const position = new THREE.Vector3();
    const scale = new THREE.Vector3();
    matrix.decompose(position, quaternion, scale);
    this.state.camera_path_keyframes.push({
      id: this._assign_keyframe_id(),
      quaternion,
      position,
      fov: undefined,
    });
    const duration = this.state.camera_path_duration || 0;
    const n = this.state.camera_path_keyframes.length;
    if (n !== 2 || this.state.camera_path_loop) {
      this.state.camera_path_duration = modifyNumberKeepPrecision(
        [this.state.camera_path_duration, this.state.camera_path_default_transition_duration],
        (a, b) => a + b);
        this.state.camera_path_default_transition_duration;
    }
    this.notifyChange({ property: "camera_path_duration" });
    this.notifyChange({ property: "camera_path_keyframes" });
  }

  clear_selected_keyframe() {
    this.state.camera_path_selected_keyframe = undefined;
    this.notifyChange({ property: "camera_path_selected_keyframe" });
  }

  _attach_computed_properties() {
    this.addComputedProperty({
      name: "has_renderer",
      dependencies: ["output_types"],
      getter: ({ output_types }) => output_types && output_types.length > 0
    });
    this.addComputedProperty({
      name: "has_output_split",
      dependencies: ["output_types"],
      getter: ({ output_types }) => output_types && output_types.length > 1
    });

    this.addComputedProperty({
      name: "render_appearance_train_index_options",
      dependencies: [
        "dataset_train_appearance_options",
        "supported_appearance_train_indices",
      ],
      getter: ({ 
        dataset_train_appearance_options, 
        supported_appearance_train_indices 
      }) => [
        { value: "", label: "none" }, 
          ...(dataset_train_appearance_options?.filter(x => {
            if (x.value === "") return false;
            if (supported_appearance_train_indices === "all") return true;
            if (!supported_appearance_train_indices) return false;
            return supported_appearance_train_indices.includes(x.value);
        }) || [])],
    });
    this.addComputedProperty({
      name: "render_appearance_train_index_enabled",
      dependencies: ["render_appearance_train_index_options"],
      getter: ({ render_appearance_train_index_options }) =>
        render_appearance_train_index_options.length > 1,
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
      name: "camera_path_show_computed_duration",
      dependencies: ["camera_path_time_interpolation", "camera_path_interpolation"],
      getter: ({
        camera_path_time_interpolation,
        camera_path_interpolation }) => {
        if (camera_path_interpolation === "none") return true;
        if (camera_path_time_interpolation === "time") return true;
        return false;
      },
    });

    this.addComputedProperty({
      name: "camera_path_computed_duration",
      dependencies: [
        "camera_path_keyframes", 
        "camera_path_default_transition_duration", 
        "camera_path_duration", 
        "camera_path_time_interpolation", 
        "camera_path_loop",
        "camera_path_interpolation"],
      getter: ({ 
        camera_path_default_transition_duration,
        camera_path_keyframes, 
        camera_path_duration, 
        camera_path_loop,
        camera_path_time_interpolation,
        camera_path_interpolation }) => {
        if (camera_path_interpolation === "none" || camera_path_time_interpolation === "time") {
          return camera_path_keyframes.map((x, i) => {
            const duration = (x.duration === undefined || x.duration === null) ? 
              camera_path_default_transition_duration : x.duration;
            if (camera_path_time_interpolation === "time" && !camera_path_loop && i === camera_path_keyframes.length - 1) {
              return 0;
            }
            return duration;
          }).reduce((a, b) => a + b, 0);
        }
        return camera_path_duration;
      }
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
          property !== 'camera_path_default_transition_duration' &&
          property !== 'camera_path_duration' &&
          property !== 'preview_is_playing') return;
      const {
        camera_path_trajectory,
        camera_path_framerate,
        camera_path_interpolation,
        camera_path_duration,
        camera_path_default_transition_duration,
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
          delays = camera_path_keyframes.map(x => 1000 * ((x.duration === undefined || x.duration === null)
            ? camera_path_default_transition_duration : x.duration));
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

  _getFrameRepeats(durations, fps) {
    // Add frame repeats to match target FPS
    const frameRepeats = [];
    let time = 0;
    let nFrames = 0;
    durations.forEach((duration, i) => {
      const numFrames = Math.max(0, Math.round((time + duration) * fps) - nFrames);
      frameRepeats.push(numFrames);
      nFrames += numFrames;
      time = nFrames / fps;
    });
    return frameRepeats
  }

  export_trajectory() {
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
      default_transition_duration: state.camera_path_default_transition_duration,
      time_interpolation: state.camera_path_time_interpolation,
    };
    let fps = state.camera_path_framerate;
    if (source.interpolation === "kochanek-bartels") {
      source.is_cycle = state.camera_path_loop;
      source.tension = state.camera_path_tension;
      source.continuity = state.camera_path_continuity;
      source.bias = state.camera_path_bias;
      source.distance_alpha = state.camera_path_distance_alpha;
    } else if (source.interpolation === "linear" || source.interpolation === "circle") {
      source.is_cycle = state.camera_path_loop;
      source.distance_alpha = state.camera_path_distance_alpha;
    }
    const data = {
      format: 'nerfbaselines-v1',
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
      data.frame_repeats = this._getFrameRepeats(
        keyframes.map(x => x.duration || state.camera_path_default_transition_duration),
        fps);
    }
    return data
  }

  async save_trajectory() {
    try {
      const data = this.export_trajectory();
      await saveAs(new Blob([JSON.stringify(data, null, '  ')]), { 
        type: "application/json",
        filename: "trajectory.json",
        extension: "json",
        description: "Camera trajectory JSON",
      });
    } catch (error) {
      if (error.name === "AbortError") return;
      console.error("Error saving camera path:", error);
      this.update_notification({
        header: "Error saving camera path",
        detail: error.message,
        type: "error",
      });
    }
  }

  async render_video() {
    const state = this.state;
    const width = this.state.camera_path_resolution_1;
    const height = this.state.camera_path_resolution_2;
    const fps = this.state.camera_path_framerate;
    const { 
      positions, 
      quaternions, 
      fovs, 
      weights, 
      appearanceTrainIndices,
      keyframeDurations,
    } = this.state.camera_path_trajectory;
    let repeats;
    let numFrames = positions.length;
    if (state.camera_path_interpolation === "none") {
      repeats = this._getFrameRepeats(
        keyframeDurations, fps);
      numFrames = repeats.reduce((a, b) => a + b, 0);
    }
    const writer = new VideoWriter({ 
      width, height, fps, 
      type: this.state.camera_path_render_format,
      mp4Codec: this.state.camera_path_render_mp4_codec,
      webmCodec: this.state.camera_path_render_webm_codec,
      keyframeInterval: this.state.camera_path_render_keyframe_interval,
      numFrames });
    try {
      await writer.saveAs();
    } catch (error) {
      if (error.name === "AbortError") { return; }
      console.error("Error saving video:", error);
      this.update_notification({
        header: "Error saving video",
        detail: error.message,
        type: "error",
      });
    }

    const renderId = this._renderVideoId = (this._renderVideoId || 0) + 1;
    let closed = false;
    this.update_notification({
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
        const request = {
          pose: matrix4ToArray(matrix).map(round),
          intrinsics: [focal, focal, width/2, height/2].map(round),
          image_size: [width, height],
          appearance_weights,
          appearance_train_indices,
          lossless: true,
          palette: state.output_palette,
          output_range: [
            notempty(state.output_range_min) ? 1*state.output_range_min : null,
            notempty(state.output_range_max) ? 1*state.output_range_max : null,
          ],
        };
        request.output_type = state.output_type === "" ? undefined : state.output_type;
        if (state.split_enabled && state.split_output_type) {
          request.split_output_type = state.split_output_type === "" ? undefined : state.split_output_type;
          request.split_percentage = round(state.split_percentage === undefined ? 0.5 : state.split_percentage);
          request.split_tilt = round(state.split_tilt || 0.0);
          request.split_palette = state.split_palette;
          request.split_range = [
            notempty(state.split_range_min) ? 1*state.split_range_min : null,
            notempty(state.split_range_max) ? 1*state.split_range_max : null,
          ];
        }
        const frame = await this.frame_renderer.render(request);
        if (closed) break;
        try {
          await writer.addFrame(frame, { repeats: repeats?.[i] });
        } finally {
          frame?.close?.();
        }
        this.update_notification({
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
        this.update_notification({
          id: renderId,
          header: "Rendering finished",
          autoclose: notification_autoclose,
          closeable: true,
        });
      }
    } catch (error) {
      console.error("Error rendering video:", error);
      this.update_notification({
        id: renderId,
        header: "Rendering failed",
        detail: error.message,
        type: "error",
        closeable: true,
      });
    }
  }

  reset_camera() {
    this.set_camera({ matrix: this._viewer_initial_pose });
    const target = new THREE.Vector3(0, 0, 0);
    this.camera.up = new THREE.Vector3(0, 0, 1);
    this.controls.updateUp();
    this.controls.target?.copy(target);
    this.controls.update();
  }

  set_up_direction() {
    let up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.camera.quaternion);
    this.camera.up.copy(up);
    this.controls.updateUp();
  }

  async load_plugin({ code, id }) {
    try {
      const plugin = await evaluateES6(code);
      await plugin(this);
    } catch (error) {
      console.error("Error loading plugin:", error);
      this.update_notification({
        header: `Error loading plugin ${id}`,
        detail: error.message,
        type: "error",
      });
    }
  }

  _assign_keyframe_id() {
    let i = this._keyframeCounter || 0;
    this.state.camera_path_keyframes?.forEach((keyframe) => {
      i = Math.max(i, parseInt(keyframe.id) || 0);
    });
    i = i + 1;
    this._keyframeCounter = i;
    return i.toString();
  }

  load_trajectory({ data }) {
    try {
      if (!data) {
        throw new Error("No data provided");
      }
      if (data.format !== "nerfbaselines-v1") {
        throw new Error(`Unsupported format ${data.format}. Only 'nerfbaselines-v1' is supported`);
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
        default_transition_duration,
        time_interpolation,
        distance_alpha,
      } = source;
      if (correctnull(default_fov) !== undefined)
        state.camera_path_default_fov = default_fov;
      if (correctnull(default_transition_duration) !== undefined)
        state.camera_path_default_transition_duration = default_transition_duration;
      if (correctnull(distance_alpha) !== undefined)
        state.camera_path_distance_alpha = distance_alpha;
      if (correctnull(time_interpolation) !== undefined) {
        if (["velocity", "time"].indexOf(time_interpolation) === -1) {
          throw new Error("Time interpolation must be either 'velocity' or 'time'");
        }
        state.camera_path_time_interpolation = time_interpolation;
      } else {
        // This is a legacy trajectory format where velocity intepolation was not implemented
        state.camera_path_time_interpolation = "time";
      }
      state.camera_path_duration = correctnull(duration) === undefined ? 
        (state.camera_path_default_transition_duration*source["keyframes"].length) : duration;
      if (state.camera_path_duration === undefined || state.camera_path_duration === null)
        state.camera_path_duration = 0;

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
          id: this._assign_keyframe_id(),
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
      this.update_notification({
        header: "Error loading camera path",
        detail: error.message,
        type: "error",
      });
    }
  }

  hide_menu() {
    this.state.menu_visible = false;
    this.notifyChange({ property: "menu_visible" });
  }

  show_menu() {
    this.state.menu_visible = true;
    this.notifyChange({ property: "menu_visible" });
  }

  attach_gui({ elements } = {}) {
    elements = elements || [document.body];
    const state = this.state;

    this.dispatchEvent({
      type: "attach_gui_started",
      elements,
      target: this,
      state,
    });

    // Handle state change
    function getValue(element) {
      let { name, value, type, checked } = element;
      if (type === "checkbox") value = checked;
      else if (type === "number" || type === "range") value = element.valueAsNumber;
      const dataType = element.getAttribute("data-type");
      if (dataType === "bool") value = value === "true" || value === true || value === "1" || value === 1;
      return value;
    }
    function setValue(element, value) {
      const { type } = element;
      if (type === "checkbox") {
        element.checked = value;
      } else if (type === "radio") {
        const checked = (
          element.value === value || 
          (element.value === "true" && value === true) ||
          (element.value === "false" && value === false));
        element.checked = checked;
      } else {
        element.value = value;
        const e = new Event("input");
        e.simulated = true;
        element.dispatchEvent(e);
      }
    }
    const query = (selector) => querySelectorAll(elements, selector);
    query("[data-set-viewer-ref]").forEach(element => { element.viewer = this });
    query("input[name],select[name]").forEach(element => {
      element.addEventListener("change", (event) => {
        if (event.simulated) return;
        state[name] = getValue(event.target);
        this.notifyChange({ 
          property: name, 
          origin: element,
          trigger: "gui_change",
        });
      });
      element.addEventListener("input", (event) => {
        if (event.simulated) return;
        state[name] = getValue(event.target);
        this.notifyChange({ 
          property: name, 
          origin: element,
          trigger: "gui_input",
        });
        if (name === "preview_frame" && type === "range") {
          // Changing preview_frame stops the preview
          state.preview_is_playing = false;
          this.notifyChange({ 
            property: "preview_is_playing", 
            origin: element,
            trigger: "gui_input",
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
    query("[data-bind]").forEach(element => {
      const name = element.getAttribute("data-bind");
      this.addEventListener("change", ({ property }) => {
        if (property !== name && property !== undefined) return;
        element.innerText = state[name];
      });
    });

    query("[data-options]").forEach(element => {
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

    query("[data-enable-if]").forEach(element => {
      const [evalFn, dependencies] = parseBinding(element.getAttribute("data-enable-if"), "bool");
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

    query("[data-visible-if]").forEach(element => {
      const [evalFn, dependencies] = parseBinding(element.getAttribute("data-visible-if"), "bool");
      let display = element.style.display;
      if (display === "none") display = null;
      this.addEventListener("change", ({ property, state }) => {
        if (property !== undefined && !dependencies.includes(property)) return;
        element.style.display = evalFn(state) ? display : "none";
      });
    });

    query("[data-action]").forEach(element => {
      const action = element.getAttribute("data-action");
      element.addEventListener("click", (e) => {
        if (e.simulated) return;
        if (!element.getAttribute("disabled"))
          this.dispatchAction(action);
      });
    });

    // ata-bind-class has the form "class1:property1"
    query("[data-bind-class]").forEach(element => {
      const attr = element.getAttribute("data-bind-class");
      attr.split(";").forEach((attr) => {
        const [class_name, expr] = attr.split(":");
        const [evalFn, dependencies] = parseBinding(expr, "string");
        this.addEventListener("change", ({ property, state }) => {
          if (property !== undefined && !dependencies.includes(property)) return;
          if (evalFn(state)) {
            element.classList.add(class_name);
          } else {
            element.classList.remove(class_name);
          }
        });
      });
    });

    // data-bind-attr has the form "attribute:property"
    query("[data-bind-attr]").forEach(element => {
      const [attr, name] = element.getAttribute("data-bind-attr").split(":");
      this.addEventListener("change", ({ property, state }) => {
        if (property !== name && property !== undefined) return;
        element.setAttribute(attr, state[name]);
      });
      if (state[name] !== undefined)
        element.setAttribute(attr, state[name]);
    });

    query('#input_trajectory').forEach((input) => {
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
          this.load_trajectory({ data });
        };
        reader.readAsText(file);
      });
    });

    // Camera pose textarea
    const camera_pose_elements = query("[name=camera_pose]");
    const updateCameraElements = () => {
      const { matrix } = this.get_camera_params();
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
    query("#method_info").forEach((method_info_element) => {
      const update_method_info = ({ method_info }) => {
        const newHtml = method_info ? buildMethodInfo(method_info) : "";
        method_info_element.innerHTML = newHtml;
      }
      this.addEventListener("change", ({ property, state }) => {
        if (property === "method_info" || property === undefined)
          update_method_info(state);
      });
    });
    
    // Dataset info
    query("#dataset_info").forEach((dataset_info_element) => {
      const update_dataset_info = ({ dataset_info }) => {
        const newHtml = dataset_info ? buildDatasetInfo(dataset_info) : "";
        dataset_info_element.innerHTML = newHtml;
      }
      this.addEventListener("change", ({ property, state }) => {
        if (property === "dataset_info" || property === undefined)
          update_dataset_info(state);
      });
    });

    // Method hparams
    query("#method_hparams").forEach((method_hparams_element) => {
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
    });

    query(".dialog").forEach(element => {
      const id = element.id;
      this.setActionHandler(`open_dialog_${id}`, () => {
        element.classList.add("dialog-open");
      });
    });

    // Only attach once when attach_gui is called initially
    query("body").forEach(() => {
      this.addEventListener("change", ({ property, state }) => {
        if (property === "theme_color" || property === undefined) {
          document.documentElement.style.setProperty("--theme-color", state.theme_color);
        }
        if (property === "viewer_font_size" || property === undefined) {
          document.documentElement.style.fontSize = `${state.viewer_font_size}rem`;
        }
      });
    });

    this.dispatchEvent({
      type: "gui_attached",
      elements,
      target: this,
      state,
    });

    // Notify gui is attached and propagate changes
    this.notifyChange({ property: undefined, trigger: "gui_attached" });
  }

  update_notification({ id, header, progress, autoclose=undefined, detail="", detailHTML=undefined, type="info", onclose, closeable=true }) {
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
    if (!this._last_frames[0]) {
      this.renderer_scene.background = null;
      return
    }
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

function formatTime(seconds) {
  return Math.round(seconds / 3600) + "h " + (seconds % 3600 / 60).toFixed(0) + "m";
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
  if (info.licenses !== undefined) {
    const licenses = info.licenses.map(x => x.url ? `<a href="${x.url}" target="_blank">${x.name}</a>` : x.name);
    out += `<strong>Licenses:</strong><span>${licenses.join(", ")}</span>\n`;
  }
  if (info.supported_outputs !== undefined)
    out += `<strong>Outputs:</strong><span>${info.supported_outputs.join(", ")}</span>\n`;
  if (info.nb_version !== undefined)
    out += `<strong>NB version:</strong><span>${info.nb_version}</span>\n`;
  if (info.applied_presets !== undefined)
    out += `<strong>Presets:</strong><span>${info.applied_presets.join(', ')}</span>\n`;
  if (info.config_overrides !== undefined) {
    let config_overrides = "";
    for (const k in info.config_overrides) {
      const v = info.config_overrides[k];
      config_overrides += `${k} = ${v}<br/>\n`
    }
    out += `<strong>Config overrides:</strong><span>${config_overrides}</span>\n`;
  }
  if (info.datetime !== undefined)
    out += `<strong>Datetime:</strong><span>${info.datetime}</span>\n`;
  if (info.total_train_time !== undefined)
    out += `<strong>Train time:</strong><span>${formatTime(info.total_train_time)}</span>\n`;
  if (info.num_iterations !== undefined)
    out += `<strong>Iterations:</strong><span>${info.num_iterations}</span>\n`;
  if (info.resources_utilization !== undefined && info.resources_utilization.gpu_memory > 0) {
    const { gpu_memory, gpu_name } = info.resources_utilization;
    out += `<strong>GPU mem:</strong><span>${(gpu_memory/1024).toFixed(2)} GB</span>\n`;
    out += `<strong>GPU type:</strong><span>${gpu_name}</span>\n`;
  }
  if (info.nb_version !== undefined)
    out += `<strong>NB version:</strong><span>${info.nb_version}</span>\n`;
  if (info.checkpoint_sha !== undefined)
    out += `<strong>Checkpoint SHA:</strong><span>${info.checkpoint_sha}</span>\n`;
  return out;
}

function buildDatasetInfo(info) {
  let out = "";
  if (info.id !== undefined)
    out += `<strong>Dataset ID:</strong><span>${info.id}</span>\n`;
  if (info.name !== undefined)
    out += `<strong>Name:</strong><span>${info.name}</span>\n`;
  if (info.scene !== undefined)
    out += `<strong>Scene:</strong><span>${info.scene}</span>\n`;
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
  if (info.color_space !== undefined)
    out += `<strong>Color space:</strong><span>${info.color_space}</span>\n`;
  if (info.downscale_factor !== undefined)
    out += `<strong>Downscale factor:</strong><span>${info.downscale_factor}</span>\n`;
  if (info.type !== undefined)
    out += `<strong>Type:</strong><span>${info.type}</span>\n`;
  if (info.metrics !== undefined) {
    out += `<strong>Metrics:</strong><span>${info.metrics.map(x=>x.name).join(", ")}</span>\n`;
  }
  return out;
}


function modifyNumberKeepPrecision(source, callback) {
  const countDecimals = (value) => {
    if (Math.floor(value) !== value)
      return value.toString().split(".")[1].length || 0;
    return 0;
  }
  const precision = source.map(countDecimals).reduce((a, b) => Math.max(a, b), 0);
  const result = callback(...source);
  return parseFloat(result.toFixed(precision));
}


function makeMatrix4(elements) {
  if (!elements || elements.length !== 12) {
    throw new Error("Invalid elements array. Expected 12 elements.");
  }
  return new THREE.Matrix4().set(...elements, 0, 0, 0, 1);
}

function matrix4ToArray(matrix) {
  const e = matrix.elements;
  return [e[0], e[4], e[8], e[12], e[1], e[5], e[9], e[13], e[2], e[6], e[10], e[14]];
}

function formatMatrix4(matrix, round) {
  let a = matrix4ToArray(matrix);
  if (round !== undefined) {
    // For each element, round to the nearest `round` decimal places
    a = a.map(x => x.toFixed(round)).map(x => x.startsWith("-") ? x : ` ${x}`);
    // Add trailing zeros to ensure the number of decimal places is `round`
  }
  return `${a[0]}, ${a[1]}, ${a[2]}, ${a[3]},\n${a[4]}, ${a[5]}, ${a[6]}, ${a[7]},\n${a[8]}, ${a[9]}, ${a[10]}, ${a[11]}`
}


// TODO: Add metrics
// const x = [0, 1, 2, 3, 4, 5];
// const y = [0, 1, 4, 9, 16, 25];
// const chart = [...document.getElementsByTagName('svg')].forEach((svg) => drawChart({svg, x, y, xLabel: "iterations", yLabel: "PSNR"}));
