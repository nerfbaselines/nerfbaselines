import * as THREE from "three";
import * as GaussianSplats3D from "./third-party/gaussian-splats-3d.module.min.js";



function makeMatrix4(elements) {
  if (!elements || elements.length !== 12) {
    throw new Error("Invalid elements array. Expected 12 elements.");
  }
  return new THREE.Matrix4().set(...elements, 0, 0, 0, 1);
}


export class GaussianSplattingFrameRenderer {
  constructor({ 
    scene_url, 
    scene_url_per_appearance,
    background_color, 
    znear=0.001, zfar=1000,
    update_notification,
    onready,
    is_2DGS=false,
    antialias_2D_kernel_size,
    transform,
    ...options
  }) {
    this._notificationId = "GaussianSplattingFrameRenderer-" + (
      GaussianSplattingFrameRenderer._notificationIdCounter = (GaussianSplattingFrameRenderer._notificationIdCounter || 0) + 1);
    this._onready = onready;
    this.update_notification = update_notification;

    this.near = znear || 0.001;
    this.far = zfar || 1000;
    this.scene = new THREE.Scene();
    if (Array.isArray(background_color))
      background_color = new THREE.Color(...background_color);
    this.scene.background = new THREE.Color(background_color || 0x000000);
    this.camera = new THREE.Camera();
    this.canvas = new OffscreenCanvas(1, 1);
    this._flipCanvas = new OffscreenCanvas(1, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, canvas: this.canvas });
    this.gs_viewer = new GaussianSplats3D.DropInViewer({
      ignoreDevicePixelRatio: true,
      gpuAcceleratedSort: false,
      sharedMemoryForWorkers: false,
      sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
    });
    this.scene_url = scene_url;
    this.scene_url_per_appearance = scene_url_per_appearance;
    this.antialias_2D_kernel_size = antialias_2D_kernel_size;
    this.is_2DGS = is_2DGS;
    this.scene.add(this.gs_viewer);
    this.output_types = ["color"];
    this.transform = {}
    if (transform) {
      const rotation = new THREE.Quaternion();
      const position = new THREE.Vector3();
      const scale = new THREE.Vector3();
      makeMatrix4(transform).decompose(position, rotation, scale);
      this.transform = { 
        position: position.toArray(),
        rotation: rotation.toArray(),
        scale: scale.toArray(),
      };
    }
    this._changeScene(scene_url);
  }

  _changeScene(scene_url) {
    if (this._currentSceneUrl === scene_url) {
      return;
    }
    this._currentSceneUrl = scene_url;
    const changeScene = () => new Promise(async (resolve, reject) => {
      const updateProgress = (percentage, _, stage) => {
        if (stage === 2 && percentage >= 100) {
          // Trigger render to start sorting
          this.renderer.render(this.scene, this.camera);
        }
        if (stage === 1 && percentage >= 100) {
          const complete = () => {
            this._onready({
              output_types: this.output_types,
              supported_appearance_train_indices: this.scene_url_per_appearance ? Object.keys(this.scene_url_per_appearance) : null,
            });
            this.update_notification({
              id: this._notificationId,
              autoclose: 0,
            });
            resolve();
          };
          setTimeout(complete, 0);
        } else if (stage === 0) {
          this.update_notification({
            id: this._notificationId,
            header: "Loading 3DGS renderer - downloading scene",
            progress: percentage / 100,
            closeable: false,
          });
        } else if (stage === 1) {
          this.update_notification({
            id: this._notificationId,
            header: "Loading 3DGS renderer - processing",
            closeable: false,
          });
        }
      };

      updateProgress(0, undefined, 0);
      if (this.gs_viewer.getSceneCount() > 0) {
        await this.gs_viewer.removeSplatScene(0);
      }
      this.gs_viewer.addSplatScene(scene_url, {
        showLoadingUI: false,
        antialiased: this.antialias_2D_kernel_size > 0,
        kernel2DSize: this.antialias_2D_kernel_size || 0.3,
        splatRenderMode: this.is_2DGS ? "TwoD" : "ThreeD",
        sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
        onProgress: updateProgress,
        onError: (error) => {
          this.update_notification({
            id: this._notificationId,
            header: "Error loading 3DGS renderer",
            detail: error.message,
            type: "error",
            closeable: true,
          });
          reject(error);
        },
        ...this.transform
      });
    });
    if (this._changeScenePromise) {
      this._changeScenePromise = this._changeScenePromise.then(() => changeScene());
    } else {
      this._changeScenePromise = changeScene();
    }
  }

  _flipY(imageBitmap) {
    this._flipCanvas.width = imageBitmap.width;
    this._flipCanvas.height = imageBitmap.height;
    const ctx = this._flipCanvas.getContext("2d");
    ctx.translate(0, imageBitmap.height);
    ctx.scale(1, -1);
    ctx.drawImage(imageBitmap, 0, 0);
    return this._flipCanvas.transferToImageBitmap();
  }

  _makePerspective(matrix, w, h, fx, fy, cx, cy, coordinateSystem = THREE.WebGLCoordinateSystem) {
    const ti = matrix.elements;
    const w2 = w / 2;
    const h2 = h / 2;
    const near = this.near;
    const far = this.far;
    let c, d;

    if (coordinateSystem === THREE.WebGLCoordinateSystem) {
			c = - (far + near) / (far - near);
			d = (-2 * far * near) / (far - near);
		} else if (coordinateSystem === THREE.WebGPUCoordinateSystem) {
			c = - far / ( far - near );
			d = (-far * near) / (far - near);
		} else {
			throw new Error('Invalid coordinate system: ' + coordinateSystem);
		}
    ti[0] = 2*fx/w; ti[4] = 0;      ti[8] = 2*(cx-w2)/w; ti[12] = 0;
    ti[1] = 0;      ti[5] = 2*fy/h; ti[9] = 2*(cy-h2)/h; ti[13] = 0;
    ti[2] = 0;      ti[6] = 0;      ti[10] = c;          ti[14] = d;
    ti[3] = 0;      ti[7] = 0;      ti[11] = -1;         ti[15] = 0;
  }

  async render(params, { flipY = false } = {}) {
    const [width, height] = params.image_size;
    const needResize = this.canvas.width !== width || this.canvas.height !== height;
    if (needResize) {
      this.renderer.setSize(width, height, false);
    }

    this._makePerspective(
      this.camera.projectionMatrix,
      width, height, ...params.intrinsics,
      this.camera.coordinateSystem
    );
		this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
    const _R_threecam_cam = new THREE.Matrix4().makeRotationX(Math.PI);

    if (this.scene_url_per_appearance) {
      const scene_url = this.scene_url_per_appearance[params.appearance_train_indices?.[0]] || this.scene_url;
      this._changeScene(scene_url);
    }

    const matrix = makeMatrix4(params.pose);
    matrix.multiply(_R_threecam_cam.clone().invert());
    matrix.premultiply(this.scene.matrixWorld);
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    matrix.decompose(position, quaternion, scale);
    this.camera.position.copy(position);
    this.camera.quaternion.copy(quaternion);
    this.renderer.render(this.scene, this.camera);
    // We need to do the re-rendering due to imperfect sorting
    if (!this._lastCameraPosition || 
      (this._lastCameraPosition.distanceTo(position) > 0.001 && 
        performance.now() - this._lastRenderTime > 300)) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
      await new Promise((resolve) => setTimeout(resolve, 30));
      this.renderer.render(this.scene, this.camera);
    }
    this._lastCameraPosition = position.clone();
    this._lastRenderTime = performance.now();
    // Flip the image vertically
    let imageBitmap = await this.canvas.transferToImageBitmap();
    if (flipY)
        imageBitmap = this._flipY(imageBitmap);
    return imageBitmap;
  }
}


let renderer;
onmessage = async (e) => {
  if (e.data.type === "init") {
    renderer = new GaussianSplattingFrameRenderer({
      ...e.data.options,
      update_notification: (notification) => {
        postMessage({ type: "notification", notification });
      },
      onready: (data) => {
        postMessage({ type: "ready", ...data });
      }
    });
  }
  if (e.data.type === "render") {
    const requestId = e.data.requestId;
    if (!renderer) {
      postMessage({ type: "error", message: "Renderer not initialized" });
      return;
    }
    try {
      const imageBitmap = await renderer.render(e.data.params, e.data.options);
      postMessage({ type: "rendered", imageBitmap, requestId }, [imageBitmap]);
    } catch (error) {
      console.error(error);
      postMessage({ type: "rendered", error, requestId });
    }
  }
};
postMessage({ type: "loaded" });
