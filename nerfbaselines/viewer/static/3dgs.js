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
    scene_url, background_color, 
    znear=0.001, zfar=1000,
    update_notification,
    onready,
    ...options
  }) {
    this._notificationId = "GaussianSplattingFrameRenderer-" + (
      GaussianSplattingFrameRenderer._notificationIdCounter = (GaussianSplattingFrameRenderer._notificationIdCounter || 0) + 1);
    this._onready = onready;
    this.update_notification = update_notification;

    const updateProgress = (percentage, _, stage) => {
      if (stage === 1 && percentage >= 100) {
        this._onready({
          output_types: this.output_types,
        });
        this.update_notification({
          id: this._notificationId,
          autoclose: 0,
        });
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

    this.near = znear || 0.001;
    this.far = zfar || 1000;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(background_color || 0x000000);
    this.camera = new THREE.Camera();
    this.canvas = new OffscreenCanvas(1, 1);
    this._flipCanvas = new OffscreenCanvas(1, 1);
    updateProgress(0, undefined, 0);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, canvas: this.canvas });
    this.gs_viewer = new GaussianSplats3D.DropInViewer({
      ignoreDevicePixelRatio: true,
      gpuAcceleratedSort: false,
      sharedMemoryForWorkers: false,
      sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
    });
    this.gs_viewer.addSplatScene(scene_url, {
      showLoadingUI: false,
      onProgress: updateProgress,
      onError: (error) => {
        this.update_notification({
          id: this._notificationId,
          header: "Error loading 3DGS renderer",
          detail: error.message,
          type: "error",
          closeable: true,
        });
      }
    });
    this.scene.add(this.gs_viewer);
    this.output_types = ["color"];
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
