import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';


function makeMatrix4(elements) {
  if (!elements || elements.length !== 12) {
    throw new Error("Invalid elements array. Expected 12 elements.");
  }
  return new THREE.Matrix4().set(...elements, 0, 0, 0, 1);
}


export class MeshFrameRenderer {
  constructor({ 
    mesh_url, 
    background_color, 
    znear=0.001, zfar=1000,
    update_notification,
    onready,
  }) {
    this._onready = onready;
    this.update_notification = update_notification;
    this._notificationId = "MeshFrameRenderer-" + (
      MeshFrameRenderer._notificationIdCounter = (MeshFrameRenderer._notificationIdCounter || 0) + 1);
    this.near = znear || 0.001;
    this.far = zfar || 1000;
    this.scene = new THREE.Scene();
    if (Array.isArray(background_color))
      background_color = new THREE.Color(...background_color);
    this.mesh_url = mesh_url;
    this.scene.background = new THREE.Color(background_color || 0x000000);
    this.camera = new THREE.Camera();
    this.canvas = new OffscreenCanvas(1, 1);
    this._flipCanvas = new OffscreenCanvas(1, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, canvas: this.canvas });
    this._loadPlyModel(mesh_url);
    this.output_types = ["color"];
  }

  async _loadPlyModel(mesh_url) {
    const controller = new AbortController();
    let cancelled = false;
    this._cancelLoading = () => {
      if (!cancelled) {
        controller.abort();
      }
      cancelled = true;
      this.update_notification({ id: this.notificationId, autoclose: 0 });
    };
    try {
      // Update progress callback
      const updateProgress = (percentage) => {
        if (cancelled) return;
        this.update_notification({
          id: this._notificationId,
          header: "Loading mesh renderer",
          progress: percentage,
          closeable: false,
          onclose: () => this._cancelLoading?.(),
        });
      };
      updateProgress(0);

      // Fetch
      let response = await fetch(mesh_url, { signal: controller.signal });
      if (!response.ok) {
        throw new Error(`Failed to load mesh: ${response.statusText}`);
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
      geometry.computeVertexNormals();
      const material = new THREE.MeshBasicMaterial({ vertexColors: true });
      const mesh = new THREE.Mesh(geometry, material);
      if (cancelled) return;
      this.scene.add(mesh);
      this.update_notification({ id: this._notificationId, autoclose: 0 });
      this._onready?.({
        output_types: this.output_types,
      });
    } catch (error) {
      if (cancelled) return;
      console.error('An error occurred while loading the PLY file:', error);
      this.update_notification({
        id: this._notificationId,
        header: "Error loading mesh renderer",
        detail: error.message,
        type: "error",
        closeable: true,
      });
    } finally {
      this._cancelLoading = undefined;
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

    const matrix = makeMatrix4(params.pose);
    const _R_threecam_cam = new THREE.Matrix4().makeRotationX(Math.PI);
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
    renderer = new MeshFrameRenderer({
      ...e.data.options,
      update_notification: (notification) => {
        notification.onclose = "cancel";
        postMessage({ type: "notification", notification });
      },
      onready: (data) => {
        postMessage({ type: "ready", ...data });
      }
    });
  }
  if (e.data.type === "cancel") {
    if (renderer) {
      renderer._cancelLoading?.();
    }
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
