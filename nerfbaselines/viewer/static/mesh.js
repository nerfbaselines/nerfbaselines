import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import palettes from './palettes.js';


function makeMatrix4(elements) {
  if (!elements || elements.length !== 12) {
    throw new Error("Invalid elements array. Expected 12 elements.");
  }
  return new THREE.Matrix4().set(...elements, 0, 0, 0, 1);
}





export const depth_vertex_shader = /* glsl */`
varying vec2 vUv;
varying vec2 vHighPrecisionZW;

void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
	vHighPrecisionZW = gl_Position.zw;
}
`;

export const depth_fragment_shader = /* glsl */`
uniform float znear;
uniform float zfar;
uniform float range_min;
uniform float range_max;
uniform sampler2D palette;

varying vec2 vHighPrecisionZW;

void main() {
  float fragCoordZ = vHighPrecisionZW[0] / vHighPrecisionZW[1];
  float zbuffer = (2.0 * znear * zfar) / (zfar + znear - fragCoordZ * (zfar - znear));
  float mapped = 1.0 - 1.0 / (1.0 + zbuffer);
  float zvalue = (mapped - range_min) / (range_max - range_min);
  vec3 zcolor = texture(palette, vec2(1.0-zvalue, 0)).rgb;
  gl_FragColor = vec4(zcolor, 1.0);
}
`;

const mapValue = x => 1.0 - (1.0 / (1.0 + x));


class MeshDepthMaterial extends THREE.ShaderMaterial {
  constructor({ palette, ...rest }) {
    super({
      uniforms: {
        'tDiffuse': { value: null },
        'opacity': { value: 1.0 },
        'znear': { value: 0.001 },
        'zfar': { value: 1000 },
        'range_min': { value: mapValue(2.0) },
        'range_max': { value: mapValue(6.0) },
        'palette': { value: palette, type: 't'},
      },
      vertexShader: depth_vertex_shader,
      fragmentShader: depth_fragment_shader,
      ...rest,
    });
  }
};


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
    this.defaultBackground = new THREE.Color(background_color || 0x000000);
    this.scene.background = null;
    this.camera = new THREE.Camera();
    this.canvas = new OffscreenCanvas(1, 1);
    this._flipCanvas = new OffscreenCanvas(1, 1);
    this.renderer = new THREE.WebGLRenderer({antialias: true, canvas: this.canvas, stencil: true});
    this.renderer.autoClear = false;
    this._loadPlyModel(mesh_url);
    this._splitScene = new THREE.Scene();
    this._splitScene.background = null;
    this._splitCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this._splitGeometry = new THREE.BufferGeometry();
    this._splitScene.add(new THREE.Mesh(
      this._splitGeometry,
      new THREE.MeshBasicMaterial({ 
        color: 0xffffff,
        colorWrite: true,
        depthWrite: false, 
        depthTest: false,
        stencilWrite: true,
        stencilRef: 1,
        stencilFunc: THREE.AlwaysStencilFunc,
        stencilFail: THREE.ReplaceStencilOp,
        stencilZFail: THREE.ReplaceStencilOp,
        stencilZPass: THREE.ReplaceStencilOp,
      })
    ));

    this.material = new THREE.MeshBasicMaterial({ 
      vertexColors: true,
      stencilWrite: true,
      stencilRef: 1,
      stencilFunc: THREE.EqualStencilFunc,
      stencilFail: THREE.KeepStencilOp,
      stencilZFail: THREE.KeepStencilOp,
      stencilZPass: THREE.KeepStencilOp,
    });
    this.normalMaterial = new THREE.MeshNormalMaterial({
      stencilWrite: true,
      stencilRef: 1,
      stencilFunc: THREE.EqualStencilFunc,
      stencilFail: THREE.KeepStencilOp,
      stencilZFail: THREE.KeepStencilOp,
      stencilZPass: THREE.KeepStencilOp,
    });
    this.output_types = ["color", "depth", "normal"];
    this.palettes = {};
    for (const name in palettes) {
      const palette = palettes[name];
      const numColors = palette.length / 3;
      const data = new Uint8Array(numColors * 4);
      for (let i = 0; i < numColors; i++) {
        data[i * 4 + 0] = palette[i*3];
        data[i * 4 + 1] = palette[i*3+1];
        data[i * 4 + 2] = palette[i*3+2];
        data[i * 4 + 3] = 255;
      }
      this.palettes[name] = new THREE.DataTexture(data, numColors, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
      this.palettes[name].needsUpdate = true;
    }
    this.depthMaterial = new MeshDepthMaterial({ 
      palette: null,
      stencilWrite: true,
      stencilRef: 1,
      stencilFunc: THREE.EqualStencilFunc,
      stencilFail: THREE.KeepStencilOp,
      stencilZFail: THREE.KeepStencilOp,
      stencilZPass: THREE.KeepStencilOp,
    });
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
      const mesh = new THREE.Mesh(geometry, this.material);
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

  _setMaterial(params, isSplit) {
    // Set material
    const outputType = isSplit ? params.split_output_type : params.output_type;
    const palette = isSplit ? params.split_palette : params.palette;
    const outputRange = isSplit ? params.split_range : params.output_range;
    this.scene.children.forEach((mesh) => {
      let material = null;
      if (outputType === "normal") {
        mesh.material = this.normalMaterial;
      } else if (outputType === "depth") {
        mesh.material = this.depthMaterial;
        const paletteData = this.palettes[palette || "viridis"];
        this.depthMaterial.uniforms.palette.value = paletteData;
        this.depthMaterial.uniforms.znear.value = this.near;
        this.depthMaterial.uniforms.zfar.value = this.far;
        const concreteValue = x => x !== null && x !== undefined && x !== "" && isFinite(x);
        this.depthMaterial.uniforms.range_min.value = concreteValue(outputRange?.[0]) ? mapValue(outputRange[0]) : 0;
        this.depthMaterial.uniforms.range_max.value = concreteValue(outputRange?.[1]) ? mapValue(outputRange[1]) : 1;
      } else {
        mesh.material = this.material;
      }
      mesh.material.stencilFunc = isSplit ? THREE.EqualStencilFunc : THREE.NotEqualStencilFunc;
    });
  }

  _updateSplitGeometry({ split_tilt=0, split_percentage=0.5, width, height } = {}) {
    const tiltRadians = -split_tilt * Math.PI / 180;
    const splitDir = new THREE.Vector2(Math.cos(tiltRadians), Math.sin(tiltRadians));
    const splitDirLen = width / 2 * Math.abs(splitDir.x) + height / 2 * Math.abs(splitDir.y);
    const m = split_percentage * 2 - 1;
    // const sp = new THREE.Vector3(splitDir.x * m, splitDir.y * m, 0);
    const sp = new THREE.Vector3(
      2 * (splitDir.x * splitDirLen * m) / width,
      2 * (splitDir.y * splitDirLen * m) / height,
      0
    );
    const sdNorm = new THREE.Vector2(splitDir.x * width, splitDir.y * height).normalize();
    const sd = new THREE.Vector3(sdNorm.x, sdNorm.y, 0);
    const so = new THREE.Vector3(sd.y, -sd.x, 0);
    const c = 4;
    this._splitGeometry.setFromPoints([
      sp.clone().addScaledVector(so, -c),
      sp.clone().addScaledVector(so, c),
      sp.clone().addScaledVector(so, -c).addScaledVector(sd, c),
      sp.clone().addScaledVector(so, c),
      sp.clone().addScaledVector(so, c).addScaledVector(sd, c),
      sp.clone().addScaledVector(so, -c).addScaledVector(sd, c),
    ]);
  }

  _getBackground(params, isSplit) {
    // Set material
    const outputType = isSplit ? params.split_output_type : params.output_type;
    if (outputType === "normal") {
      return new THREE.Color(0x000000);
    } else if (outputType === "depth") {
      const palette = isSplit ? params.split_palette : params.palette;
      const outputRange = isSplit ? params.split_output_range : params.output_range;
      const paletteData = this.palettes[palette || "viridis"];
      const concreteValue = x => x !== null && x !== undefined && x !== "" && isFinite(x);
      const rangeMin = concreteValue(outputRange?.[0]) ? mapValue(outputRange[0]) : 0;
      const rangeMax = concreteValue(outputRange?.[1]) ? mapValue(outputRange[1]) : 1;
      const coloridx = (rangeMax > rangeMin) ? 
        0 : paletteData.source.data.data.length/4-1;
      return new THREE.Color().setRGB(
        paletteData.source.data.data[coloridx*4+0]/255,
        paletteData.source.data.data[coloridx*4+1]/255,
        paletteData.source.data.data[coloridx*4+2]/255,
        THREE.SRGBColorSpace
      );
    } else {
      return this.defaultBackground;
    }
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
    this.camera.updateMatrixWorld();

    // Clear all
    this._setMaterial(params);
    this.renderer.setClearColor(this._getBackground(params, false), 1);
    this.renderer.clear();

    // If split is enabled, render two images by using stencil buffer
    if (params.split_percentage) {
      this._updateSplitGeometry({ ...params, width, height });
      // Transform points camera space to world space
      this._splitGeometry.computeBoundingSphere();
      this._splitScene.children[0].material.color = this._getBackground(params, true);

      // Render split mask, but do not clear buffers
      this.renderer.render(this._splitScene, this._splitCamera);
      this.renderer.clearDepth();

      // Render left half
      this.renderer.render(this.scene, this.camera);
      
      // Set split material
      this._setMaterial(params, true);
      this.renderer.render(this.scene, this.camera);
    } else {
      this.renderer.clear();
      this._setMaterial(params);
      this.renderer.render(this.scene, this.camera);
    }

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
