# Viewer
```{figure} images/viewer.jpg

**NerfBaselines viewer.** The viewer enables users to visualize the trained models, input datasets, generate camera trajectories, and render videos. The viewer runs a HTTP webpage which can be accessed from the web browser or used inside a Jupyter Notebook.
```

## Getting started
The viewer runs a HTTP webpage which can be accessed from the web browser.
The intended use case is to launch the viewer on the machine with the GPU used for fast training/rendering,
and open the viewer either locally (if running on the same machine), or [remotely](#connecting-to-the-viewer-remotely) (if running on remote machine, e.g. a cluster). To start the viewer, run the following command:
```bash
nerfbaselines viewer --checkpoint <checkpoint> --data <data>
```
where `<checkpoint>` is the path to the directory with the trained model checkpoint and `<data>` is the path to the dataset directory. You can specify either only `--checkpoint` or only `--data` or both (recommended). By default, the viewer will run on port 6006, but it can be changed by using the `--port` argument.


## Camera control
Currently, there are two camera control options implemented ({menuselection}`Control --> Camera mode`):
- `orbit` - the camera will orbit around a single point called **target**.
- `fps` - first-person-shooter-style camera control where moving mouse will rotate the camera around its center.

For both modes, camera movement is activated by pressing down the left mouse button and moving the mouse (or pressing the touch pad).
Moving two fingers (without pressing) down/up will zoom-in/zoom-out (move camera forward/backward in the direction it is looking).
The same can be achieved by using the mouse wheel. Two-finger press gesture or using right mouse key will result in panning 
motion (moving the camera in the plane perpendicular to the look-at direction).
Note, the sensitivity can by changed in the {octicon}`gear` *Settings* tab.

Keyboard can also be used for the movement. {kbd}`W`, {kbd}`S`, {kbd}`A`, {kbd}`D` move across the plane perpendicular to the up direction.
{kbd}`Q`, {kbd}`E` move up and down. When {kbd}`Alt` is pressed, the movement is not perpendicular to the up direction, but relative to the
current camera orientation. When {kbd}`Shift` is pressed, the movement is faster.
For the `fps` mode, arrow keys {kbd}`←`, {kbd}`→`, {kbd}`↑`, {kbd}`↓` move the camera in the plane perpendicular to the look-at direction (panning).
In the `orbit` mode, arrow keys {kbd}`←`, {kbd}`→`, {kbd}`↑`, {kbd}`↓` rotate the camera around the target point.
Keys {kbd}`Z`, {kbd}`X` will rotate the camera around the look-at direction (changing the camera-up vector).
The sensitivity can be changed in the {octicon}`gear` {menuselection}`Settings --> Keyboard speed`.

(connecting-to-the-viewer-remotely)=
## Connecting to the viewer remotely
If you want to run the viewer on a remote machine, e.g. a cluster, you can launch the viewer from there, and forward
the appropriate port to connect to the viewer server from the web browser. 
In case SSH connection to the remote server is possible, 
connecting to the remote viewer could be establishing an SSH tunnel (we assume the viewer was ran on port 6006):
```bash
ssh -L 6006:localhost:6006 <user>@<host>
```

```{figure} images/viewer-setup-public-url.jpg
:align: right
:figwidth: 35%

By pressing {guilabel}`Setup public url`, the public url will be generated and shown in the *public url* box.
```

However, other cases (SSH not available or is difficult to setup), NerfBaselines enables you
to setup a proxy server to make the server publicly available from anywhere.
The viewer will generate a *public url* which can be inputted into the web browser directly.
Go to {octicon}`gear` {menuselection}`Settings --> Setup public url` and the public url will appear in the *public url* box above the 
*Setup public url* button.  NerfBaselines will download and install `cloudflared` to setup the proxy.
You will be requested to accept Cloudflare's terms and conditions before proceeding and 
after accepting, the tunnel will be set up. It can take around 20 seconds to setup the tunnel.
A notification in the notification are will keep you informed about the current status.
After the public url is generated, it can be copied to clipboard by pressing the *Copy public url* button
in the {octicon}`gear` *Settings* tab.

## Viewing dataset
```{figure} images/viewer-dataset.jpg

**Viewer interface with dataset visualization.** In the {menuselection}`Control --> Dataset` section, you can select to visualize the train or test dataset cameras and the point cloud. Each camera is represented by a clickable pyramid which upon clicking will display the camera details in the control panel.
```

```{figure} images/viewer-dataset-camera.jpg
:align: right
:figwidth: 40%

**Camera details** are displayed in the control panel after clicking on the camera pyramid. The details include the camera parameters, image, etc.
```

The dataset (cameras and point cloud) can be visualized by the NerfBaselines viewer. In the {menuselection}`Control --> Dataset` section, you can select to visualize the train or test dataset cameras and the point cloud.
After selecting any of the options, the dataset will be downloaded and visualized. A notification with a progress bar will appear in the notification area, keeping you informed about the current status. After the cameras are downloaded, they will appear in the viewer. Each camera is represented by a pyramid with the camera position at the apex and the camera orientation at the base. 
By clicking on the camera, the camera will be selected and details will be displayed in the control panel.
The details include the camera pose and intrinsics, the image filename, and the image itself. There is also a {guilabel}`Navigate` button which will move the camera to the center of the view.
The point cloud will be visualized as a set of points, where the size of the points can be adjusted ({menuselection}`Control --> Point cloud size`).


## Creating camera trajectories
```{figure} images/viewer-trajectory.jpg

**Camera trajectory editor.** Keyframes can be added by pressing {guilabel}`Add keyframe`. The keyframe appear both in a list of keyframe and in the viewer window. By clicking on the keyframe, the keyframe becomes selected and can be edited. The camera trajectory can be previewed by pressing {guilabel}`Preview trajectory` and {guilabel}`Play`.
```

The camera trajectory can be created by adding keyframes. Keyframes can be added by pressing the {guilabel}`Add keyframe` button. After adding the keyframe, it will appear in the {menuselection}`Trajectory --> Keyframes` list and also in the viewer window as a camera frustum. By clicking on the keyframe, the keyframe becomes selected and can be edited. A window will appear with keyframe properties including fov, velocity, appearance, etc. In the viewer window, the selected keyframe can be moved and rotated by dragging the rotation and translation handles. In the {menuselection}`Trajectory --> Keyframes` list, the keyframes can be reordered by dragging them up and down. They can be deleted by pressing the {octicon}`trash` icon, duplicated by pressing the {octicon}`copy` button, or moved up and down by pressing the {octicon}`arrow-up` and {octicon}`arrow-down` buttons. 

The camera trajectory can be previewed in the {menuselection}`Trajectory --> Playback` panel. By pressing the {guilabel}`Preview render` button, the viewer will enter `preview` mode, where it will render the scene from the camera position at a fixed time ({guilabel}`Preview frame`) and it will use selected aspect ratio, fov, and other properties. By dragging the {guilabel}`Preview frame` slider, the camera trajectory can be inspected frame by frame as it will appear in the final video. The camera trajectory can be played by pressing the {guilabel}`Play` button.

```{figure} images/viewer-trajectory-interpolation.jpg
:figwidth: 35%
:align: right

**Camera trajectory interpolation.** The keyframes are interpolated to obtain a smooth trajectory. The interpolation method can be set in the {menuselection}`Trajectory --> Interpolation` dropdown.
```
After adding the keyframes, they are interplated to obtain a smooth trajectory. There are different options on how the keyframes are interpolated ({menuselection}`Trajectory --> Interpolation`):
- `linear` - linear interpolation between the keyframes.
- `kochanek-bartels` - Kochanek-Bartels spline interpolation. This method allows for more control over the interpolation by setting the tension, bias, and continuity parameters. The tension parameter controls the tightness of the curve, the bias parameter controls the direction of the curve, and the continuity parameter controls the smoothness of the curve.
- `circle` - circular interpolation. A best-fit circle is found for the keyframes and the camera is moved along the circle.
- `none` - no interpolation. The keyframes are treated as individual images. See [Rendering individual images](#rendering-individual-images).

There are two modes of how the transition times are computed ({guilabel}`Speed interpolation`):
- `velocity` - first, the distance of all trajectory segment is computed. The distance is exponentiated to the power of {guilabel}`Distance alpha`.
A base velocity is set at 1, but at each keyframe it can be overriden. After that, the velocity is interpolated using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) and the transition times are computed by integrating and normalizing the interpolated velocity. By changing the value of {guilabel}`Distance alpha`, you can choose to put more emphasis on the longer or shorter segments.
- `time` - the transition times are set manually by the user. The default transition duration can be set in the {menuselection}`Trajectory --> Transition duration` input. The transition duration can be set for each keyframe individually. The transition duration is the time it takes to move from the current keyframe to the next keyframe (in seconds). Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) is used to smooth over sharp changes in transition duration between keyframes. In this mode, the {guilabel}`Distance alpha` parameter is used in `kochanek-bartels` interpolation to ensure smooth trajectory (putting more weight on longer segments).
Finally, there is the {guilabel}`Loop` option which will make the camera trajectory loop back to the first keyframe after reaching the last keyframe.


(rendering-individual-images)=
## Rendering individual images
Instead of creating a smooth trajectory, the {menuselection}`Trajectory` tab can be used to create a sequence of images from manually placed cameras to be later exported as individual images (see [Rendering video](#rendering-video)).
This can be done by setting {guilabel}`Interpolation` to `none`. In this mode, the keyframes will not be interpolated, but treated as individual images. Notice, that after selecting this option, the {guilabel}`Transition duration` and {guilabel}`FPS` options are still visible. They will be used when generating a video from the images (see [Rendering video](#rendering-video)).


## Exporting/loading camera trajectories
After creating a camera trajectory, it can be exported to a `json` file to be loaded later or to be used with `nerfbaselines render-trajectory` command. The camera trajectory can be exported by pressing the {guilabel}`Export trajectory` button in the {menuselection}`Trajectory` tab. Later, the trajectory can be loaded by pressing the {guilabel}`Load trajectory` button in the same tab.

(rendering-video)=
## Rendering video
After you create a camera trajectory and are satisfied with the result (you can preview the video by pressing the {guilabel}`Preview render` button and pressing {guilabel}`Play`), you can render the video by pressing the {guilabel}`Render video` button. A *save-as* dialog will pop up and after selecting a file, the video will start rendering in the background and a notification will appear in the notification area with the progress bar. After the video is rendered, it will  be downloaded to the location selected in the *save-as* dialog. The export resolution can be set by changing the value of {menuselection}`Trajectory --> Export resolution` input. The output video can be saved as either a `mp4`, a `webm`, or a sequence of `png` images which will be zipped. You can also set the number of frames per second in the {menuselection}`Trajectory --> FPS` input, or the total duration of the video in the {menuselection}`Trajectory --> Duration` input. You can also specify which output type will be rendered (color, depth, etc.) in the {menuselection}`Trajectory --> Output type` input. Detailed export settings including codec, etc. can be set in the {octicon}`gear` {menuselection}`Settings --> Render configuration` tab.

The codec string is has the following format:
```
codec[;option1=value1;option2=value2;...]
```
For detailed information about the codec string, see [Web Codecs documentation](https://developer.mozilla.org/en-US/docs/Web/Media/Formats/codecs_parameter#codec_options_by_container).
The options can include the following:
- `bps` - bitrate in bits per second.
- `bppf` - bitrate per frame in bits per pixel. `bps = bppf * width * height * fps`.
- `crf` - constant rate factor. When specified, `bps` and `bppf` are ignored. The values are in the range 0-51, where 0 is best and 51 is the worst quality. For some codecs, the max value is 63. Recommended default value is 23. See [VideoEncoder.encode](https://developer.mozilla.org/en-US/docs/Web/API/VideoEncoder/encode).

## Implementation details
The viewer is implemented in the {py:class}`nerfbaselines.viewer.Viewer` class. 
The class spawns a background process which handles the HTTP requests and communicates with the connected browser clients.
The background process launches a simple HTTP server (using the `http.server` module) which serves the static files (HTML, JS, CSS) and handles the WebSocket connections (custom lightweight implementation). Each time a connected browser client wants to render a frame,
its sends a message to the server using the WebSocket connection. The viewer background process then adds the request to a queue called `request_queue` and waits for a corresponding message to appear in the `output_queue` containing the rendered frame. The rendering itself is, therefore, not done in the background process, but in the main process. The main motivation is to avoid overhead with locking (when simultaneously rendering and training) and to give full control to the users over the rendering process. To handle the actual rendering, users must periodically call `Viewer.update()` which will check the `request_queue` and handle a single request if it is present (taking a single message from the `request_queue` and putting the rendered frame into the `output_queue`).
For convenience, the `Viewer` class also provides a method `Viewer.run()` which will periodically call `Viewer.update()` until the viewer is closed. 

## Using Jupyter Notebook or Google Colab
```{button-link} https://colab.research.google.com/github/nerfbaselines/nerfbaselines/blob/main/docs/viewer-notebook.ipynb
:color: primary
:outline:
:align: left
Open in Colab
```

```{eval-rst}
.. include:: viewer-notebook.ipynb
    :parser: myst_nb.docutils_
```

