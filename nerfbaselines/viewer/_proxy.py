import shutil
import json
import signal
import tempfile
import time
import re
import io
import tarfile
import os
import stat
import platform
import pathlib
import subprocess
import logging
from contextlib import contextmanager
from nerfbaselines.io import wget


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proxy")

# Directory where we'll cache the cloudflared binary
CACHE_DIR = pathlib.Path.home() / ".cache" / "nerfbaselines" / "cache"

# Example: stable or latest release. Adjust or pin the version as needed.
CLOUDFLARED_LATEST_API = "https://api.github.com/repos/cloudflare/cloudflared/releases/latest"

def _get_os_arch():
    system = platform.system().lower()   # e.g., 'windows', 'linux', 'darwin'
    machine = platform.machine().lower() # e.g., 'x86_64', 'amd64', 'arm64', etc.

    # Normalize architecture naming
    # Cloudflare typically uses "amd64" or "arm64"
    if machine in ["x86_64", "amd64"]:
        arch = "amd64"
    elif machine in ["aarch64", "arm64"]:
        arch = "arm64"
    else:
        # fallback
        arch = machine

    return system, arch

def _construct_filename(system: str, arch: str) -> str:
    """
    Construct the expected filename for the cloudflared binary on GitHub releases.
    """
    # For windows, cloudflared is shipped as "cloudflared-windows-amd64.exe"
    # For macOS, typically "cloudflared-darwin-amd64" or "cloudflared-darwin-arm64"
    # For linux, "cloudflared-linux-amd64" or "cloudflared-linux-arm64", etc.
    ext = ""
    if system == "windows": ext = ".exe"
    if system == "darwin": ext = ".tgz"
    filename = f"cloudflared-{system}-{arch}{ext}"
    return filename

def _download_file(url: str, dest: pathlib.Path) -> None:
    """
    Download a file from `url` to `dest`.
    """
    logger.info(f"Downloading {url} -> {dest}")
    with open(dest, "wb") as fout:
        if str(dest).endswith(".tgz"):
            with io.BytesIO() as buffered:
                wget(url, buffered)
                buffered.seek(0)
                with tarfile.open(fileobj=buffered, mode="r:gz") as tar:
                    for member in tar.getmembers():
                        if not member.isfile():
                            continue
                        file = tar.extractfile(member)
                        if file is None:
                            continue
                        with file:
                            fout.write(file.read())
        else:
            wget(url, fout)

    # On Unix-like systems, make the file executable
    if os.name != "nt":
        st = os.stat(dest)
        os.chmod(dest, st.st_mode | stat.S_IEXEC)

def _get_cloudflared_binary_path() -> pathlib.Path:
    """
    Returns the path to the cloudflared binary in the local cache,
    downloading it if necessary.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    system, arch = _get_os_arch()
    filename = _construct_filename(system, arch)
    cloudflared_path = CACHE_DIR / filename

    if cloudflared_path.is_file():
        return cloudflared_path

    # Otherwise, we need to download the binary from the releases
    print("Cloudflared not found in cache; downloading the latest release...")

    # Hit the GitHub releases API to get the latest version
    with wget(CLOUDFLARED_LATEST_API) as f:
        data = json.load(f)

    # data["assets"] is a list of all published binaries
    # We'll search for the one that matches our `filename`
    assets = data.get("assets", [])
    download_url = None
    for asset in assets:
        if asset["name"].startswith(filename):
            download_url = asset["browser_download_url"]
            break

    if not download_url:
        raise RuntimeError(
            f"Could not find a suitable cloudflared binary for {filename} "
            f"in the latest release assets."
        )

    _download_file(download_url, cloudflared_path)
    return cloudflared_path

@contextmanager
def cloudflared_tunnel(local_url: str, *, accept_license_terms=False):
    """
    Context manager to:
    1. Ensure cloudflared is downloaded.
    2. Launch `cloudflared tunnel --url local_url`.
    3. Wait until we see a line in the output that indicates success
       (or the process ends).
    4. Parse the output to find the 'trycloudflare.com' (or similar) URL.
    5. Yield just the discovered tunnel URL.
    6. Cleanup: kill the process upon exit.
    """
    if not accept_license_terms:
        raise RuntimeError("License terms (https://www.cloudflare.com/website-terms/) must be accepted")
    cloudflared_path = _get_cloudflared_binary_path()

    # Launch a Popen process
    with tempfile.TemporaryDirectory() as tmpdir:
        process = subprocess.Popen(
            [str(cloudflared_path), "tunnel", "--url", local_url, "--logfile", os.path.join(tmpdir, "cloudflared.log")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True
        )
        time.sleep(4)
        tunnel_url = None  # We’ll store the extracted URL here (if found)
        # Sometimes it takes long time to display URL in cloudflared metrices.
        printed_lines = []
        try:
            finished = False
            for _ in range(20):
                if process.poll() is not None:
                    message = "Cloudflared failed with code:" + str(process.returncode) + "."
                    if process.stdout:
                        error = process.stdout.read().splitlines()[-1]
                        message += " " + error
                    raise RuntimeError(message)
                with open(os.path.join(tmpdir, "cloudflared.log")) as f:
                    lines = f.read().splitlines()
                    for line in lines[len(printed_lines):]:
                        line_msg = json.loads(line)
                        line = line_msg.get("message")
                        level = line_msg.get("level")
                        if level not in ["info", "error", "debug"]:
                            level = "info"
                        getattr(logger, level)(line)
                        match = re.search(r"(https://[^\s]+\.trycloudflare\.com)", line)
                        if match:
                            tunnel_url = match.group(1)
                        elif "Registered tunnel connection" in line:
                            finished = True
                            break
                    printed_lines = lines
                    if finished:
                        break
                time.sleep(1)
            if tunnel_url is None or not finished:
                raise RuntimeError("Failed to establish a cloudflared tunnel")
            yield tunnel_url

        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait()


if __name__ == "__main__":
    # Example usage:
    with cloudflared_tunnel("http://localhost:5001", accept_license_terms=True) as proc:
        print("Tunnel started", proc)
        while True:
            pass
