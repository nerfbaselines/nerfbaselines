#!/bin/bash
method="$1"
zipfile="$2"
outfile="$3"
path="$PWD"

if [ "$method" == "mip-splatting" ]; then exit 1; fi

if [ ! -e /tmp/gaussian-splats-3d ]; then
    rm -rf "/tmp/gaussian-splats-3d-tmp"
    git clone https://github.com/mkkellogg/GaussianSplats3D.git "/tmp/gaussian-splats-3d-tmp"
    cd /tmp/gaussian-splats-3d-tmp
    npm install
    npm run build
    cd "$PWD"
    mv /tmp/gaussian-splats-3d-tmp /tmp/gaussian-splats-3d
fi

plyfile="/tmp/$RANDOM.ply"
unzip -p "$zipfile" "**/*.ply" > "$plyfile"
node /tmp/gaussian-splats-3d/util/create-ksplat.js "$plyfile" "$outfile"

