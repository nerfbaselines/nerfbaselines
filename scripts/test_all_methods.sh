#!/usr/bin/env bash
exec 3>&1

args=()
method="*"
data="*"
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --method)
            method="$2"
            shift
            shift
            ;;
        --data)
            data="$2"
            shift
            shift
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done
set -- "${args[@]}"

result_rows=""

# Run all tests
function test_method {
    _method=$1
    _data=$2
    if [[ "$_method" != $method ]]; then
        echo "Skipping $_method on $_data (only running $method)"
        return
    fi
    if [[ "$_data" != $data ]]; then
        echo "Skipping $_method on $_data (only running on $data)"
        return
    fi
    echo "Running $_method on $_data"
    report="$(set -o pipefail; nerfbaselines test-method --method $_method --data external://$_data "${args[@]}" | tee /dev/fd/3)"
    result=$?

    # Strip first line of the output
    if [ "$result" -eq 0 ]; then
        result_rows+="\033[92m\u2713 $_method on $_data: PASSED\033[0m\n"
    else
        result_rows+="\033[91m\u2717 $_method on $_data: FAILED\033[0m\n"
        if [ ! -z "$report" ]; then
            report="$(echo -e "$report" | tail -n +2)"
            result_rows+="$report\n"
        fi
    fi
}

mipnerf360="mipnerf360/garden"
mipnerf360_sparse="mipnerf360-sparse/garden-n12"
blender="blender/lego"
phototourism="phototourism/sacre-coeur"
seathru_nerf="seathru-nerf/curasao"
llff="llff/fern"

# Run all tests
test_method zipnerf $mipnerf360
test_method zipnerf $blender

test_method gaussian-opacity-fields $mipnerf360
test_method gaussian-opacity-fields $blender

test_method gaussian-splatting $mipnerf360
test_method gaussian-splatting $blender

test_method gaussian-splatting-wild $phototourism

test_method kplanes $phototourism
test_method kplanes $blender

test_method mip-splatting $mipnerf360
test_method mip-splatting $blender

test_method mipnerf360 $mipnerf360
test_method mipnerf360 $blender
test_method mipnerf360 $llff

test_method nerf $blender
test_method nerf $llff

test_method instant-ngp $mipnerf360
test_method instant-ngp $blender

test_method nerfacto $mipnerf360
test_method nerfacto $blender

test_method nerfw-reimpl $phototourism

test_method scaffold-gs $blender
test_method scaffold-gs $mipnerf360
test_method scaffold-gs $phototourism

test_method seathru-nerf $seathru_nerf

test_method taming-3dgs $mipnerf360

test_method 3dgs-mcmc $mipnerf360

test_method tensorf $blender
test_method tensorf $llff

test_method tetra-nerf $mipnerf360
test_method tetra-nerf $blender

test_method water-splatting $seathru_nerf

test_method wild-gaussians $mipnerf360
test_method wild-gaussians $phototourism

test_method sparsegs $mipnerf360_sparse
test_method dropgaussian $mipnerf360_sparse


# Print results
echo ""
echo "Summary:"
echo -ne "$result_rows"
