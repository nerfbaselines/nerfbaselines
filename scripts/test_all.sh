#!/usr/bin/env bash
args=()
method="*"
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --method)
            method="$2"
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
    echo "Running $_method on $_data"
    nerfbaselines test-method --method $_method --data external://$_data "${args[@]}"
    result=$?
    if [ $result -eq 0 ]; then
        result_row="  \033[92m\u2713$_method on $_data: PASSED\033[0m"
    else
        result_row="  \033[91m\u2717$_method on $data: FAILED\033[0m"
    fi
    result_rows="$result_row\n"
}

# Run all tests
test_method gaussian-splatting mipnerf360/garden
test_method gaussian-splatting blender/lego
# test_method mip-splatting mipnerf360/garden
# test_method mip-splatting blender/lego
# test_method gaussian-opacity-fields mipnerf360/garden
# test_method gaussian-opacity-fields blender/lego
# test_method zipnerf mipnerf360/garden
# test_method zipnerf blender/lego

# Print results
echo -ne $result_rows
