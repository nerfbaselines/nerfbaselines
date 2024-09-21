#!/usr/bin/env bash
exec 3>&1

args=()
function usage() {
    echo "Usage: $0 [input-path] [output-path]"
}
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --help|-h)
            usage
            exit 0
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done
set -- "${args[@]}"
if [ $# -lt 2 ]; then
    usage
    exit 1
fi
echo "Using input path: $1"
echo "Using output path: $2"
inpath="$1"
outpath="$2"


result_rows=""

# Process all output artifacts in the baselines directory
while IFS= read -r -d '' file; do
    # Base path to store output without the extension,
    # relative to the repository
    # e.g. blender/lego
    name=${file#"$inpath/"};name=${name%.zip}
    if [[ "$name" == *"/output" ]]; then
        name=$(dirname $name)
        if [ -e "$inpath/$name.zip" ]; then
            continue
        fi
    fi
    method=${name%%/*}

    # Absolute output path without the extension
    out="$outpath/${name}_mesh"
    mkdir -p "$(dirname $out)"

    # Check if there exists sha files for the results
    echo "Processing $name"
    oldsha="$([ -e "$out.zip.sha256" ] && cat "$out.zip.sha256")"
    newsha="$(sha256sum $file | cut -d' ' -f1)"
    
    if [ "$oldsha" == "$newsha" ]; then
        echo "No change in $name, skipping..."
        # Add skip row to results with info unicode character and gray color
        result_rows+="\033[90m\u2022 $name\033[0m\n"
        continue
    fi

    # Try to generate the demo
    (
        set -e
        nerfbaselines export-mesh --checkpoint $file/checkpoint --data external://$name --output ${out}
        # Export generic demo
        python -c "path='${out}'
from nerfbaselines.datasets import load_dataset
from nerfbaselines.methods._mesh_demo import export_generic_demo
dataset = load_dataset('external://$name', split='train', load_features=False)
options = { 'dataset_metadata': dataset['metadata'] }
export_generic_demo(path, options=options)
"
    )
    result=$?

    # Strip first line of the output
    if [ "$result" -eq 0 ]; then
        result_rows+="\033[92m\u2713 $name\033[0m\n"
    else
        result_rows+="\033[91m\u2717 $name\033[0m\n"
        continue
    fi

    # Store new sha
    echo $newsha > "$out.zip.sha256" || exit 1
done < <(find $inpath -iname '*.zip' -print0)

# Print results
echo ""
echo "Summary:"
echo -ne "$result_rows"
