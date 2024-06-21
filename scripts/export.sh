#!/bin/bash
export inpath="$PREFIX/baselines/nerfbaselines"
export outpath="$inpath"

POSITIONAL_ARGS=()
pre_commit=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --pre-commit)
            pre_commit=1
            shift
            ;;
        -*|--*)
            echo "Unknown option $1"
            echo "Usage: $0 [path] [--pre-commit]"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1") # save positional arg
            shift # past argument
            ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters
if [ "$pre_commit" -eq 1 ]; then
    echo "Using git to determine changed files"
else
    echo "Using sha to determine changed files"
fi

if [ $# -eq 1 ]; then
    export inpath="$(realpath $1)"
elif [ $# -gt 1 ]; then
    echo "Usage: $0 [path] [--pre-commit]"
    exit 1
fi

echo "Using repository path: $inpath"


generate-ksplat () {
    method="$1"
    zip="$2"
    ksplat="$3"
}

# Process all results in the baselines directory
find $inpath -iname '*.zip' -print0  | while IFS= read -r -d '' file; do
    # Base path to store output without the extension,
    # relative to the repository
    # e.g. blender/lego
    name=${file#"$inpath/"};name=${name%.zip}
    method=${name%%/*}

    # Absolute output path without the extension
    out="$outpath/$name"
    mkdir -p "$(dirname $out)"

    # Check if there exists sha files for the results
    echo "Processing $name"
    oldsha="$([ -e "$out.zip.sha256" ] && cat "$out.zip.sha256")"
    newsha="$(sha256sum $file | cut -d' ' -f1)"
    
    if [ "$oldsha" == "$newsha" ]; then
        echo "No change in $name, skipping..."
        continue
    fi

    # Extract the results.json file from the zip
    unzip -p $file results.json > $out.json || exit 1
    echo "Extracted $name.json"

    # For each splat result, generate the ksplat file
    if [ "$method" == "gaussian-splatting" ] || [ "$method" == "mip-splatting" ]; then
        $(dirname $0)/make_ksplat.sh $method $out.zip $out.ksplat || exit 1
    fi

    # Store new sha
    echo $newsha > "$out.zip.sha256" || exit 1
done
