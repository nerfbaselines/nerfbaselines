#!/bin/bash
POSITIONAL_ARGS=()
pre_commit=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --pre-commit)
            pre_commit=1
            shift
            ;;
        --repo)
            outpath="$2"
            shift # past argument
            shift # past value
            ;;
        -*|--*)
            echo "Unknown option $1"
            echo "Usage: $0 <in-path> --repo <repo-path> [--pre-commit]"
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
    echo "Usage: $0 <in-path> --repo <repo-path> [--pre-commit]"
    exit 1
fi
if [ -z "$outpath" ]; then
    echo "Usage: $0 <in-path> --repo <repo-path> [--pre-commit]"
    exit 1
fi

echo "Using repository path: $outpath"
echo "Using input path: $inpath"


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
    if [[ "$name" == *"/output" ]]; then
        name=$(dirname $name)
        if [ -e "$inpath/$name.zip" ]; then
            continue
        fi
    fi
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

    # Copy the file if it doesn't exist
    if [ ! -e "$out" ] && [ "$inpath" != "$outpath" ]; then
        echo "Copying $name"
        cp -p "$file" "$out.zip"
    fi

    # Extract the results.json file from the zip
    unzip -p $file results.json > $out.json || exit 1
    echo "Extracted $name.json"

    # Store new sha
    echo $newsha > "$out.zip.sha256" || exit 1
done
