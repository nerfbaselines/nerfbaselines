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
    out="$outpath/$name"
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
        if [[ "$name" == "phototourism/"* ]]; then
            # For phototourism, we generate demos for different appearances
            if [[ "$name" == *"/trevi-fountain" ]]; then
                embedding_ids=(7 5 174)
            elif [[ "$name" == *"/sacre-coeur" ]]; then
                embedding_ids=(7 14 29)
            elif [[ "$name" == *"/brandenburg-gate" ]]; then
                embedding_ids=(0 1 9)
            else
                echo "Unknown phototourism scene: $name"
                exit 1
            fi
            tmpdir=$(mktemp -d)
            trap "rm -rf $tmpdir" EXIT
            nerfbaselines export-demo --checkpoint $file/checkpoint --data external://$name --output ${out}_demo --train-embedding ${embedding_ids[0]}
            nerfbaselines export-demo --checkpoint $file/checkpoint --data external://$name --output ${tmpdir}/e2 --train-embedding ${embedding_ids[1]}
            mv ${tmpdir}/e2/params.json ${out}_demo/params_e2.json
            mv ${tmpdir}/e2/scene.ksplat ${out}_demo/scene_e2.ksplat
            python -c 'import json;p=json.load(open("'$out'_demo/params_e2.json"));p["sceneUri"]="scene_e2.ksplat";json.dump(p,open("'$out'_demo/params_e2.json","w"))'
            nerfbaselines export-demo --checkpoint $file/checkpoint --data external://$name --output ${tmpdir}/e3 --train-embedding ${embedding_ids[2]}
            mv ${tmpdir}/e3/params.json ${out}_demo/params_e3.json
            mv ${tmpdir}/e3/scene.ksplat ${out}_demo/scene_e3.ksplat
            python -c 'import json;p=json.load(open("'$out'_demo/params_e3.json"));p["sceneUri"]="scene_e3.ksplat";json.dump(p,open("'$out'_demo/params_e3.json","w"))'
            python -c 'import json;p=json.load(open("'$out'_demo/params.json"));p["links"]={"Appearance 1": "params.json", "Appearance 2": "params_e2.json", "Appearance 3": "params_e3.json"};json.dump(p,open("'$out'_demo/params.json","w"))'
        else
            nerfbaselines export-demo --checkpoint $file/checkpoint --data external://$name --output ${out}_demo
        fi
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
