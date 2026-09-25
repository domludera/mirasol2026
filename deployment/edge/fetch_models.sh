#!/bin/sh
# Download the released MIRASOL ONNX graphs onto the device.
#
# Paper artifacts: the graphs benchmarked in Table 4 and evaluated in Sec. 3.5.
# The weights are NOT committed to the repository (18 files, ~302 MiB, one of
# them 109.4 MiB, above GitHub's 100 MiB blob limit), so they are published as
# GitHub Release assets and fetched by this script.
#
# POSIX sh. Uses only curl, sha256sum, tar, mkdir, cp and rm -- no `gh`, no
# Python -- so it runs on a bare Armbian image before requirements-edge.txt is
# installed. Every download is verified against SHA256SUMS.txt from the same
# release tag, and the script exits non-zero on the first mismatch rather than
# leaving a corrupt graph in place.
#
# Layout created (the layout benchmark.py and infer.py expect):
#     <dest>/<architecture>/<variant>.onnx
#     e.g. models/efficientnet_b0/model_fp32.onnx
#
# Release-asset contract: each tarball extracts to <architecture>/<variant>.onnx
# paths; the two loose .onnx assets are installed under the architecture
# directory named in the asset table below.

set -eu

REPO_SLUG="domludera/mirasol2026"
DEFAULT_TAG="v1.0.0"

GROUP="fp32"
DEST="models"
ONLY_ARCH=""
TAG="${MIRASOL_RELEASE_TAG:-$DEFAULT_TAG}"
KEEP_DOWNLOADS=0
INSTALLED=0

ALL_ARCHITECTURES="efficientnet_b0 fastvit htsat mobilenetv3 resnet18 tinyvit"

die() {
    echo "fetch_models.sh: $*" >&2
    exit 1
}

usage() {
    cat <<'USAGE'
Usage: fetch_models.sh [options]

  --group GROUP   fp32 | int8-static | int8-dynamic | fp16 | all
                  (default: fp32 -- the recommended deployable build)
  --model ARCH    install only this architecture:
                  efficientnet_b0 fastvit htsat mobilenetv3 resnet18 tinyvit
  --dest DIR      destination directory (default: models)
  --tag TAG       release tag to download from
                  (default: $MIRASOL_RELEASE_TAG, else v1.0.0)
  --keep-downloads  keep the verified archives under <dest>/.downloads
  -h, --help      show this message

Examples:
  ./fetch_models.sh
  ./fetch_models.sh --group int8-dynamic
  ./fetch_models.sh --group all --model efficientnet_b0
  ./fetch_models.sh --group fp32 --dest /srv/models --tag v1.0.0
USAGE
    exit "${1:-0}"
}

need_tool() {
    command -v "$1" >/dev/null 2>&1 || die "required tool '$1' not found on PATH"
}

# ── Argument parsing ─────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --group)  [ $# -ge 2 ] || die "--group needs a value";  GROUP="$2";     shift 2 ;;
        --model)  [ $# -ge 2 ] || die "--model needs a value";  ONLY_ARCH="$2"; shift 2 ;;
        --dest)   [ $# -ge 2 ] || die "--dest needs a value";   DEST="$2";      shift 2 ;;
        --tag)    [ $# -ge 2 ] || die "--tag needs a value";    TAG="$2";       shift 2 ;;
        --keep-downloads) KEEP_DOWNLOADS=1; shift ;;
        -h|--help) usage 0 ;;
        *) die "unknown argument '$1' (try --help)" ;;
    esac
done

case "$GROUP" in
    fp32|int8-static|int8-dynamic|fp16|all) ;;
    *) die "--group must be one of: fp32 int8-static int8-dynamic fp16 all" ;;
esac

if [ -n "$ONLY_ARCH" ]; then
    matched=0
    for candidate in $ALL_ARCHITECTURES; do
        [ "$candidate" = "$ONLY_ARCH" ] && matched=1
    done
    [ "$matched" -eq 1 ] || die "--model must be one of: $ALL_ARCHITECTURES"
fi

need_tool curl
need_tool sha256sum
need_tool tar
need_tool awk

BASE_URL="https://github.com/${REPO_SLUG}/releases/download/${TAG}"
DOWNLOAD_DIR="${DEST}/.downloads"
SUMS_FILE="${DOWNLOAD_DIR}/SHA256SUMS.txt"

# ── Asset table ──────────────────────────────────────────────────────────────
# Each line is either
#     tar|<asset name>
# or
#     file|<asset name>|<architecture>|<installed file name>
#
# The two loose .onnx assets exist so that their caveats attach to exactly one
# download: htsat/model_fp32.onnx is a latency-benchmark graph that is NOT
# accuracy-equivalent to the PyTorch model (it bicubic-interpolates 2001 mel
# frames down to 1024 and runs one forward pass), and
# resnet18/model_int8_static.onnx is the ONE static-INT8 build that does not
# collapse, so it must not inherit the degenerate bundle's warning.
assets_for_group() {
    case "$1" in
        fp32)
            echo "tar|mirasol-onnx-fp32-cnn.tar.gz"
            echo "tar|mirasol-onnx-fp32-transformer.tar.gz"
            echo "file|htsat-model_fp32.onnx|htsat|model_fp32.onnx"
            ;;
        int8-static)
            echo "tar|mirasol-onnx-int8-static-DEGENERATE-do-not-deploy.tar.gz"
            echo "file|resnet18-model_int8_static.onnx|resnet18|model_int8_static.onnx"
            ;;
        int8-dynamic)
            echo "tar|mirasol-onnx-int8-dynamic.tar.gz"
            ;;
        fp16)
            echo "tar|mirasol-onnx-fp16.tar.gz"
            ;;
        all)
            assets_for_group fp32
            assets_for_group int8-static
            assets_for_group int8-dynamic
            assets_for_group fp16
            ;;
    esac
}

# ── Download, verify, install ────────────────────────────────────────────────
download() {
    asset="$1"
    target="${DOWNLOAD_DIR}/${asset}"
    if [ -f "$target" ]; then
        echo "  cached    ${asset}"
        return 0
    fi
    echo "  fetching  ${asset}"
    curl -fL --progress-bar --retry 3 --retry-delay 2 --output "${target}.part" \
         "${BASE_URL}/${asset}" \
        || die "download failed: ${BASE_URL}/${asset}
Check that release tag '${TAG}' exists at https://github.com/${REPO_SLUG}/releases"
    mv "${target}.part" "$target"
}

verify() {
    asset="$1"
    line=$(awk -v name="$asset" '$2 == name { print; exit }' "$SUMS_FILE")
    [ -n "$line" ] || die "SHA256SUMS.txt for tag '${TAG}' has no entry for '${asset}'"
    if ! ( cd "$DOWNLOAD_DIR" && printf '%s\n' "$line" | sha256sum -c - >/dev/null ); then
        die "CHECKSUM MISMATCH for ${asset}.
The download is corrupt or the release asset changed. Delete
${DOWNLOAD_DIR}/${asset} and retry; do not use this file."
    fi
    echo "  verified  ${asset}"
}

install_tarball() {
    asset="$1"
    stage="${DOWNLOAD_DIR}/.stage"
    rm -rf "$stage"
    mkdir -p "$stage"
    tar -xzf "${DOWNLOAD_DIR}/${asset}" -C "$stage" || die "could not extract ${asset}"
    for arch_dir in "$stage"/*; do
        [ -d "$arch_dir" ] || continue
        arch=$(basename "$arch_dir")
        if [ -n "$ONLY_ARCH" ] && [ "$arch" != "$ONLY_ARCH" ]; then
            continue
        fi
        mkdir -p "${DEST}/${arch}"
        for graph in "$arch_dir"/*.onnx; do
            [ -f "$graph" ] || continue
            cp "$graph" "${DEST}/${arch}/"
            echo "  installed ${DEST}/${arch}/$(basename "$graph")"
            INSTALLED=$((INSTALLED + 1))
        done
    done
    rm -rf "$stage"
}

install_loose() {
    asset="$1"; arch="$2"; filename="$3"
    if [ -n "$ONLY_ARCH" ] && [ "$arch" != "$ONLY_ARCH" ]; then
        return 0
    fi
    mkdir -p "${DEST}/${arch}"
    cp "${DOWNLOAD_DIR}/${asset}" "${DEST}/${arch}/${filename}"
    echo "  installed ${DEST}/${arch}/${filename}"
    INSTALLED=$((INSTALLED + 1))
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "MIRASOL model fetch"
echo "  repository: https://github.com/${REPO_SLUG}"
echo "  tag:        ${TAG}"
echo "  group:      ${GROUP}"
echo "  dest:       ${DEST}"
[ -n "$ONLY_ARCH" ] && echo "  model:      ${ONLY_ARCH}"
echo

mkdir -p "$DOWNLOAD_DIR"

# SHA256SUMS.txt is the trust anchor for everything else, so it is always
# re-fetched rather than reused from a previous run.
echo "Checksums:"
rm -f "$SUMS_FILE"
download SHA256SUMS.txt

# export_manifest.json records, per graph, the source checkpoint, its held-out
# accuracy, the fold, the exporter versions and the sha256, so an on-device
# result can always be traced back to a checkpoint.
echo
echo "Provenance:"
download export_manifest.json
verify export_manifest.json
mkdir -p "$DEST"
cp "${DOWNLOAD_DIR}/export_manifest.json" "${DEST}/export_manifest.json"
echo "  installed ${DEST}/export_manifest.json"

echo
echo "Models:"
SELECTED=$(assets_for_group "$GROUP")
OUTER_IFS=$IFS
IFS='
'
for spec in $SELECTED; do
    IFS=$OUTER_IFS
    [ -n "$spec" ] || continue
    kind=$(printf '%s' "$spec" | cut -d'|' -f1)
    asset=$(printf '%s' "$spec" | cut -d'|' -f2)
    download "$asset"
    verify "$asset"
    case "$kind" in
        tar)
            install_tarball "$asset"
            ;;
        file)
            install_loose "$asset" \
                "$(printf '%s' "$spec" | cut -d'|' -f3)" \
                "$(printf '%s' "$spec" | cut -d'|' -f4)"
            ;;
        *)
            die "malformed asset spec: $spec"
            ;;
    esac
    IFS='
'
done
IFS=$OUTER_IFS

if [ "$INSTALLED" -eq 0 ]; then
    die "nothing was installed: group '${GROUP}' contains no graph for '${ONLY_ARCH}'"
fi

if [ "$KEEP_DOWNLOADS" -eq 0 ]; then
    rm -rf "$DOWNLOAD_DIR"
else
    echo
    echo "Kept verified downloads in ${DOWNLOAD_DIR}"
fi

echo
echo "Installed ${INSTALLED} graph(s) under ${DEST}/."

case "$GROUP" in
    int8-static|all)
        echo
        echo "**********************************************************************"
        echo "WARNING - DEGENERATE STATIC-INT8 GRAPHS"
        echo "efficientnet_b0, tinyvit and mobilenetv3 model_int8_static.onnx are"
        echo "CONSTANT classifiers (always 'Absent'); fastvit's always answers"
        echo "'Unknown'. They are published only to reproduce Sec. 3.5(3) and MUST"
        echo "NOT be used as screeners - an always-Absent screener never refers"
        echo "anyone. resnet18/model_int8_static.onnx is the one static-INT8 build"
        echo "that survives quantization. The recommended deployable build is FP32."
        echo "**********************************************************************"
        ;;
esac

echo
echo "Next:  python3 benchmark.py --models-dir ${DEST}"
echo "       python3 infer.py --model ${DEST}/<arch>/model_fp32.onnx --audio rec.wav"
echo "Research prototype - NOT A MEDICAL DEVICE."
