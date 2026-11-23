#!/usr/bin/env sh
# AllyH installer for Linux and macOS.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/alexvegadev/allyh/main/scripts/install.sh | sh
#
# Optional environment variables:
#   ALLYH_VERSION      Release version/tag to install (e.g. v0.2.0 or 0.2.0). Defaults to latest.
#   ALLYH_INSTALL_DIR  Destination directory for the binary. Defaults to "$HOME/.local/bin".
#   ALLYH_REPO         GitHub repository in "owner/name" form. Defaults to "alexvegadev/allyh".
#   GH_TOKEN / GITHUB_TOKEN  Optional token for private repos or higher GitHub API rate limits.

set -eu

REPO="${ALLYH_REPO:-alexvegadev/allyh}"
BIN_NAME="allyh"
INSTALL_DIR="${ALLYH_INSTALL_DIR:-${HOME}/.local/bin}"
REQUESTED_VERSION="${ALLYH_VERSION:-}"
AUTH_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
TMP_DIR=""

say() {
    printf 'allyh-installer: %s\n' "$1"
}

err() {
    printf 'allyh-installer: error: %s\n' "$1" >&2
    exit 1
}

cleanup() {
    if [ -n "${TMP_DIR}" ] && [ -d "${TMP_DIR}" ]; then
        rm -rf "${TMP_DIR}"
    fi
}
trap cleanup EXIT INT TERM

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        err "required command '$1' was not found in PATH"
    fi
}

detect_downloader() {
    if command -v curl >/dev/null 2>&1; then
        DOWNLOADER="curl"
    elif command -v wget >/dev/null 2>&1; then
        DOWNLOADER="wget"
    else
        err "either 'curl' or 'wget' is required to download releases"
    fi
}

download_file() {
    url="$1"
    dest="$2"
    if [ "${DOWNLOADER}" = "curl" ]; then
        if [ -n "${AUTH_TOKEN}" ]; then
            curl -fsSL \
                -H "Authorization: Bearer ${AUTH_TOKEN}" \
                -H "User-Agent: allyh-installer" \
                -o "${dest}" \
                "${url}"
        else
            curl -fsSL \
                -H "User-Agent: allyh-installer" \
                -o "${dest}" \
                "${url}"
        fi
    else
        if [ -n "${AUTH_TOKEN}" ]; then
            wget -qO "${dest}" \
                --header="Authorization: Bearer ${AUTH_TOKEN}" \
                --header="User-Agent: allyh-installer" \
                "${url}"
        else
            wget -qO "${dest}" \
                --header="User-Agent: allyh-installer" \
                "${url}"
        fi
    fi
}

detect_target() {
    os="$(uname -s)"
    arch="$(uname -m)"

    case "${os}" in
        Linux)
            case "${arch}" in
                x86_64 | amd64)
                    TARGET="x86_64-unknown-linux-gnu"
                    ;;
                *)
                    err "unsupported Linux architecture '${arch}'. Build from source with: cargo install --git https://github.com/${REPO} allyh-tui"
                    ;;
            esac
            ;;
        Darwin)
            case "${arch}" in
                arm64 | aarch64)
                    TARGET="aarch64-apple-darwin"
                    ;;
                *)
                    err "unsupported macOS architecture '${arch}'. Build from source with: cargo install --git https://github.com/${REPO} allyh-tui"
                    ;;
            esac
            ;;
        *)
            err "unsupported operating system '${os}'. On Windows, use scripts/install.ps1 instead."
            ;;
    esac
}

resolve_tag() {
    if [ -n "${REQUESTED_VERSION}" ]; then
        case "${REQUESTED_VERSION}" in
            v*) TAG="${REQUESTED_VERSION}" ;;
            *) TAG="v${REQUESTED_VERSION}" ;;
        esac
        return
    fi

    # Avoid unauthenticated GitHub API rate limits on public repos by following
    # the /releases/latest redirect when no token is provided.
    if [ -z "${AUTH_TOKEN}" ] && [ "${DOWNLOADER}" = "curl" ]; then
        effective_url="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/${REPO}/releases/latest" 2>/dev/null || true)"
        case "${effective_url}" in
            */releases/tag/*)
                TAG="${effective_url##*/}"
                if [ -n "${TAG}" ]; then
                    return
                fi
                ;;
        esac
    fi

    api_json="${TMP_DIR}/latest.json"
    download_file "https://api.github.com/repos/${REPO}/releases/latest" "${api_json}" \
        || err "failed to query latest release for ${REPO} (has a release been published yet?)"

    TAG="$(sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${api_json}" | head -n 1)"
    if [ -z "${TAG}" ]; then
        err "could not determine latest release tag from GitHub API response"
    fi
}

verify_checksum() {
    archive_file="$1"
    archive_name="$2"
    sums_file="$3"

    expected="$(awk -v name="${archive_name}" '$2 == name || $2 == ("*" name) { print $1; exit }' "${sums_file}")"
    if [ -z "${expected}" ]; then
        err "no checksum entry found for ${archive_name} in SHA256SUMS"
    fi

    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "${archive_file}" | awk '{ print $1 }')"
    elif command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "${archive_file}" | awk '{ print $1 }')"
    else
        say "warning: neither sha256sum nor shasum found; skipping checksum verification"
        return
    fi

    if [ "${expected}" != "${actual}" ]; then
        err "SHA-256 checksum mismatch for ${archive_name} (expected ${expected}, got ${actual})"
    fi
    say "verified SHA-256 checksum for ${archive_name}"
}

print_path_hint() {
    case ":${PATH}:" in
        *":${INSTALL_DIR}:"*)
            return
            ;;
    esac

    say ""
    say "${INSTALL_DIR} is not currently in your PATH."
    say "Add it by appending this line to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
    printf '\n    export PATH="%s:$PATH"\n\n' "${INSTALL_DIR}"
}

main() {
    need_cmd uname
    need_cmd tar
    need_cmd mktemp
    need_cmd awk
    need_cmd sed
    detect_downloader
    detect_target

    TMP_DIR="$(mktemp -d)"
    resolve_tag

    archive_name="${BIN_NAME}-${TAG}-${TARGET}.tar.gz"
    base_url="https://github.com/${REPO}/releases/download/${TAG}"
    archive_path="${TMP_DIR}/${archive_name}"
    sums_path="${TMP_DIR}/SHA256SUMS"

    say "downloading ${BIN_NAME} ${TAG} (${TARGET})..."
    download_file "${base_url}/${archive_name}" "${archive_path}" \
        || err "failed to download ${base_url}/${archive_name}"

    if download_file "${base_url}/SHA256SUMS" "${sums_path}" 2>/dev/null; then
        verify_checksum "${archive_path}" "${archive_name}" "${sums_path}"
    else
        say "warning: SHA256SUMS asset not found for ${TAG}; skipping checksum verification"
    fi

    extract_dir="${TMP_DIR}/extract"
    mkdir -p "${extract_dir}"
    tar -xzf "${archive_path}" -C "${extract_dir}"

    extracted_bin="${extract_dir}/${BIN_NAME}"
    if [ ! -f "${extracted_bin}" ]; then
        err "expected binary '${BIN_NAME}' was not found inside ${archive_name}"
    fi

    mkdir -p "${INSTALL_DIR}"
    dest_path="${INSTALL_DIR}/${BIN_NAME}"
    if command -v install >/dev/null 2>&1; then
        install -m 755 "${extracted_bin}" "${dest_path}"
    else
        cp "${extracted_bin}" "${dest_path}"
        chmod 755 "${dest_path}"
    fi

    say "installed ${BIN_NAME} to ${dest_path}"
    if installed_ver="$("${dest_path}" --version 2>/dev/null)"; then
        say "version: ${installed_ver}"
    fi
    print_path_hint
}

main "$@"
