#!/usr/bin/env bash

# Self-locate the manifest from THIS script's own directory, so the package works under any folder
# name (versioned or not) with no hardcoded /hive/miners/custom/keryx-miner path and no symlink.
# No cd / no exit here: HiveOS may source this file.
__MD="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" && pwd)"
. "$__MD/h-manifest.conf"

conf=""
conf+=" -s $CUSTOM_URL --mining-address $CUSTOM_TEMPLATE"

[[ ! -z $CUSTOM_USER_CONFIG ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf"
echo "$conf" > $CUSTOM_CONFIG_FILENAME
