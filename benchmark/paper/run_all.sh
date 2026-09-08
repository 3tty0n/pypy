#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
exec bash "$HERE/bench.sh" all "$@"
