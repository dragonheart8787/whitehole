#!/bin/sh
set -e
# Allow: whitesearch <cmd> | pytest | bash | python -m whitesearch
case "$1" in
  whitesearch|pytest|bash|python)
    exec "$@"
    ;;
  ""|--help|-h)
    exec whitesearch --help
    ;;
  *)
    exec whitesearch "$@"
    ;;
esac
