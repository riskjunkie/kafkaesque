#!/bin/bash
cd "$(dirname "$0")"
echo "Starting the \$KAFKA game server — leave this window open while you play."
echo "Close this window (or press Ctrl+C) to stop it."
(sleep 1; open "http://localhost:8935/index.html") &
python3 -m http.server 8935
