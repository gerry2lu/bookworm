#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib
export PATH=/usr/local/bin:/usr/bin:/bin
env > /home/gerrylu/bookworm-env.log
groups > /home/gerrylu/bookworm-groups.log
ls -l /dev/snd/ > /home/gerrylu/bookworm-dev-snd.log
exec /usr/bin/python3 /home/gerrylu/voice-ai/BookWorm.py
