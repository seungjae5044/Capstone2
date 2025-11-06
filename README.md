## Execution
'''
1. ollama serve
2. python whisper_web_ui.py
3. cd src_F
4. npm run build
5. cd ..
6. python -m py_compile whisper_web_ui.py
7. cd src_F
8. npm run dev
'''

## System Audio (macOS)
- Install a loopback input device (e.g., BlackHole 2ch).
- In Audio MIDI Setup, create a Multi‑Output Device (Built‑in Output + BlackHole) so you can hear audio while capturing.
- In the Setup screen (frontend), choose Audio Source = "system" and optionally select the BlackHole device; otherwise the server auto‑picks common loopback devices.

API helpers:
- List devices: `GET /audio/devices`
- Start: `POST /start` with `{ topic, participants, source: "mic"|"system", device_index? }`
