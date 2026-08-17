# Demo Script and Recording Commands

## Demo Script (3–5 minutes)

1. Intro (10–15s)
   - "Hello, I'm M. Ali Sanwal. This demo covers the Medicine Recommendation System prototype..."

2. Quick Architecture (20–30s)
   - Briefly describe data → models → recommender → Streamlit UI

3. Show Dataset & Medicine DB (30s)
   - Open `Medicine Database`, search a medicine, expand an entry

4. Run a Recommendation (60–90s)
   - Fill patient form with symptoms, age, gender, severity
   - Click 'Get Recommendations' and explain top recommendations and confidence

5. Analytics Dashboard (30s)
   - Show confusion matrix and mention evaluation metrics (accuracy, F1)

6. Wrap-up (10–15s)
   - Mention limitations and next steps, provide GitHub link and live demo URL

## Recording with `ffmpeg` (quick, command-line)
Capture the whole screen (Linux/macOS):
```bash
# example: record full screen at 1280x720, 30fps
ffmpeg -f x11grab -s 1280x720 -i :0.0 -r 30 -preset ultrafast medirec_demo.mp4
```

Capture a single window (Linux, find window by name):
```bash
WINDOW_ID=$(xdotool search --name "Streamlit")
ffmpeg -f x11grab -i :0.0+0,0 -window_id $WINDOW_ID -r 30 -preset ultrafast medirec_demo.mp4
```

Windows (PowerShell) — using `ffmpeg` and capturing desktop:
```powershell
ffmpeg -f gdigrab -framerate 30 -i desktop -preset ultrafast medirec_demo.mp4
```

Notes
- `ffmpeg` is lightweight but does not provide editing UI — use OBS Studio for richer recordings.
- Record audio separately or enable microphone capture in `ffmpeg`/OBS.
- Keep demo concise and narrate the key points.
