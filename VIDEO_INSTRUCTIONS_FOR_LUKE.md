# VIDEO INSTRUCTIONS FOR LUKE

**Goal**: Create 4-minute video for Presidential AI Challenge Track II
**Deadline**: Tonight
**Time needed**: 2-3 hours

---

## STEP 1: TEST THE DEMO (5 minutes)

Open terminal and run:
```bash
cd /Users/lvxcas/ATLAS-Presidential-AI-Challenge
python3 run_live_demo_with_graph.py
```

**You should see**:
1. Console output showing real EURUSD data being analyzed
2. A graph window pop up showing risk scores over 50 days

If this works, you're ready to record.

---

## STEP 2: RECORD VOICEOVER (30-45 minutes)

### Option A: Use GarageBand (Mac - Easiest)

1. Open GarageBand (Cmd+Space → type "GarageBand")
2. Create New Project → "Voice"
3. Hit the red record button
4. Open file: `submission/VOICEOVER_SCRIPT_FINAL.txt`
5. Read the script clearly (don't rush - take 3:50)
6. Stop recording
7. Share → Export Song to Disk → MP3
8. Save as: `atlas_voiceover.mp3`

### Option B: Use Audacity (Free for Windows/Mac/Linux)

1. Download Audacity: https://www.audacityteam.org/download/
2. Open Audacity
3. Hit red record button
4. Read the script from `submission/VOICEOVER_SCRIPT_FINAL.txt`
5. Stop
6. File → Export → Export as MP3
7. Save as: `atlas_voiceover.mp3`

**Tips**:
- Speak clearly, slightly slower than normal
- Pause for 3 seconds where script says [PAUSE]
- Don't worry about perfection - good is good enough
- Total duration should be 3:45-3:55

---

## STEP 3: RECORD SCREEN (10 minutes)

### Mac (QuickTime - Built-in)

1. Open QuickTime Player
2. File → New Screen Recording
3. Click the red record button
4. Click anywhere on screen to start recording
5. Open Terminal
6. Run: `python3 run_live_demo_with_graph.py`
7. Wait for graph to appear
8. Let graph display for 10 seconds
9. Stop recording (menu bar stop button)
10. File → Save → `atlas_demo_screen.mov`

### Windows (OBS Studio - Free)

1. Download OBS: https://obsproject.com/download
2. Install and open OBS
3. Sources → Add → Display Capture
4. Click "Start Recording"
5. Open terminal/command prompt
6. Run: `python3 run_live_demo_with_graph.py`
7. Wait for graph to appear
8. Let it display for 10 seconds
9. Click "Stop Recording"
10. Video saves automatically to Videos folder

---

## STEP 4: EDIT VIDEO (60-90 minutes)

### Mac (iMovie - Built-in, Easiest)

1. Open iMovie
2. Create New Project → Movie
3. Import Media:
   - Drag `atlas_demo_screen.mov` to timeline
   - Drag `atlas_voiceover.mp3` to audio track
4. Add Title at start (0:00):
   - Click "Titles" → Choose "Centered"
   - Type: "ATLAS: Educational AI for Financial Literacy"
   - Duration: 3 seconds
5. Add Title at end (3:50):
   - Click "Titles" → Choose "Centered"
   - Type: "github.com/LVXCAS/ATLAS-Presidential-AI-Challenge"
   - Duration: 5 seconds
6. Trim video to match voiceover length (~3:50)
7. File → Share → File
   - Format: High Quality (1080p)
   - Save as: `ATLAS_Track2_Video.mp4`

### Windows/Linux (DaVinci Resolve - Free, Professional)

1. Download DaVinci Resolve: https://www.blackmagicdesign.com/products/davinciresolve/
2. Install and open
3. Create New Project
4. Import:
   - Drag `atlas_demo_screen.mov` to video track
   - Drag `atlas_voiceover.mp3` to audio track
5. Add text overlays:
   - Start: "ATLAS: Educational AI for Financial Literacy"
   - End: "github.com/LVXCAS/ATLAS-Presidential-AI-Challenge"
6. Deliver → Custom Export
   - Format: MP4
   - Resolution: 1920x1080
   - Frame Rate: 30fps
   - Click "Add to Render Queue" → "Start Render"
7. Save as: `ATLAS_Track2_Video.mp4`

---

## STEP 5: UPLOAD TO YOUTUBE (15 minutes)

1. Go to: https://www.youtube.com/upload
2. Sign in with Google account
3. Click "SELECT FILES"
4. Choose `ATLAS_Track2_Video.mp4`
5. Fill in details:
   - **Title**: "ATLAS Track II: Educational AI for Financial Literacy"
   - **Description**:
     ```
     ATLAS is a multi-agent AI system for K-12 financial literacy.

     - 13 specialized AI agents
     - 100% improvement over baseline
     - Transparent, explainable decisions
     - Addresses Presidential priorities

     Learn more: https://github.com/LVXCAS/ATLAS-Presidential-AI-Challenge
     ```
   - **Visibility**: UNLISTED (very important!)
6. Click "PUBLISH"
7. Wait 5-10 minutes for processing
8. **COPY THE VIDEO URL**: `https://www.youtube.com/watch?v=XXXXXXX`

---

## STEP 6: CREATE PDF SUBMISSION (5 minutes)

1. Open Google Docs: https://docs.google.com
2. Create new document
3. Type:

```
ATLAS TRACK II SUBMISSION
Presidential AI Challenge

═══════════════════════════════════════════
VIDEO DEMONSTRATION (4 Minutes)
═══════════════════════════════════════════

https://www.youtube.com/watch?v=XXXXXXX

[Paste your actual YouTube URL above]

Right-click the link to open in new tab.

═══════════════════════════════════════════
```

4. File → Download → PDF Document (.pdf)
5. Save as: `ATLAS_Track2_Submission.pdf`

---

## CHECKLIST BEFORE SUBMITTING

- [ ] Video runs for 3:45-4:00 (not longer)
- [ ] Audio is clear and understandable
- [ ] Screen recording shows demo + graph
- [ ] YouTube link works (test in incognito mode)
- [ ] Video is set to UNLISTED
- [ ] PDF has working YouTube link
- [ ] All files saved

---

## TIMELINE

| Task | Time |
|------|------|
| Test demo | 5 min |
| Record voiceover | 30-45 min |
| Record screen | 10 min |
| Edit video | 60-90 min |
| Upload to YouTube | 15 min |
| Create PDF | 5 min |
| **TOTAL** | **2.5-3 hours** |

---

## TROUBLESHOOTING

**Demo won't run?**
```bash
# Make sure you're in the right directory
cd /Users/lvxcas/ATLAS-Presidential-AI-Challenge

# Try running with full path
python3 run_live_demo_with_graph.py
```

**No microphone in GarageBand?**
- System Preferences → Sound → Input → Select microphone

**Video too long?**
- Cut out some pauses in the editor
- Trim start/end

**YouTube processing stuck?**
- Try uploading to Vimeo instead: https://vimeo.com/upload

---

## YOU'VE GOT THIS

1. Read the voiceover script once (practice)
2. Record audio (30 min)
3. Record screen demo (10 min)
4. Edit together (60 min)
5. Upload (15 min)

**Total: 2-3 hours. You'll be done before midnight.**

Any questions? Everything you need is in this repo.

START NOW 🚀
