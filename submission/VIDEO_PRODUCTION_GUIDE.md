# ATLAS Video Production Guide

**Quick Start**: Follow the 5-day production timeline to create a professional 4-minute video submission.

---

## STEP 1: PREPARE YOUR DEMO (Day 1 - Morning)

### Pre-Flight Checklist

Before recording anything, make sure ATLAS runs cleanly:

```bash
# Test the demo command
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --window regime-shift --data-source synthetic

# Verify unit tests pass
python3 -m unittest Agents/ATLAS_HYBRID/tests/test_agents.py

# Check website is live
open https://lvxcas.github.io/ATLAS-Presidential-AI-Challenge
```

### Demo Sequences to Capture

#### Sequence 1: Terminal Output (2-3 minutes)
Run this command and let it complete fully:
```bash
python3 Agents/ATLAS_HYBRID/quant_team_demo.py --window regime-shift --data-source synthetic
```

**What to capture**:
- Initial agent roster display (shows all 13 agents)
- Scenario summary (stress detection statistics)
- One complete agent breakdown (Step 100 is good)
- Final posture output

**Terminal setup for recording**:
- Font size: 18pt minimum (readable on video)
- Background: Dark theme (easier on eyes)
- Width: 120 characters or wider
- Record at 1080p resolution

#### Sequence 2: Website Tour (1-2 minutes)

Navigate to: https://lvxcas.github.io/ATLAS-Presidential-AI-Challenge

**Screens to capture**:
1. Homepage/dashboard
2. "Results" section (shows evaluation metrics)
3. "About" section (team narrative)
4. One interactive demo (if available)

**Recording tip**: Move slowly between sections so viewers can read

#### Sequence 3: Comparison Chart (30 seconds)

Create a simple visual showing:
```
Baseline System:  9.27% false confidence in stress
ATLAS System:     0.00% false confidence in stress
Improvement:      100%
```

Can be:
- Terminal output
- Screenshot
- Simple graphic in PowerPoint/Keynote
- Just mention verbally with text overlay

---

## STEP 2: RECORD VOICEOVER (Day 1 - Afternoon)

### Audio Setup

**Microphone Quality Matters**:
- Use your laptop microphone (acceptable)
- Use USB headset (better)
- Use desk microphone (best)
- Avoid built-in laptop mic if you have alternatives

**Recording Software**:

**macOS**:
```bash
# Use GarageBand (free, comes with Mac)
# Record audio while looking at VIDEO_SCRIPT_4MIN.md
```

**Windows**:
```bash
# Use Audacity (free)
# Download: https://www.audacityteam.org/
```

**Linux**:
```bash
# Use Audacity or Audiotool
```

### Voiceover Recording Tips

1. **Read the script naturally** - Practice 2-3 times before recording
2. **Speak clearly** - Enunciate, especially technical terms
3. **Use good pacing** - Slightly slower than normal conversation
4. **Take breaks** - Record in 30-second chunks, don't do it all at once
5. **Pause at transitions** - Natural 1-2 second pauses between sections
6. **Room noise** - Record in a quiet room (empty room with carpet is ideal)
7. **Multiple takes** - You can always re-record sections

### Voiceover Sections

Break recording into 6 sections (one per SECTION in script):

| Section | Duration | File |
|---------|----------|------|
| 1. Problem | 0:45 | vo_01_problem.mp3 |
| 2. Solution | 0:45 | vo_02_solution.mp3 |
| 3. Demo Narration | 1:30 | vo_03_demo.mp3 |
| 4. Differentiation | 0:35 | vo_04_diff.mp3 |
| 5. Impact | 0:20 | vo_05_impact.mp3 |
| 6. Closing | 0:05 | vo_06_closing.mp3 |

---

## STEP 3: EDIT VIDEO (Day 2)

### Video Editing Software

**Free Options**:
- **DaVinci Resolve** (professional, free tier)
  - Download: https://www.blackmagicdesign.com/products/davinciresolve/
  - Works: Mac/Windows/Linux
- **OpenShot** (simpler, free)
  - Download: https://www.openshot.org/
- **iMovie** (macOS only, free if you own a Mac)
- **Windows Photos** (Windows only, basic)

**Recommended for beginners**: iMovie (Mac) or DaVinci Resolve (all platforms)

### Editing Checklist

1. **Import Assets**
   - Voiceover audio files
   - Screen recordings (demo terminal, website)
   - Comparison chart/graphic

2. **Create Timeline**
   - Voiceover on primary audio track
   - Screen recordings synced to voiceover
   - Transitions between sections (fade, cut)

3. **Add Text Overlays**
   - Agent names (when discussing agents)
   - Key metrics (9.27% vs. 0.00%)
   - URLs (at beginning and end)
   - Agent weights (optional, if showing config)

4. **Adjust Timing**
   - Each section should match voiceover duration
   - Demo footage should show at normal speed (don't speed up)
   - Allow 1-2 second pause for metric comparisons

5. **Color & Levels**
   - Ensure all text is readable (white text on dark background is good)
   - No audio peaks (watch waveform for clipping)
   - Consistent brightness across clips

### Timeline Structure

```
0:00-0:45    [VO: Problem] [Visual: Title slide or office setting]
0:45-1:30    [VO: Solution] [Visual: ATLAS website/13 agents graphic]
1:30-3:00    [VO: Demo] [Visual: Terminal output + agent breakdown]
3:00-3:35    [VO: Differentiation] [Visual: Comparison chart]
3:35-3:55    [VO: Impact] [Visual: Impact metrics graphic]
3:55-4:00    [VO: Closing] [Visual: ATLAS logo + GitHub URL]
```

---

## STEP 4: EXPORT & UPLOAD (Day 3-4)

### Export Settings

**Video Format**: MP4 (H.264 codec)
- Resolution: 1920x1080 (1080p)
- Frame rate: 30fps
- Bitrate: 5000-8000 kbps (good quality, reasonable file size)
- Audio: 128 kbps, 44.1 kHz

**File Size Expected**: 50-200 MB (depending on quality)

**Export Time**: 5-15 minutes (depending on your computer)

### Upload to Hosting Platform

**Option 1: YouTube** (Recommended)
1. Go to: https://www.youtube.com/upload
2. Sign in (or create account)
3. Select "Create" → "Upload video"
4. Upload your MP4 file
5. Set visibility to "Unlisted" (accessible with link, not searchable)
6. Wait for processing (5-10 minutes for 1080p)
7. Copy video URL: `https://www.youtube.com/watch?v=xxxxxxx`

**Option 2: Vimeo** (Professional alternative)
1. Go to: https://vimeo.com/upload
2. Upload your MP4 file
3. Set privacy to "Only people with the private link"
4. Copy video URL: `https://vimeo.com/xxxxxxx`

**Option 3: Google Drive** (Quick backup option)
1. Upload to Google Drive
2. Right-click → "Share"
3. Set to "Viewer" access, anyone with link
4. Copy sharing link

### Video Quality Check (Before Submitting)

- [ ] Video plays smoothly (no stuttering)
- [ ] Audio is clear (no distortion)
- [ ] Text overlays are readable
- [ ] Timing matches script (3:45-4:00)
- [ ] URL link in description works
- [ ] Link is accessible (test in incognito mode)

---

## STEP 5: CREATE PDF SUBMISSION (Day 5)

### PDF Structure

Your PDF submission should include:

1. **Title Page**
   - Project: ATLAS AI Quant Team
   - Track: II (Educational AI)
   - Date: [submission date]

2. **Executive Summary** (1-2 pages)
   - Problem statement
   - Solution overview
   - Key innovation

3. **System Description** (2-3 pages)
   - Architecture diagram (13 agents)
   - How the system works
   - Key features

4. **Video Link** (1 page)
   ```
   DEMONSTRATION VIDEO (4 MINUTES):
   [CLICKABLE LINK]

   https://www.youtube.com/watch?v=xxxxxxx

   [Or your Vimeo/Drive link]

   Right-click to open in new tab
   ```

5. **Results & Impact** (1-2 pages)
   - 100% performance improvement
   - Learning outcomes framework
   - Administration alignment

6. **Team & Documentation** (1 page)
   - Links to key documents:
     - GitHub repo
     - Website
     - Design rationale
     - Safety ethics statement

### PDF Tools

**macOS/Windows/Linux**:
- Google Docs → Export as PDF
- Microsoft Word → Save as PDF
- Pages → Export as PDF

**File Format for Submission**:
- Name: `ATLAS_Track2_Submission.pdf`
- Size: < 50 MB
- Include video link with clear instructions

---

## QUICK REFERENCE: FILE CHECKLIST

Before submitting, you should have:

```
submission/
├── VIDEO_SCRIPT_4MIN.md (this script)
├── VIDEO_PRODUCTION_GUIDE.md (this guide)
├── ATLAS_Track2_Demo_4min.mp4 (final video file)
├── ATLAS_Track2_Submission.pdf (PDF with video link)
├── ADMINISTRATION_RELEVANCE.md
├── ML_INTEGRATION_TECHNICAL_BRIEF.md
├── JUDGES_BRIEFING.md
├── EDUCATIONAL_VALIDATION_FRAMEWORK.md
└── [other existing documents]
```

Video Link in Submission:
```
PRIMARY: https://www.youtube.com/watch?v=xxxxxxx
BACKUP: https://vimeo.com/xxxxxxx
```

---

## TROUBLESHOOTING

### Video Quality Issues

**Problem**: Video is blurry or pixelated
- **Solution**: Re-record with higher resolution (1080p minimum)

**Problem**: Audio is hard to hear
- **Solution**: Record voiceover in quieter room, increase mic gain

**Problem**: Video is too dark
- **Solution**: Adjust monitor brightness before recording, increase video brightness in editor

**Problem**: Demo output isn't visible
- **Solution**: Increase terminal font size to 20pt+, use high contrast terminal theme

### Timing Issues

**Problem**: Video is shorter than 3:45
- **Solution**: Add pauses between sections, slow down voiceover slightly

**Problem**: Video exceeds 4:00
- **Solution**: Cut filler phrases, remove redundant explanations

### Upload Issues

**Problem**: YouTube video stays "processing" for 30+ minutes
- **Solution**: Try uploading to Vimeo instead (usually faster)

**Problem**: Link doesn't work
- **Solution**: Test link in incognito/private browser mode, check sharing settings

---

## OPTIONAL: Make It Even Better

### Enhancement Ideas (if you have time)

1. **Add background music** (very low volume during transitions)
   - Music: YouTube Audio Library (free)
   - Volume: -40dB (barely audible)

2. **Create animated graphics** for:
   - 13 agents clustering into 4 pillars
   - Risk posture calculation
   - Score improvement (9.27% vs. 0%)

3. **Add captions** (YouTube auto-generates, but you can improve them)
   - Especially helpful for accessibility
   - Helps if audio isn't perfect

4. **Include a brief team introduction** (30 sec at start)
   - "Hi, we're the ATLAS team..."
   - Creates personal connection

5. **Show student testimonial** (optional)
   - "I now understand why the system recommended caution..."

---

## FINAL SUBMISSION FORMAT

### For Presidential AI Challenge PDF:

**Track II Video Requirement**:
> "Include a link to a 4-minute maximum video demonstrating the technology solution."

**What to Include in PDF**:

```
═══════════════════════════════════════════════════════
TRACK II VIDEO DEMONSTRATION (4 minutes)
═══════════════════════════════════════════════════════

https://www.youtube.com/watch?v=xxxxxxx

[Right-click to open link in new tab]

Video Contents:
✓ Problem statement and context
✓ System architecture explanation
✓ Live demonstration of ATLAS running
✓ Agent reasoning walkthrough
✓ Performance results (100% improvement)
✓ Use cases and impact

Backup link: https://vimeo.com/xxxxxxx
═══════════════════════════════════════════════════════
```

---

## SUCCESS CRITERIA

Your video submission is ready when:

- [ ] Video plays smoothly with no errors
- [ ] Audio is clear and professional
- [ ] Timing is 3:45-4:00 (not longer)
- [ ] All visuals are legible (font size ≥ 18pt)
- [ ] Voiceover follows the script pattern ("Our solution is...")
- [ ] Demo sequences are visible and understandable
- [ ] Link is accessible with no password
- [ ] Video explains both the problem AND solution
- [ ] Key metrics are highlighted (0% vs 9.27%)
- [ ] Team confident presenting the content

You're ready to submit! 🚀
