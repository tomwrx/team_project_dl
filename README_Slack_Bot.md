# Privacy Masker — Slack Integration

A Slack bot that automatically detects and blurs sensitive information in images before they are shared in channels. The unblurred image never appears publicly, and every action requires explicit approval.

---


## Project Overview

Privacy Masker bot could be used to protect companies from accidental leakage of personally identifiable information (PII) in Slack channels. 
Slack has become one of the most widely used workplace communication platforms. Employees routinely share screenshots to report bugs, discuss customers, or coordinate work — and these screenshots often contain sensitive information in the background that the sender never intended to expose. Privacy masker helps prevent that by intercepting images sent privately via DM, scanning them using a fine-tuned vision model, and only posting the blurred version to the intended channel — the unblurred image never appears publicly.

---

## How It Works

Privacy Masker is built around a private DM workflow. Instead of uploading images directly to a channel, users send them to the Privacy Masker bot in a direct message. The bot processes the image, shows a blurred preview with the detected regions listed, and waits for the user to decide what to do. Only after explicit approval does the blurred image get posted to the intended channel — the original, unblurred image is never seen by anyone else.

The system is built on three components that work together:

**Slack Bot** — built with Slack Bolt for Python. Listens for file uploads in DMs, handles user interactions (channel selection, edit/approve/discard), and posts the final blurred image to the chosen channel with the sender's name and original message text preserved.

**FastAPI Backend** — a REST API running on Google Colab with a GPU. Receives images as base64, runs inference to detect sensitive regions, applies Gaussian blur to each detected region using PIL, and returns the blurred image. Also manages editor sessions for the manual box editing feature.

**Web Editor** — a browser-based canvas tool served from the same backend. When the user wants to add additional blur regions beyond what the model detected, the bot sends them a private link to the editor. The user draws boxes on the original image by clicking and dragging, then confirms. The backend applies the additional blur and the bot posts the result.

The backend is exposed via ngrok — a secure tunnel that gives it a private HTTPS URL. All requests to the backend require a secret API key.

---

## File Structure

```
project/
│
├── app.py              # FastAPI backend — inference, blur, session management
├── slack_bot.py        # Slack Bolt bot — DM flow, Accept/Edit/Discard
└── addin/
    └── editor.html     # Web-based box editor (draw additional blur regions)
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          USER                                 │
│                                                               │
│   Sends image to Privacy Masker bot via private DM            │
│   (image never posted to any channel at this stage)           │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     SLACK BOT                                 │
│                                                               │
│  1. Receives image via DM                                     │
│  2. Downloads image from Slack                                │
│  3. Sends to backend for inference                            │
│  4. Uploads blurred preview back to DM                        │
│  5. Shows detected regions + action buttons:                  │
│                                                               │
│     ┌──────────────┬─────────────┬────────────┐              │
│     │ Post to      │ ✏️ Edit      │ ✕ Discard  │              │
│     │ channel ▼   │ boxes       │            │              │
│     └──────┬───────┴──────┬──────┴────────────┘              │
│            │              │                                   │
│            │              ▼                                   │
│            │    User opens web editor                         │
│            │    Draws additional boxes                        │
│            │    Confirms → backend re-blurs                   │
│            │              │                                   │
│            └──────────────┘                                   │
│                     │                                         │
│                     ▼                                         │
│      Posts BLURRED image to chosen channel                    │
│      "Shared by @username\nmessage text"                      │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND (Colab + ngrok)             │
│                                                               │
│  POST /process-image                                          │
│    → Model inference → bounding boxes                         │
│    → Apply Gaussian blur (PIL)                                │
│    → Return blurred image + detected boxes                    │
│                                                               │
│  POST /create-session                                         │
│    → Store original image for the web editor                  │
│                                                               │
│  GET  /session/{id}                                           │
│    → Return image data to editor page                         │
│                                                               │
│  POST /confirm-edit                                           │
│    → Apply user-drawn boxes → re-blur → store result          │
│                                                               │
│  GET  /edit-result/{id}                                       │
│    → Bot polls this until editor confirms                     │
└──────────────────────────────────────────────────────────────┘
```

---

## User Flow

```
User DMs bot with image
        │
        ▼
Bot scans image
        │
        ├── No sensitive info → "Safe to share!" ✅
        │
        └── Sensitive info detected
                │
                ▼
        Blurred preview shown in DM
        Detected regions listed
                │
        ┌───────┴────────────────────┐
        │                            │
        ▼                            ▼
Post to channel              Edit boxes
(select channel)             (open editor)
        │                            │
        ▼                            │
Blurred image posted         Draw additional boxes
"Shared by @username"        Confirm in editor
                                     │
                             Select channel
                                     │
                             Blurred image posted
                             "Shared by @username"
```

## Slack App Permissions Required

| Scope | Purpose |
|---|---|
| `chat:write` | Send messages |
| `files:read` | Download uploaded images |
| `files:write` | Upload blurred images |
| `im:history` | Read DM messages |
| `im:read` | Check if channel is a DM |
| `im:write` | Send DM responses |
| `channels:read` | List channels for selector |
| `groups:read` | List private channels |
| `mpim:read` | List group DMs |
| `users:read` | Get sender's real name |

---

## Security

- All backend API calls require an `x-api-key` secret header — requests without it are rejected with 403.
- The backend URL is a private ngrok tunnel that changes every session — it is never published anywhere.
- Unblurred images are only ever seen by the sending user in their private DM with the bot.
- Images are processed in memory — nothing is stored permanently on disk.
- The bot only processes images sent in DMs — images uploaded directly to channels are ignored.
- If ran on local company servers, images with user specified bounding boxes could be stored for further model training for specific company needs. 
