# Team Documentation & Sync Guide

Hey team! I’ve set up this project documentation to live in **Obsidian** and sync automatically via **GitHub**. This means we don't have to send files back and forth; as soon as I update a roadmap or a tool list, it will show up on your computer within minutes.

Follow these steps to get synced up with me.

---

## Initial Setup (Do this once)

### 1. Clone the Repository

Open your Terminal and run: `git clone [INSERT_YOUR_GITHUB_REPO_URL_HERE]`

### 2. Open as a Vault

1. Open **Obsidian**.
    
2. Click the **Vault Switcher** icon (bottom left).
    
3. Click **Open folder as vault**.
    
4. Select the project folder you just cloned.
    

### 3. Install the Sync Plugin

1. Go to **Settings** > **Community Plugins** and click **Browse**.
    
2. Search for **Git** (by Vinzent) and click **Install**, then **Enable**.
    

---

## Required Settings

To make sure we stay in sync without manual effort, please configure the plugin exactly like this:

### Automatic Syncing

- **Auto commit-and-sync interval:** Set to `5`.
    
- **Auto commit-and-sync after stopping file edits:** Toggle **ON**.
    
- **Auto pull interval:** Set to `5`.
    

### Connection Prefs

- **Pull on startup:** Toggle **ON**.
    
- **Push on commit-and-sync:** Toggle **ON**.
    
- **Pull before push:** Toggle **ON**.
    

### Identity (Important!)

Scroll to the bottom to **Commit author** and enter your name and GitHub email. Without this, GitHub will block your updates.

---

## How to Contribute

- **Read-Only Mode?** No! You can edit any file here.
    
- **Saving:** You don't need to save manually; Obsidian does it for you.
    
- **Syncing:** The plugin will "Push" your changes to the team and "Pull" my changes every 5 minutes automatically.
    
- **Status Check:** Look at the bottom right corner of this window. If you see a **Checkmark**, you are up to date.
    

> [!WARNING] **Conflict Prevention:** To avoid "Merge Conflicts," try not to edit the exact same paragraph as someone else at the same time. If we edit different files or different sections of the same file, Git handles it perfectly.