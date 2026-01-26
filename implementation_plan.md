# Plan: Restore Electron App for Synapse AI

## Goal
The user wants to use the "Electron App" wrapper they had before, but with the new AI capabilities (LLM Feed, etc.) we just built.

## Steps
1.  **Repoint Electron**: Update `ElectronApp/src/main/main.js` to load `http://localhost:3030` (our AI Dashboard) instead of the old legacy HTML.
2.  **Update Launcher**: Modify `LAUNCH_COMMAND_CENTER.bat` to:
    *   Start the **AI Company OS** backend (Port 3030).
    *   Launch the **Electron App**.
3.  **Verify**: Ensure the shortcut `Zero Human Command Center.lnk` (which points to the BAT file) now opens the App window with the new content.

## Risk
*   If `ai-company-os` takes time to start, the Electron app might show a "Connection Refused" error initially. We should add a small delay or retry logic if possible, or just a `timeout` in the batch file.
