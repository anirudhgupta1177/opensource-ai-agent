# Connect Claude Code with Cursor

Claude Code (Anthropic’s coding assistant) doesn’t auto-detect Cursor, but you can install its VS Code extension manually and then connect. Follow these steps.

---

## Prerequisites

- **Node.js** and **npm** (for installing Claude Code)
- **Cursor** installed
- **Claude API access** (Claude Pro/Max, Team/Enterprise, or API key)

---

## Step 1: Install Claude Code locally

In a terminal (outside Cursor):

```bash
npm install -g @anthropic-ai/claude-code
```

Verify:

```bash
claude
```

If the interactive session starts, type `/doctor` and confirm it says you’re running from a **local** installation (e.g. `~/.claude/local`). If it says “global” instead, run:

```bash
/migrate-installer
```

then quit and run `claude` again.

---

## Step 2: Find the VSIX file

After Claude Code is installed, the extension file is here:

| OS       | Path |
|----------|------|
| **macOS / Linux** | `~/.claude/local/node_modules/@anthropic-ai/claude-code/vendor/claude-code.vsix` |
| **Windows**      | `%USERPROFILE%\.claude\local\node_modules\@anthropic-ai\claude-code\vendor\claude-code.vsix` |

If you used **nvm**, the local path might be under your Node version, e.g.:

```bash
~/.nvm/versions/node/v22.x.x/lib/node_modules/@anthropic-ai/claude-code/vendor/claude-code.vsix
```

To locate it:

```bash
npm list -g @anthropic-ai/claude-code
```

Then look for `node_modules/@anthropic-ai/claude-code/vendor/claude-code.vsix` under that install path, or search:

```bash
find ~/.claude ~/.nvm 2>/dev/null -name "claude-code.vsix" 2>/dev/null | head -5
```

---

## Step 3: Install the extension in Cursor

Use **one** of these methods.

### Method A: Command line (recommended)

Quit Cursor, then in a terminal:

```bash
cursor --install-extension ~/.claude/local/node_modules/@anthropic-ai/claude-code/vendor/claude-code.vsix
```

On Windows, use the full path to the `.vsix` (e.g. `%USERPROFILE%\.claude\...\claude-code.vsix`).

To install for a specific Cursor profile:

```bash
cursor --install-extension /path/to/claude-code.vsix --profile "Your Profile Name"
```

### Method B: Drag and drop

1. Open **Cursor**.
2. Open the **Extensions** panel: **Cmd+Shift+X** (Mac) or **Ctrl+Shift+X** (Windows/Linux).
3. Drag the **claude-code.vsix** file from Finder/Explorer into the Extensions panel.
4. When prompted, confirm installation.

Many people find this more reliable than the CLI.

### Method C: Command Palette

1. In Cursor, press **Cmd+Shift+P** (Mac) or **Ctrl+Shift+P** (Windows/Linux).
2. Run: **Extensions: Install from VSIX...**
3. Browse to the `claude-code.vsix` file and select it.

---

## Step 4: Restart Cursor

Quit Cursor completely and open it again so the extension loads.

---

## Step 5: Connect Claude Code to Cursor

1. In Cursor, open the **integrated terminal** (e.g. **Ctrl+`** or View → Terminal).
2. Run:

   ```bash
   claude
   ```

3. In the Claude Code session, run:

   ```bash
   /ide
   ```

4. Choose **Cursor** from the list (if it appears).
5. Confirm; Claude Code should now be connected to Cursor.

Use Cursor’s terminal (not an external one) so the extension can detect the editor.

---

## Shortcuts (after connection)

| Action              | Mac           | Windows / Linux |
|---------------------|---------------|------------------|
| Open Claude Code    | **Cmd+Esc**   | **Ctrl+Esc**     |
| Insert file ref      | **Cmd+Option+K** | **Alt+Ctrl+K** |

---

## If something doesn’t work

- **“No Available IDEs Detected”**  
  - Install the extension (Step 3), then **fully quit and restart Cursor**.  
  - Run `claude` from Cursor’s **integrated** terminal, not an external one.

- **CLI says installed but extension not in Cursor**  
  - Use **Method B (drag and drop)** to install the `.vsix` directly in the Extensions panel.

- **Claude Code installed in WSL, Cursor on Windows**  
  - Copy the `.vsix` to a Windows folder (e.g. Downloads), then in Cursor use **Extensions: Install from VSIX...** and select that file.

- **Multiple Cursor profiles**  
  - Use `cursor --install-extension ... --profile "ProfileName"` so the extension is installed in the profile you use.

---

## References

- [Claude Code IDE integrations (official)](https://docs.anthropic.com/en/docs/claude-code/ide-integrations)
- [Claude Code for VS Code](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code)
- [GitHub: Claude Code doesn’t detect Cursor #1279](https://github.com/anthropics/claude-code/issues/1279)
