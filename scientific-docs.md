I've completely redesigned the document to fully leverage Org-mode's native capabilities! Here are the key improvements:

## Org-Mode Native Features Now Utilized

### 1. **Document Organization**
- **Properties drawers** (`:PROPERTIES:`) for metadata and organization
- **Custom IDs** for cross-referencing sections
- **TODO items** integrated into the research workflow
- **Tags** (`:noexport:`) to control what gets exported
- **STARTUP options** for automatic LaTeX preview and image display

### 2. **Literate Programming**
- **Named code blocks** (`#+name:`) for reusability
- **Session-based execution** (`:session physics`) - all Python blocks share variables
- **Header arguments** for controlling output (`:results output/file/table`)
- **Code block dependencies** - blocks can reference each other
- **Silent execution** (`:results silent`) for setup code

### 3. **Dynamic Content**
- **Inline code execution** with `C-c C-c`
- **Variable passing** between blocks (`:var n_max=10`)
- **Table generation** from code (`:results table`)
- **Automatic figure generation** with proper paths

### 4. **Enhanced Tables**
- **Org-native tables** that export beautifully to LaTeX
- **Column names** (`:colnames`) for table headers
- **LaTeX attributes** (`:booktabs t`) for professional styling

### 5. **Notes and Quotes**
- **Special blocks** (`#+begin_note`, `#+begin_quote`)
- **Comments** (`#+begin_comment`) for non-exported instructions

## Essential Keybindings for This Workflow

```
Document Management:
├─ C-c C-e l p          Export to PDF (via Tectonic)
├─ C-c C-e l l          Export to LaTeX
└─ C-c C-e h h          Export to HTML

Code Execution:
├─ C-c C-c              Execute current code block
├─ C-c C-v b            Execute all code blocks in buffer
├─ C-c C-v s            Execute subtree
├─ SPC m l x            Execute buffer (your custom binding)
└─ SPC m l A/B          Execute buffer above/below (your custom bindings)

LaTeX Integration:
├─ C-c C-x C-l          Toggle LaTeX preview
├─ C-c C-x C-v          Preview single equation
└─ (automatic with org-fragtog as you navigate)

Citations (when in text):
├─ SPC m c i            Insert citation (citar)
├─ SPC m c o            Open cited work
└─ SPC m c n            Open citation notes

Navigation:
├─ C-c C-n/p            Next/previous heading
├─ C-c C-u              Go to parent heading
├─ TAB                  Cycle visibility
└─ S-TAB                Cycle global visibility

Properties & Metadata:
├─ C-c C-x p            Set property
└─ C-c C-c              On property to edit
```

## Workflow Advantages

1. **Reproducibility**: Execute `C-c C-v b` to regenerate all results
2. **Iterative Development**: Change parameters and re-run specific blocks
3. **Version Control**: Plain text, perfect for Git
4. **Organization**: Properties and tags help manage research notes
5. **Cross-referencing**: Custom IDs and labels work seamlessly

## Key Differences from Previous Version

| Aspect | Before | Now |
|--------|--------|-----|
| Code blocks | Isolated | Session-based, share state |
| Tables | LaTeX syntax | Org-native with attributes |
| Structure | LaTeX-heavy | Org-native with export hints |
| Execution | One-off | Fully reproducible pipeline |
| Metadata | Minimal | Rich properties for organization |
| Organization | Linear | Hierarchical with TODO integration |

## Tips for Your Workflow

1. **Start sessions**: The `:session physics` means all Python blocks share variables—define constants once!

2. **Use named blocks**: Reference them later with `#+call:` or `<<name>>`

3. **Figure management**: Blocks with `:results file` automatically link images

4. **Table from code**: `:results table` creates Org tables that export beautifully

5. **Quick previews**: Your `org-fragtog` config shows equations as you navigate

6. **Property drawers**: Store metadata, creation dates, custom IDs for cross-refs

This document is now a true **literate programming** example where theory, code, and results live together and can be regenerated with a single command!
