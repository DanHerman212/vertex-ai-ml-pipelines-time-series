import nbformat
import sys
import os

def clean_notebook(file_path):
    print(f"Cleaning {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # 1. Remove Colab-specific Metadata
    keys_to_remove = ['accelerator', 'colab', 'widgets']
    for key in keys_to_remove:
        if key in nb.metadata:
            del nb.metadata[key]
            
    # Clean up language info if it has colab specific stuff (optional, usually fine)

    # 2. Filter Cells
    new_cells = []
    for cell in nb.cells:
        # A. Remove "Open in Colab" badge cells (Markdown)
        if cell.cell_type == 'markdown' and 'colab-badge.svg' in cell.source:
            print("  - Removing 'Open in Colab' badge")
            continue
            
        # B. Remove Drive Mounting cells (Code)
        if cell.cell_type == 'code' and 'from google.colab import drive' in cell.source:
            print("  - Removing Google Drive mount cell")
            continue
            
        # C. Remove cells starting with specific Colab commands if needed (like !pip install that you don't want)
        # (Optional: keeping pip installs is often useful for reproducibility, but you might want to comment them out)
        
        new_cells.append(cell)
    
    nb.cells = new_cells

    # 3. Save
    with open(file_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Done. Saved to {file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_colab_notebook.py <notebook.ipynb>")
        sys.exit(1)
    
    target = sys.argv[1]
    if os.path.isfile(target):
        clean_notebook(target)
    else:
        print(f"File not found: {target}")
