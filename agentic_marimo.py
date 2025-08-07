import marimo as mo
from marimo_notebook import app

if __name__ == "__main__":
    # Initialize the app
    app._maybe_initialize()
    # Get the runner from the app
    runner = app._runner
    # Get the first cell of the app
    first_cell_id = next(iter(app._cell_manager._cell_data.keys()))
    # Run the cell
    output, defs = runner.run_cell_sync(first_cell_id, {})
    print("Output:", output)
    print("Defs:", dict(defs))
    if hasattr(output, "_mime_"):
        print("MIME representation:", output._mime_())
