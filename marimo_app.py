import marimo as mo

app = mo.App()

@app.cell
def __(mo):
    mo.md("## Marimo App")
    mo.md("This is a simple marimo app.")
    name = mo.ui.text(placeholder="Enter your name")
    mo.md(f"Hello, {name.value}!")

if __name__ == "__main__":
    app.run()
