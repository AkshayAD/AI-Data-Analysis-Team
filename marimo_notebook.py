import marimo as mo

app = mo.App()

@app.cell
def __(mo):
    return mo.md("Hello, marimo!")
