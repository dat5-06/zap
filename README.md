# EV Charging ML Model

## Getting Started

On **Unix** based systems:

```bash
python -m venv venv
source ./venv/bin/activate
pip install -e .
```

On **Windows**:

```ps1
python -m venv venv
venv\Scripts\activate
pip install -e .
```

## Notebooks

First, we must add the virtual environment to the jupyter notebook kernel.

```bash
ipython kernel install --user --name=venv
```

Then, we can start the jupyter notebook server.

```bash
jupyter lab
```

Now, navigate to [http://localhost:8888](http://localhost:8888), and remember to select the venv as the kernel.

## Linting

In **VSCode**, install the [ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

In **neovim**, look at the [documentation](https://docs.astral.sh/ruff/editors/setup/#neovim) for ruff.
> [!NOTE]
> If using LazyVim, `ruff` is already included in the `lang.python` LazyExtras package.
