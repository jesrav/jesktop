# Jesktop
A personal web app for interacting with my notes.

WIP

## Getting started

### Prerequisites
- [Node.js](https://nodejs.org/en/)
- Python 3.11
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- [Just](https://github.com/casey/just) command line runner

### Setup
1. copy `.env.example` to `.env` and fill in the values. See `notes_chat/config.py` for options.
2. Set up a virtual python env with Python 3.12
3. Install the python dependencies with `just install_dev`

### Running the app
```bash
just run
```

### Running the tests
```bash
just test
just lint
```

See the `justfile` for more commands