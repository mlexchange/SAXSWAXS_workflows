# SAXS/WAXS Analysis Workflows

Data analysis workflows for SAXS/WAXS

# Clone repository and setup Python environment

```bash
git clone git@github.com:als-computing/SAXSWAXS_workflows.git
cd SAXSWAXS_workflows
python -m venv saxswaxs-workflows-env
source saxswaxs-workflows-env/bin/activate
pip install -r requirements.txt
```

The command `source saxswaxs-workflows-env/bin/activate` may need to be adapted for the specific operating system, see the [venv](https://docs.python.org/3/library/venv.html) documentation

# Set up environment file with relevant paths

Create a file `.env` with the following content

```bash
TILED_URI="http://127.0.0.1:8888"
PREFECT_API_URL="http://127.0.0.1:4200/api"
TILED_API_KEY="<randomly generated key>"
PATH_TO_DATA="<path to folder that has structure as beamtime (with subfolders /raw, /processed/ ...)>"
```

# Prefect Server

In one terminal that has the environment activated start a prefect server

```bash
prefect server start
```

As instructed in the Prefect server startup prompt, make sure prefect is configured with the correct `PREFECT_API_URL`:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

# Tiled server

Within the interface folder (the other repository `/workflow-viz`), follow the instructions to start a Tiled server, and a Tiled watch process that observes changes in a directory

# First test

Adapt the parameters in the example in `reduction.py` to point to a dataset that is contained in the folder and run it

```
python
```

(Later once we confirmed that the first part runs)

In another terminal, create a work-pool and start a work-pool

```bash
prefect work-pool create reduction-pool
prefect worker start --pool 'reduction-pool'
```