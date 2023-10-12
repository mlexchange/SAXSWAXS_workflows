# SAXS/WAXS Analysis Workflows

Data analysis workflows for SAXS/WAXS

# Setup

# Clone repository and setup Python environment

```bash
git clone git@github.com:als-computing/SAXSWAXS_workflows.git
cd SAXSWAXS_workflows
python3 -m venv saxswaxs-workflows-env
source saxswaxs-workflows-env/bin/activate
```

# Set up environment file with relevant paths

Create a file `.env` with the following content

```bash
TILED_URI="http://127.0.0.1:8888"
PREFECT_API_URL="http://127.0.0.1:4200/api"
TILED_API_KEY="<randomly generated key>"
PATH_TO_DATA="<path to folder that has structure as beamtime (with /raw, /processed)>"
```

# Prefect

In one terminal that has the environment activated start a prefect server

```bash
prefect server start
```

# Tiled server

Within the interface folder (the other repository), run the two Tiled commands

# First test

Adapt the parameters in the example in `reduction.py` to point to a dataset that is contained in the folder

(Later once we confirmed that the first part runs)

In another terminal, create a work-pool and start a work-pool

```bash
prefect work-pool create reduction-pool
prefect worker start --pool 'reduction-pool'
```