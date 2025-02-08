# SAXS/WAXS Analysis Workflows

Data analysis workflows for SAXS/WAXS

## Clone repository and setup Python environment

```bash
git clone git@github.com:als-computing/SAXSWAXS_workflows.git
cd SAXSWAXS_workflows
python -m venv saxswaxs-workflows-env
source saxswaxs-workflows-env/bin/activate
pip install .
```

If you are using this in a prefect and tiled environment, also run (quotes are required for zsh):
```bash
pip install '.[prefect_tiled]'
```

If you are using this in a development environment, also run:
```bash
pip install '.[dev]'
```


The command `source saxswaxs-workflows-env/bin/activate` may need to be adapted for the specific operating system, see the [venv](https://docs.python.org/3/library/venv.html) documentation

## Set up environment file with relevant paths

Create a file `.env` with the following content

```bash
TILED_URI="http://127.0.0.1:8888"
PREFECT_API_URL="http://127.0.0.1:4200/api"
TILED_API_KEY="<randomly generated key>"
PATH_TO_DATA="<path to folder that contains raw data>"
PATH_TO_PROCESSED_DATA="<path to folder where processed data can be written>"
PREFECT_WORK_DIR="<path to folder where this code resides>"
```

## Prefect Server

In one terminal that has the environment activated start a prefect server

```bash
prefect server start
```

As instructed in the Prefect server startup prompt, make sure prefect is configured with the correct `PREFECT_API_URL`:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

## Tiled server

Within the interface folder (the other repository `/workflow-viz`), follow the instructions to start a Tiled server, and a Tiled watch process that observes changes in a directory

## First test

Adapt the parameters in the example in `reduction.py` to point to a dataset that is contained in the folder and run it

```bash
python saxswaxs-workflows/flows/reduction.py
```

## Setup for beamtime

(Once we confirmed that the first part runs)

In another terminal, create work-pools:, the `reduction-pool` for reducing data, the `fitting-pool` for feature extraction, and the `gpcam-pool` for running gpCAM,
and deploy all flows that are defined within `prefect.yaml`. For convinience, these steps are summarized in the script

```bash
./create_deployments.sh
```

Finally, start the workers with

```bash
prefect worker start --pool 'reduction-pool'
```

## Copyright

MLExchange Copyright (c) 2023, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at <IPO@lbl.gov>.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
