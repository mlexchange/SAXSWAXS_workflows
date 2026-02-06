# SAXS/WAXS Analysis Workflows

A collection of Prefect workflows for data reduction and feature extraction in SAXS/WAXS experiments.

This repository provides the execution and orchestration layer for modular data-processing workflows
used in near–real-time and autonomous scattering experiments.

## Software architecture

This repository defines Prefect workflows composed of modular tasks for data reduction and feature
extraction. Experimental data and intermediate results are accessed through **Tiled**, while
execution is orchestrated using **Prefect** for consistent runs across local workstations and
central compute resources.

## Relationship to workflow-viz

The workflows defined here can be executed and validated through the Dash-based user interface provided
by workflow-viz: <https://github.com/mlexchange/workflow-viz>

The same Prefect tasks and workflows are used during interactive configuration, near–real-time
processing, and autonomous operation, ensuring that validated parameters are directly transferable
between modes.

## Initial setup: Clone the repository and create an environment

```bash
git clone git@github.com:als-computing/SAXSWAXS_workflows.git
cd SAXSWAXS_workflows
python -m venv saxswaxs-workflows-env
source saxswaxs-workflows-env/bin/activate
pip install -r requirements.txt
```

The command `source saxswaxs-workflows-env/bin/activate` may need to be adapted for the specific operating system; see the [venv](https://docs.python.org/3/library/venv.html) documentation.

## Initial + beamtime setup: Configure environment

Create a `.env` file based on the provided example:

```bash
cp .env.example .env
```

Edit the file to configure data paths, Tiled access, Prefect settings, and (optionally) beamline control parameters. Environment variables related to data access must be consistent with those used by `workflow-viz`.

## Prefect server

In one terminal with the environment activated, start a Prefect server:

```bash
prefect server start
```

As instructed in the Prefect server startup prompt, make sure Prefect is configured with the correct `PREFECT_API_URL`:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

## Tiled server

Within the workflow-viz repository, follow the instructions to start a Tiled server and ingest raw and already present processed data.

## First test

Adapt the parameters in the example in `reduction.py` to point to a dataset contained in the folder and run it:

```bash
python saxswaxs-workflows/flows/reduction.py
```

## Beamtime setup

Once the first part runs, in another terminal create work pools (`reduction-pool`, `fitting-pool`, and `gpcam-pool`) and deploy all flows defined in `prefect.yaml`. For convenience, these steps are summarized in the script:

```bash
./create_deployments.sh
```

Finally, start the workers for reduction and fitting with

```bash
prefect worker start --pool 'reduction-pool'
```

```bash
prefect worker start --pool 'fitting-pool'
```

## Beamtime execution

For beamtime deployments, workflows can be triggered either programmatically or through file-based
monitoring of incoming detector data. Optional components support continuous operation and autonomous
control via ZMQ-based messaging.

The following scripts provide reference implementations of these components:

- `saxswaxs_workflows/flows/file_watcher.py` — monitors `PATH_TO_DATA` for new detector files and
  schedules the selected reduction workflow (deployment name and parameters are defined in the script).
- `saxswaxs_workflows/flows/autonomous_zmq.py` — listens for reduction updates over ZMQ and interfaces
  with beamline control via ZMQ, configured through `BL_SERVER` and `BL_PORT`.

## Copyright

MLExchange Copyright (c) 2023, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at <IPO@lbl.gov>.

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
