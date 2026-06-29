# Copilot Instructions

# General Repository Environment Instructions

We are using AllianceCan (Compute Canada).
You currently have terminal access to the login node, whereas any SLURM script will be run on a compute node.
The login node does not have GPU access, and does not have a SLURM_TMPDIR (temporary directory created only for the SLURM run).
Computer nodes do not have internet access.

# Environment

In this system, to create a virtual environment (which can be used both on the login nodes and the compute nodes), you must first load the modules. The following is an example:

```bash
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow
virtualenv --no-download ENV
source ENV/bin/activate
pip install --no-index --upgrade pip
pip install numpy --no-index
```