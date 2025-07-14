# Distributed Finetuning Example: VideoMAE on Kinetics-400 5k-filtered

This directory contains an example script for distributed finetuning of VideoMAE on a Kinetics-400 5k-filtered subset (156 classes) using two nodes.

## Prerequisites
- Passwordless SSH between nodes (recommended)
- Both nodes have access to the dataset and pre-trained model (paths must be valid on both)
- WANDB account (optional, for logging)
- Python environment and dependencies installed on both nodes

## SSH Tunnel Setup
To allow communication between nodes, open an SSH tunnel from **node 1** (worker) to **node 0** (master):

```
# On node 1 (worker), run:
ssh -N -L 12320:localhost:12320 <user>@<node0-ip>
```
- Replace `<user>` and `<node0-ip>` with your actual username and the IP address of node 0.
- Keep this tunnel open for the duration of training.

## How to Run

### On Node 0 (Master)
```bash
bash finetune_distributed.sh 0 127.0.0.1
```

### On Node 1 (Worker)
```bash
bash finetune_distributed.sh 1 127.0.0.1
```

- The first argument is the `node_rank` (0 for master, 1 for worker).
- The second argument is the `master_addr` (use `127.0.0.1` if using SSH tunnel as above).

## Script Features
- Output and log directories are timestamped for each run.
- WANDB run name is also timestamped.
- All key parameters are set as variables at the top of the script for easy modification.
- WANDB logging is enabled by default. Set `USE_WANDB=0` in the script to disable.
- Set your WANDB entity in the script (`WANDB_ENTITY`).

## Argument Handling in Distributed Training
- **wandb arguments** (`--use_wandb`, `--wandb_project`, `--wandb_run_name`, `--wandb_entity`) and other logging-related arguments are **only used by the main process (rank 0)**.
- On all other nodes (workers), these arguments are safely ignored by the codebase.
- **It is safe and recommended to pass the same arguments to all nodes** for simplicity. The code will ensure only the main process logs to wandb and creates output logs.

## Example SSH Tunnel Diagram

```
[Node 1] --(ssh tunnel:12320)--> [Node 0]
```

## Notes
- Make sure both nodes can access the same data and model paths.
- You can adjust batch size, learning rate, and other parameters in the script as needed.
- For more nodes, increase `NNODES` and launch the script on each node with the appropriate `node_rank`. 