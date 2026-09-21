#!/bin/bash
#SBATCH --partition=acmegroup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=4:00:00
#SBATCH --output=int.out
#SBATCH --error=int.err
#SBATCH --mail-user=rbajpai@andrew.cmu.edu
#SBATCH --mail-type=ALL

# 128 cores total, 16 cores/microstructure -> 8 microstructures run in
# parallel at a time. Adjust --cores-per-microstructure to change that.

module purge
module load cuda/11.7 aocc/3.2.0
module load anaconda3/2021.05

cd ~
cd ~/group/Constrictivity/
source activate rbenv


python3 run_batch.py \
    --data-dir data_new \
    --cores-per-microstructure 16 \
    --phase-id 3 \
    --axis 0 \
    --segmented-file segmented_percolated.npy \
    --body-array-file body_array.npy \
    --taufactor-config '{
        "iter_limit": 100000,
        "conv_crit": 0.0001
    }' \
    --berg-voxel-config '{
        "axis": 0,
        "dx": 1.0,
        "sigma": 1.0,
        "inlet_value": 1.0,
        "outlet_value": 0.0,
        "conductance_mode": "physical",
        "length_mode": "geometric_dx",
        "volume_mode": "node_unsplit",
        "current_split_mode": "flux_weighted",
        "tortuosity_mode": "chord_corrected",
        "bend_radius_vox": 5.0,
        "require_chord_inside_phase": true,
        "chord_samples_per_dx": 24,
        "use_pyamg": true,
        "kcl_tol": 1e-12,
        "kcl_maxiter": 300,
        "rel_current_tol": 1e-10,
        "abs_current_tol": 1e-12,
        "progress_every": 10000,
        "decomposition_method": "greedy_fast",
        "use_numba_decomposition": true,
        "store_legacy_path_arrays": true,
        "volume_chunk": 20000,
        "chord_chunk": 20000
    }' \
    --berg-cc-config '{
        "min_throat_size": 1,
        "dilate_both": true,
        "surface_axis": 0,
        "skip_surface_surface": true,
        "area_method": "vox_projection",
        "alpha": 0.5,
        "split_volume_equal": false,
        "sigma": 1.0,
        "inlet_value": 1.0,
        "outlet_value": 0.0,
        "direction": "x",
        "delta_V": 1.0,
        "voxel_size": 1.0,
        "streamtube_method": "flow_decomposition",
        "_force_recompute": 4
    }'\
    --save-full \
    --verbose