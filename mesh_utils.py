import numpy as np
import scipy.ndimage as ndi
from skimage import measure
from subvolume_utils2 import dilate_phase, find_isolated_multi, resampling

def premesh(voxels, dilation_n=2, axis=1, resampling_factor=1, resample=True):
    """
    Clean a 3-phase voxel array before meshing.

    Steps:
      1. Resample (nearest-neighbour, preserves labels)
      2. Dilate each phase to close small gaps and thin necks
      3. Remove disconnected regions from each phase → set to 0
      4. Report how many voxels are 0 (excluded from mesh)

    Parameters
    ----------
    voxels             : 3D numpy array with values 1, 2, 3
    dilation_n         : radius of spherical dilation structuring element
    axis               : axis (0=x, 1=y, 2=z) whose two faces define
                         the percolation direction to check
    resampling_factor  : zoom factor for resampling
    resample           : whether to resample at all

    Returns
    -------
    cleaned numpy array (values 0/1/2/3), 0 = excluded from mesh
    """
    if resample:
        result = resampling(voxels, factor=resampling_factor)
    else:
        result = voxels

    print(f"Resampled: {voxels.shape} → {result.shape}")

    # dilate in priority order so dominant phases win at boundaries
    for phase in [1, 2, 3]:
        result = dilate_phase(result, phase, n=dilation_n)
    print(f"Dilation done (radius={dilation_n})")

    # remove disconnected regions for each phase (independently),
    # checking both faces of `axis` in one call
    for phase in [1, 2, 3]:
        result = find_isolated_multi(result, phase, axes=[axis], isolated_phaseID=0)
    print("Isolated phase removal done")

    total = result.size
    n_zero = int((result == 0).sum())
    print(f"\nVoxels set to 0 (excluded from mesh): {n_zero} "
          f"({100 * n_zero / total:.1f}% of domain)")
    for phase in [1, 2, 3]:
        n = int((result == phase).sum())
        print(f"  Phase {phase}: {n} voxels ({100 * n / total:.1f}%)")

    return result


def percolated_phase(voxels, axis=0, resampling_factor=1, resample=True):
    """
    Clean a 3-phase voxel array before meshing.

    Steps:
      1. Resample (nearest-neighbour, preserves labels)
      2. Dilate each phase to close small gaps and thin necks
      3. Remove disconnected regions from each phase → set to 0
      4. Report how many voxels are 0 (excluded from mesh)

    Parameters
    ----------
    voxels             : 3D numpy array with values 1, 2, 3
    dilation_n         : radius of spherical dilation structuring element
    axis               : axis (0=x, 1=y, 2=z) whose two faces define
                         the percolation direction to check
    resampling_factor  : zoom factor for resampling
    resample           : whether to resample at all

    Returns
    -------
    cleaned numpy array (values 0/1/2/3), 0 = excluded from mesh
    """
    if resample:
        result = resampling(voxels, factor=resampling_factor)
        print(f"Resampled: {voxels.shape} → {result.shape}")
    else:
        result = voxels

    # remove disconnected regions for each phase (independently),
    # checking both faces of `axis` in one call
    for phase in [1, 2, 3]:
        result = find_isolated_multi(result, phase, axes=[axis], isolated_phaseID=0)
    print("Isolated phase removal done")

    total = result.size
    n_zero = int((result == 0).sum())
    print(f"\nVoxels set to 0 (excluded from mesh): {n_zero} "
          f"({100 * n_zero / total:.1f}% of domain)")
    for phase in [1, 2, 3]:
        n = int((result == phase).sum())
        print(f"  Phase {phase}: {n} voxels ({100 * n / total:.1f}%)")

    return result