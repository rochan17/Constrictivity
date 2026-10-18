from skimage import measure
from skimage import filters
import scipy
import scipy.ndimage
import numpy as np
import random

# Define spherical structuring element
# Circle is for 2D, Sphere is for 3D
# http://www.star.ucl.ac.uk/~kakiichi/codetools.html (Reference)

def circle(n):
    struct = np.zeros((2*n+1, 2*n+1))
    x, y   = np.indices((2*n+1 ,2*n+1))
    mask   = (x-n)**2 + (y-n)**2 <= n**2
    struct[mask] = 1
    return struct.astype(bool)

def sphere(n):
    struct = np.zeros((2*n+1, 2*n+1, 2*n+1))
    x, y, z = np.indices((2*n+1 ,2*n+1, 2*n+1))
    mask  = (x-n)**2 + (y-n)**2 + (z-n)**2 <= n**2
    struct[mask] = 1
    return struct.astype(bool)

# Convert to binary fields

def dilate_phase(input_field, phaseID, n=2):
    """
    This function is made for dilation.
    If you want to do a sequence dilation on one matrix,
    after one dilation, use the output matrix, again.

    input_field: 3D matrix, which you want to dilate.

    PhaseID: A number of phase, which you want to dilate.
             (example) 1: Pore, 2: LSM, 3: YSZ

    n: radius of dilation.
    """
    binary_field = np.zeros(input_field.shape)
    binary_field[input_field == phaseID] = 1
    dilated_field = scipy.ndimage.binary_dilation(binary_field, structure=sphere(n))
    output_field = input_field.copy()
    output_field[dilated_field == 1] = phaseID
    return output_field

# Isolated Phase Removal
# http://scipy-lectures.org/packages/scikit-image/auto_examples/plot_labels.html(Rerference)

def find_isolated(input_field, phaseID, boundary, isolated_phaseID):
    """
    This function is made for finding out an isolated phase.
    Isolated phases of phaseID in input_field will be converted into isolated_phaseID.

    input_field: 3D matrix,from which you want to find isolated phases.

    phaseID: The number of isolated phase you are trying to find.

    boundary: You can use 'top' or 'bottom'.
              Under the assumption we already made,
              Pore & LSM phases are connected to 'top' and YSZ is connected to 'bottom'.

    isolated_phaseID: To differentiate the isolated phases,
                      you can assign the number of the isolated one.
                      I usually assign islated phases as 0.
                      Keep consistency.
                      (example) isolated_phaseID = 0

    (Exmaple) For finding out isolated phases(1, 2, and 3) for one matrix, you need to do like this.

              isolated_sub = find_isolated(subvolume, 1, 'top', 0)
              isolated_sub = find_isolated(isolated_sub, 2, 'top', 0)
              isolated_sub = find_isolated(isolated_sub, 3, 'bottom', 0)

              By doing this, isolated_sub consists of 1, 2, 3, and 0.
              When you do a mesh process(specifically, threshold step), you can ignore element 0.
    """
    # apply connected component algorithm
    binary_field = input_field==phaseID
    label_field  = measure.label(binary_field, connectivity=1)

    # store boundary voxel values
    boundDict          = {'top'   : 0, 'bottom': -1}
    boundDict1         = {'front'   : 0, 'back': -1}
    boundDict2         = {'left'   : 0, 'right': -1}

    if boundary == 'top' or boundary == 'bottom':
        labels_at_boundary = np.unique(label_field[:,:,boundDict[boundary]])
    elif boundary == 'front' or boundary == 'back':
        labels_at_boundary = np.unique(label_field[:,boundDict1[boundary],:])
    else:
        labels_at_boundary = np.unique(label_field[boundDict2[boundary],:,:])

    # find isolated phases and store isolated position
    num_labels    = np.max(label_field)
    if num_labels == 1:
        print('No isolated phases')
    isolated_field = np.zeros(input_field.shape).astype(int)
    for label in range(1,num_labels+1):
        if label not in labels_at_boundary:
            isolated_field[label_field == label] = 1

    output_field = np.copy(input_field).astype(int)
    output_field[isolated_field == 1] = isolated_phaseID
    return output_field

def find_isolated_multi(input_field, phaseID, axes, isolated_phaseID=0):
    binary_field = input_field == phaseID
    label_field = measure.label(binary_field, connectivity=1)

    # Start with all labels as candidates, then intersect per axis
    all_labels = set(range(1, label_field.max() + 1))
    labels_to_keep = all_labels.copy()

    for axis in axes:
        # Labels must touch BOTH faces of this axis
        slicer0 = [slice(None)] * 3; slicer0[axis] = 0
        slicerN = [slice(None)] * 3; slicerN[axis] = -1
        at_face0 = set(np.unique(label_field[tuple(slicer0)])) - {0}
        at_faceN = set(np.unique(label_field[tuple(slicerN)])) - {0}
        labels_to_keep &= (at_face0 & at_faceN)  # must touch both

    isolated_mask = (~np.isin(label_field, list(labels_to_keep))) & binary_field
    output_field = input_field.copy()
    output_field[isolated_mask] = isolated_phaseID
    return output_field


# Resampling

def resampling(input_field, factor=2):
    resampled = scipy.ndimage.zoom(input_field, factor, order=0)
    return resampled

# Appending layer

def append_electrolyte(input_field, layer_thickness, electrolyte_phaseID):
    """
    This function is made for appending electrolyte.

    input_field: 3D matrix you want to append electrolyte layer.

    layer_thickness: Thickness of electrolyte.

    electrolyte_phaseID: The number of electrolyte phase.
                         We always use YSZ as electrolyte.
                         (example) electrolyte_phaseID=3
    """
    el_layer = np.zeros((np.size(input_field, 0), np.size(input_field, 1), layer_thickness), dtype=int)
    el_layer[:] = electrolyte_phaseID # electrolyter layer consists of YSZ
    output_field = np.concatenate((input_field, el_layer), axis=2)
    return output_field

# Revised_Triple Phase Boundary labeling
# Global variable applied + only one overwrite

def label_TPB(input_field):
    labeling_TPB = np.copy(input_field)
    labeling_test = np.zeros((input_field.shape))
    num_TPB = np.max(input_field)+1
    for i in range(0,input_field.shape[0]-1):
        for j in range(0,input_field.shape[1]-1):
            for k in range(0,input_field.shape[2]-1):
                test_unit = input_field[i:i+2,j:j+2,k:k+2]
                if set(np.unique(test_unit).astype(int)) >= set((1,2,3)):
                    labeling_test[i:i+2,j:j+2,k:k+2] = num_TPB
    labeling_TPB[labeling_test == num_TPB] = num_TPB
    return labeling_TPB

# Define the function for infiltration (Revision version & Need Random module)

def infiltration(input_field, integer_probability, infiltrated_phaseID):
    """
    This function is made for infiltration.

    input_field: 3D matrix, which will be infiltrated.

    integer_probability: Literally, the probability for infiltration as integer.
                         (example) If you want to do a infiltration with 4 % (0.04) probability,
                                    then, integer_probability = 4

    infiltrated_phaseID: You can assign the phase of infiltrated particles.
                         (example) In my case, if I want to make infiltrated phase as LSM,
                                   infiltrated_phaseID = 2
                                   1: Pore, 2: LSM, 3: YSZ
                                   I use above order for simulation.
    """
    binary_field = np.zeros((input_field.shape))
    for i in range(0,input_field.shape[0]-1):
        for j in range(0,input_field.shape[1]-1):
            for k in range(0,input_field.shape[2]-1):
                if input_field[i,j,k] == 1:
                    if (set([input_field[i+1,j,k],input_field[i,j+1,k],input_field[i,j,k+1]]) >= set([2]) or
                        set([input_field[i+1,j,k],input_field[i,j+1,k],input_field[i,j,k+1]]) >= set([3]) or
                        set([input_field[i+1,j,k],input_field[i,j+1,k],input_field[i,j,k+1]]) >= set([2,3])):
                        number = random.randint(1,100)
                        if number <= integer_probability: # This makes probability p = 6/100 = 0.06
                            binary_field[i,j,k] = 1       # Probability could be controlled by integer_probability
    infiltrated_field = dilate_phase(binary_field, phaseID=1, n=2)
    output_field = np.copy(input_field)
    output_field[infiltrated_field == 1] = infiltrated_phaseID # Making nano-infiltrated phase as whatever you want. It could be any number.
    output_field[input_field == 2] = 2 # Halved(if the phase was LSM or YSZ, turn the phase to original phase)
    output_field[input_field == 3] = 3 # Halved
    return output_field
