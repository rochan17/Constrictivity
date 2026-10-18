"""python run_tf.py <sample_dir> <out.pkl>: TauFactor on body_array>0, save potential field."""
import os, sys, pickle
os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
import numpy as np, torch
torch.set_num_threads(2)
import pipeline
if __name__ == '__main__':
    b = np.load(os.path.join(sys.argv[1], 'body_array.npy'))
    metrics, full = pipeline.run_taufactor((b > 0).astype(np.uint8), {'axis': 0}, save_full=True)
    print(metrics)
    pickle.dump({'metrics': metrics, 'phi': full['potential_field'].astype(np.float32)}, open(sys.argv[2], 'wb'))
