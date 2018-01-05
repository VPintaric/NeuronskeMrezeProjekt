import numpy as np 

def pixel_bytes_to_array(pxls, w, h):
    def get_pxl_val_at(w, i, j):
        idx = 3 * (i + j * w)
        return [pxls[idx], pxls[idx + 1], pxls[idx + 2]]
    arr = np.array([get_pxl_val_at(w, i, j) for j in range(h) for i in range(w)])
    return arr # normalize pixels
        
def array_to_pixel_bytes(arr):
    a = np.concatenate(arr, 0)
    pxls = [int(max(0., min(255, val))) for val in a]
    return bytes(pxls)