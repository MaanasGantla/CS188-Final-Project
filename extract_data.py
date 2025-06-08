import numpy as np
from load_data import reconstruct_from_npz

def build_dataset(npz_path, normalize=False):
    """
    load demos.npz, 
      return X which is (N_total, 10)
      each vector is of the shape: [(eef_x,eef_y,eef_z), (obj_x,obj_y,obj_z), (obj_qx,obj_qy,obj_qz,obj_qw)]

      return Y which is (N_total, 7) 
      7d action vector
      normalized with zero-mean unit variance per column. mean_X and std_x are stored in norm_stats.npz
    """
    demos = reconstruct_from_npz(npz_path)
    X_list, Y_list = [], []

    for demo_id, fields in demos.items():
        eef_pos    = fields['obs_robot0_eef_pos']    
        obs_object = fields['obs_object']            
        actions    = fields['actions']               

        T = eef_pos.shape[0]
        if obs_object.shape[0] != T or actions.shape[0] != T:
            raise ValueError(f"invalid demo lengths for demo {demo_id}")

        for t in range(T):
            ee = eef_pos[t]                   
            obj = obs_object[t, 0:7]          
            s_t = np.concatenate([ee, obj[0:3], obj[3:7]], axis=0)  
            a_t = actions[t]                  
            X_list.append(s_t)
            Y_list.append(a_t)


    X = np.vstack(X_list).astype(np.float32)  
    Y = np.vstack(Y_list).astype(np.float32)  

    if not normalize:
        return X, Y
    else:
        mean_X = X.mean(axis=0, keepdims=True)            
        std_X  = X.std(axis=0, keepdims=True) + 1e-8       

        X_normed = (X - mean_X) / std_X                    

        np.savez("norm_stats.npz", mean_X=mean_X, std_X=std_X)

        return X_normed, Y, mean_X, std_X


if __name__ == "__main__":

    X, Y = build_dataset("demos.npz", normalize=False)
    print("everything w/o normalization worked, norm_stats.npz is made")

    Xn, Yn, mean_X, std_X = build_dataset("demos.npz", normalize=True)
    print("everything worked, norm_stats.npz is made")



