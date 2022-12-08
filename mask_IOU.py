import torch
import numpy as np
import matplotlib.pyplot as plt
def calculateIOU(mask_1,mask_2):
  mask_1=mask_1.flatten()
  mask_2=mask_2.flatten()
  mask_1_posi_0=np.where(mask_1==0)[0]
  mask_2_posi_0=np.where(mask_2==0)[0]
  common_intersec=len(list(set(mask_1_posi_0).intersection(mask_2_posi_0)))
  union=len(mask_1_posi_0)+len(mask_2_posi_0)-common_intersec
  return 1.0*common_intersec/union

def main():
    model_1_path="results/masks/gradcam_3.5/checkpoints/checkpoint-0.pt"
    model_2_path="results/masks/magweight_3.5/checkpoints/checkpoint-0.pt"
    model_1_state_dict=torch.load(model_1_path)['model_state_dict']
    model_2_state_dict=torch.load(model_2_path)['model_state_dict']
    arr_IOU=[]
    #intermediate.dense.weight_mask
    for k in model_1_state_dict.keys():
        if(k.endswith('intermediate.dense.weight_mask')):
            print(f"Found key:{k}")
            model_1_mask=model_1_state_dict[k].detach().cpu().numpy()
            model_2_mask=model_2_state_dict[k].detach().cpu().numpy()
            curr_IOU=calculateIOU(model_1_mask,model_2_mask)
            print(f"IOU for this key is :{curr_IOU}")
            arr_IOU.append(curr_IOU)
    arr_IOU=np.array(arr_IOU)
    plt.plot(np.arange(len(arr_IOU)),arr_IOU)
    plt.savefig('gradcam_3.5_magweight_3.5.png')
    plt.show()
    print(f"Average IOU is: {np.mean(arr_IOU)}")
    print(f"Max IOU overlap is: {np.max(arr_IOU)}")
    print(f"Min IOU overlap is: {np.min(arr_IOU)}")

if __name__ == "__main__":
    main()