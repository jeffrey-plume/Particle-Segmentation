```python
import os
from cellpose import models, train
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_data(images_dir, masks_dir):
    images = os.listdir(images_dir)
    img_paths = [os.path.join(images_dir, img) for img in images]
    mask_paths = [os.path.join(masks_dir, img) for img in images]
    return img_paths, mask_paths

def prepare_data(images_dir, masks_dir, test_size=0.2):
    img_paths, mask_paths = load_data(images_dir, masks_dir)

    images = [np.array(Image.open(file).convert("RGB")) for file in img_paths]
    masks = [np.array(Image.open(file).convert("L")) for file in mask_paths]  # single-channel mask
    
    return images, masks


if __name__ == '__main__':

    images_dir = "C:/Users/Lenovo/OneDrive/Documents/GitHub/Synthetic Images/emps-main/images"
    masks_dir = "C:/Users/Lenovo/OneDrive/Documents/GitHub/Synthetic Images/emps-main/segmaps"
    
    # Prepare data_
    imgs, ground_Truth = prepare_data(images_dir, masks_dir)
    model = models.CellposeModel(model_type="C:/Users/Lenovo/OneDrive/Documents/GitHub/Synthetic Images/models/segmentEMPD.pth")
    predictions, flows, styles = model.eval(imgs, channels=[1,2])
```

    C:\Users\Lenovo\anaconda3\envs\tf-env\lib\site-packages\cellpose\resnet_torch.py:275: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      state_dict = torch.load(filename, map_location=torch.device("cpu"))
    


```python
import matplotlib.pyplot as plt
for images, masks, pred in zip(imgs, ground_Truth, predictions):

    # Display the image, ground truth mask, and prediction
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(images, cmap="gray")
    ax[0].set_title("Image")
    ax[0].axis("off")
    
    ax[1].imshow(masks, cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")
    
    ax[2].imshow(pred, cmap="gray")
    ax[2].set_title("Prediction")
    ax[2].axis("off")
    
    plt.show()
```


    
![png](predictions_files/predictions_1_0.png)
    



    
![png](predictions_files/predictions_1_1.png)
    



    
![png](predictions_files/predictions_1_2.png)
    



    
![png](predictions_files/predictions_1_3.png)
    



    
![png](predictions_files/predictions_1_4.png)
    



    
![png](predictions_files/predictions_1_5.png)
    



    
![png](predictions_files/predictions_1_6.png)
    



    
![png](predictions_files/predictions_1_7.png)
    



    
![png](predictions_files/predictions_1_8.png)
    



    
![png](predictions_files/predictions_1_9.png)
    



    
![png](predictions_files/predictions_1_10.png)
    



    
![png](predictions_files/predictions_1_11.png)
    



    
![png](predictions_files/predictions_1_12.png)
    



    
![png](predictions_files/predictions_1_13.png)
    



    
![png](predictions_files/predictions_1_14.png)
    



    
![png](predictions_files/predictions_1_15.png)
    



    
![png](predictions_files/predictions_1_16.png)
    



    
![png](predictions_files/predictions_1_17.png)
    



    
![png](predictions_files/predictions_1_18.png)
    



    
![png](predictions_files/predictions_1_19.png)
    



    
![png](predictions_files/predictions_1_20.png)
    



    
![png](predictions_files/predictions_1_21.png)
    



    
![png](predictions_files/predictions_1_22.png)
    



    
![png](predictions_files/predictions_1_23.png)
    



    
![png](predictions_files/predictions_1_24.png)
    



    
![png](predictions_files/predictions_1_25.png)
    



    
![png](predictions_files/predictions_1_26.png)
    



    
![png](predictions_files/predictions_1_27.png)
    



    
![png](predictions_files/predictions_1_28.png)
    



    
![png](predictions_files/predictions_1_29.png)
    



    
![png](predictions_files/predictions_1_30.png)
    



    
![png](predictions_files/predictions_1_31.png)
    



    
![png](predictions_files/predictions_1_32.png)
    



    
![png](predictions_files/predictions_1_33.png)
    



    
![png](predictions_files/predictions_1_34.png)
    



    
![png](predictions_files/predictions_1_35.png)
    



    
![png](predictions_files/predictions_1_36.png)
    



    
![png](predictions_files/predictions_1_37.png)
    



    
![png](predictions_files/predictions_1_38.png)
    



    
![png](predictions_files/predictions_1_39.png)
    



    
![png](predictions_files/predictions_1_40.png)
    



    
![png](predictions_files/predictions_1_41.png)
    



    
![png](predictions_files/predictions_1_42.png)
    



    
![png](predictions_files/predictions_1_43.png)
    



    
![png](predictions_files/predictions_1_44.png)
    



    
![png](predictions_files/predictions_1_45.png)
    



    
![png](predictions_files/predictions_1_46.png)
    



    
![png](predictions_files/predictions_1_47.png)
    



    
![png](predictions_files/predictions_1_48.png)
    



    
![png](predictions_files/predictions_1_49.png)
    



    
![png](predictions_files/predictions_1_50.png)
    



    
![png](predictions_files/predictions_1_51.png)
    



    
![png](predictions_files/predictions_1_52.png)
    



    
![png](predictions_files/predictions_1_53.png)
    



    
![png](predictions_files/predictions_1_54.png)
    



    
![png](predictions_files/predictions_1_55.png)
    



    
![png](predictions_files/predictions_1_56.png)
    



    
![png](predictions_files/predictions_1_57.png)
    



    
![png](predictions_files/predictions_1_58.png)
    



    
![png](predictions_files/predictions_1_59.png)
    



    
![png](predictions_files/predictions_1_60.png)
    



    
![png](predictions_files/predictions_1_61.png)
    



    
![png](predictions_files/predictions_1_62.png)
    



    
![png](predictions_files/predictions_1_63.png)
    



    
![png](predictions_files/predictions_1_64.png)
    



    
![png](predictions_files/predictions_1_65.png)
    



    
![png](predictions_files/predictions_1_66.png)
    



    
![png](predictions_files/predictions_1_67.png)
    



    
![png](predictions_files/predictions_1_68.png)
    



    
![png](predictions_files/predictions_1_69.png)
    



    
![png](predictions_files/predictions_1_70.png)
    



    
![png](predictions_files/predictions_1_71.png)
    



    
![png](predictions_files/predictions_1_72.png)
    



    
![png](predictions_files/predictions_1_73.png)
    



    
![png](predictions_files/predictions_1_74.png)
    



    
![png](predictions_files/predictions_1_75.png)
    



    
![png](predictions_files/predictions_1_76.png)
    



    
![png](predictions_files/predictions_1_77.png)
    



    
![png](predictions_files/predictions_1_78.png)
    



    
![png](predictions_files/predictions_1_79.png)
    



    
![png](predictions_files/predictions_1_80.png)
    



    
![png](predictions_files/predictions_1_81.png)
    



    
![png](predictions_files/predictions_1_82.png)
    



    
![png](predictions_files/predictions_1_83.png)
    



    
![png](predictions_files/predictions_1_84.png)
    



    
![png](predictions_files/predictions_1_85.png)
    



    
![png](predictions_files/predictions_1_86.png)
    



    
![png](predictions_files/predictions_1_87.png)
    



    
![png](predictions_files/predictions_1_88.png)
    



    
![png](predictions_files/predictions_1_89.png)
    



    
![png](predictions_files/predictions_1_90.png)
    



    
![png](predictions_files/predictions_1_91.png)
    



    
![png](predictions_files/predictions_1_92.png)
    



    
![png](predictions_files/predictions_1_93.png)
    



    
![png](predictions_files/predictions_1_94.png)
    



    
![png](predictions_files/predictions_1_95.png)
    



    
![png](predictions_files/predictions_1_96.png)
    



    
![png](predictions_files/predictions_1_97.png)
    



    
![png](predictions_files/predictions_1_98.png)
    



    
![png](predictions_files/predictions_1_99.png)
    



