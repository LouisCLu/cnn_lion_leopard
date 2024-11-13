import os
import glob
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # type: ignore

# 定義資料增強的參數
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 獅子和豹的資料夾路徑
lion_dir = r'C:\Users\Louis\Desktop\git-repos\myenv\data\lion'  # 獅子原始圖片資料夾
leopard_dir = r'C:\Users\Louis\Desktop\git-repos\myenv\data\leopard'  # 豹原始圖片資料夾

# 增強後圖片儲存的資料夾
augmented_lion_dir = r'C:\Users\Louis\Desktop\git-repos\myenv\data_augmented\lion'
augmented_leopard_dir = r'C:\Users\Louis\Desktop\git-repos\myenv\data_augmented\leopardd'

# 創建增強後資料夾，如果不存在的話
if not os.path.exists(augmented_lion_dir):
    os.makedirs(augmented_lion_dir)

if not os.path.exists(augmented_leopard_dir):
    os.makedirs(augmented_leopard_dir)

# 定義增強圖片的函數
def augment_images(image_dir, augmented_dir, prefix, img_size=(224, 224), augment_count=5):
    # 獲取所有圖片路徑
    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    
    for img_path in images:
        # 複製原圖到增強後的資料夾
        shutil.copy(img_path, augmented_dir)
        
        # 加載圖片並轉換為陣列
        img = load_img(img_path, target_size=img_size)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # reshape 為 datagen 所需的格式

        # 生成並儲存增強圖片
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_dir, save_prefix=prefix, save_format='jpg'):
            i += 1
            if i >= augment_count:  # 每張圖片生成 augment_count 張增強圖片
                break

# 增強獅子圖片，並將原圖也複製到增強資料夾
augment_images(lion_dir, augmented_lion_dir, prefix='lion', augment_count=5)

# 增強豹圖片，並將原圖也複製到增強資料夾
augment_images(leopard_dir, augmented_leopard_dir, prefix='leopard', augment_count=5)

print("增強圖片完成！原圖已包含在增強後的資料夾中。")